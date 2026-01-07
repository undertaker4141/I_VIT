#!/usr/bin/env python3
"""
Quick QAT (Quantization-Aware Training) Script for I-ViT
=========================================================

目的: 執行 1 Epoch 的快速校準，取得有效的 scaling factors
用途: 為 Pattern 抽取做準備

與原 quant_train.py 的差異:
- 關閉 Mixup/CutMix
- 只跑 1 Epoch
- 簡化輸出

Usage:
    python quick_qat.py --model deit_tiny --data ../ImageNet --epochs 1
"""

import argparse
import os
import time
import math
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy

from models import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Quick QAT for I-ViT Pattern Extraction")
    
    # Model
    parser.add_argument("--model", default='deit_tiny',
                        choices=['deit_tiny', 'deit_small', 'deit_base'],
                        help="model architecture")
    
    # Data
    parser.add_argument('--data', metavar='DIR', default='../ImageNet',
                        help='path to dataset')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    
    # Training
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs (default: 1 for quick calibration)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (lower for calibration)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    # Output
    parser.add_argument('--output', type=str, default='checkpoints/qat_calibrated.pth',
                        help='output checkpoint path')
    parser.add_argument('--print-freq', default=100, type=int)
    
    # Device
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    
    return parser.parse_args()


def str2model(name):
    """Get model constructor by name"""
    models = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
    }
    return models[name]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def train_one_epoch(args, train_loader, model, criterion, optimizer, epoch, 
                    loss_scaler, device):
    """Train for one epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    model.train()
    unfreeze_model(model)
    
    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters())
        
        # Metrics
        acc1, = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4e} ({losses.avg:.4e})\t'
                  f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})')
    
    return losses.avg, top1.avg


def validate(args, val_loader, model, criterion, device):
    """Validate the model"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    freeze_model(model)
    
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4e} ({losses.avg:.4e})\t'
                      f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                      f'Acc@5 {top5.val:.2f} ({top5.avg:.2f})')
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg


def main():
    args = parse_args()
    
    # Setup
    print("=" * 60)
    print("Quick QAT for I-ViT Pattern Extraction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset (無 Mixup/CutMix)
    print("\nLoading dataset...")
    
    # 使用簡化的 dataloader 設定
    args.data_set = 'IMNET'
    args.nb_classes = 1000
    args.color_jitter = 0.0  # 關閉 color jitter
    args.aa = None  # 關閉 AutoAugment
    args.reprob = 0.0  # 關閉 Random Erase
    args.remode = 'pixel'  # Random Erase mode
    args.recount = 1  # Random Erase count
    args.train_interpolation = 'bilinear'
    args.pin_mem = True
    
    train_loader, val_loader = dataloader(args)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    print(f"\nLoading {args.model} with pretrained weights...")
    model = str2model(args.model)(pretrained=True, num_classes=1000)
    model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Optimizer & Scheduler
    args.opt = 'adamw'
    args.opt_eps = 1e-8
    args.opt_betas = None
    args.momentum = 0.9
    args.sched = 'cosine'
    args.min_lr = args.lr / 10
    args.warmup_lr = args.lr / 100
    args.warmup_epochs = 0
    args.cooldown_epochs = 0
    args.decay_rate = 0.1
    args.patience_epochs = 10
    args.decay_epochs = 30
    args.lr_noise = None
    args.lr_noise_pct = 0.67
    args.lr_noise_std = 1.0
    
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    criterion_val = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting QAT training...")
    best_acc1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*40}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            args, train_loader, model, criterion, optimizer, 
            epoch, loss_scaler, device
        )
        lr_scheduler.step(epoch)
        
        # Validate
        print("\nValidating...")
        val_acc = validate(args, val_loader, model, criterion_val, device)
        
        # Save best
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        
        if is_best:
            print(f"\nSaving best checkpoint (Acc@1: {val_acc:.2f}%)...")
            
            # Collect all quantization parameters
            quant_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'best_acc1': best_acc1,
                'args': args,
            }
            
            # Save scaling factors for each quantization layer
            scaling_factors = {}
            for name, module in model.named_modules():
                if hasattr(module, 'act_scaling_factor'):
                    sf = module.act_scaling_factor
                    if sf is not None and sf.numel() > 0:
                        scaling_factors[f'{name}.act_scaling_factor'] = sf.cpu()
                if hasattr(module, 'fc_scaling_factor'):
                    sf = module.fc_scaling_factor
                    if sf is not None and sf.numel() > 0:
                        scaling_factors[f'{name}.fc_scaling_factor'] = sf.cpu()
                if hasattr(module, 'conv_scaling_factor'):
                    sf = module.conv_scaling_factor
                    if sf is not None and sf.numel() > 0:
                        scaling_factors[f'{name}.conv_scaling_factor'] = sf.cpu()
                if hasattr(module, 'norm_scaling_factor'):
                    sf = module.norm_scaling_factor
                    if sf is not None and sf.numel() > 0:
                        scaling_factors[f'{name}.norm_scaling_factor'] = sf.cpu()
            
            quant_state['scaling_factors'] = scaling_factors
            
            torch.save(quant_state, args.output)
            print(f"Saved to: {args.output}")
    
    # Final save
    print("\n" + "=" * 60)
    print("QAT Training Complete!")
    print(f"Best Acc@1: {best_acc1:.2f}%")
    print(f"Checkpoint saved to: {args.output}")
    print("=" * 60)
    
    # Save final checkpoint
    final_path = str(args.output).replace('.pth', '_final.pth')
    torch.save({
        'model': model.state_dict(),
        'epoch': args.epochs,
        'acc1': val_acc,
    }, final_path)
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
