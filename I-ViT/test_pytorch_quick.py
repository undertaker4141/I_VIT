#!/usr/bin/env python3
"""
PyTorch 量化模型快速測試（1000 張圖片）
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time

print("="*80)
print("PyTorch 量化模型快速測試")
print("="*80)
print()

# Config
imagenet_path = os.path.join('..', 'ImageNet')
if not os.path.exists(os.path.join(imagenet_path, 'val')):
    imagenet_path = os.path.join('ImageNet')
    
batch_size = 32
device = 'cpu'  # WSL 環境用 CPU

print(f"Device: {device}")
print()

# Data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load dataset
print("加載數據集...")
dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform=transform)

# Split into train (80%) and val (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"  Total: {len(dataset)}")
print(f"  Train: {len(train_dataset)}")
print(f"  Val: {len(val_dataset)}")
print()

# Load model
print("加載模型...")
from models.vit_quant import deit_tiny_patch16_224

checkpoint = torch.load('output_gpu/checkpoint.pth', map_location='cpu', weights_only=False)
model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

model = deit_tiny_patch16_224(pretrained=False, num_classes=1000)
ms = model.state_dict()
ld = {k: v for k, v in model_dict.items()
      if k in ms and getattr(v, 'shape', None) == ms[k].shape}
model.load_state_dict(ld, strict=False)
model.eval()

print(f"  模型已加載")
print()

# Test on validation set (1000 samples)
print("測試驗證集（前 1000 張）...")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

correct = 0
total = 0
top5_correct = 0

start_time = time.time()

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if total >= 1000:
            break
            
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        _, predicted = logits.max(1)
        
        # Top-1
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Top-5
        _, top5_pred = logits.topk(5, 1, True, True)
        top5_pred = top5_pred.t()
        top5_correct_batch = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
        top5_correct += top5_correct_batch.any(dim=0).sum().item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  處理 {total}/1000 張...")

elapsed = time.time() - start_time

print()
print("="*80)
print("驗證集結果（1000 張）")
print("="*80)
print()

top1_acc = 100. * correct / total
top5_acc = 100. * top5_correct / total

print(f"Top-1 準確率: {correct}/{total} = {top1_acc:.2f}%")
print(f"Top-5 準確率: {top5_correct}/{total} = {top5_acc:.2f}%")
print(f"推論時間: {elapsed:.1f}s")
print(f"平均每張: {elapsed/total*1000:.1f}ms")
print()

# Test on train set (1000 samples)
print("測試訓練集（前 1000 張）...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if total >= 1000:
            break
            
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        _, predicted = logits.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

train_acc = 100. * correct / total

print()
print("="*80)
print("訓練集結果（1000 張）")
print("="*80)
print()
print(f"Top-1 準確率: {correct}/{total} = {train_acc:.2f}%")
print()

# Save results
print("保存結果...")
os.makedirs('output_pytorch_quant', exist_ok=True)

with open('output_pytorch_quant/quick_test_results.txt', 'w') as f:
    f.write("PyTorch 量化模型快速測試結果\n")
    f.write("="*50 + "\n\n")
    f.write(f"模型: DeiT-Tiny (量化)\n")
    f.write(f"Checkpoint: output_gpu/checkpoint.pth\n\n")
    f.write(f"驗證集 (1000 samples):\n")
    f.write(f"  Top-1 準確率: {top1_acc:.2f}%\n")
    f.write(f"  Top-5 準確率: {top5_acc:.2f}%\n")
    f.write(f"  推論時間: {elapsed:.1f}s\n")
    f.write(f"  平均每張: {elapsed/total*1000:.1f}ms\n\n")
    f.write(f"訓練集 (1000 samples):\n")
    f.write(f"  Top-1 準確率: {train_acc:.2f}%\n")

print("  結果已保存到 output_pytorch_quant/quick_test_results.txt")
print()

print("="*80)
print("總結")
print("="*80)
print()
print(f"✅ PyTorch 量化模型測試完成")
print(f"✅ 驗證集 Top-1: {top1_acc:.2f}%")
print(f"✅ 訓練集 Top-1: {train_acc:.2f}%")
print()

if top1_acc >= 70:
    print("🎉 驗證集準確率達到預期（>= 70%）")
else:
    print("⚠️  驗證集準確率低於預期")

print()
print("="*80)
