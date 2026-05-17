"""
GPU 訓練腳本 - 使用完整 val 數據集訓練 1 epoch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# Import model
from models.vit_quant import deit_tiny_patch16_224

def main():
    print("="*80)
    print("GPU 訓練 - 完整 Val 數據集（1 Epoch）")
    print("="*80)
    print()
    
    # Config
    # ImageNet 路徑 - 使用相對路徑（從 I_VIT 根目錄）
    imagenet_path = os.path.join('..', 'ImageNet')
    # 如果 ImageNet 在其他位置，請修改為絕對路徑，例如:
    # imagenet_path = '/path/to/your/ImageNet'
    
    batch_size = 64  # 降低 batch size 以避免 OOM (原本 128)
    lr = 5e-7
    epochs = 1
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: GPU not available, using CPU (will be slow)")
    print()
    
    print(f"Configuration:")
    print(f"  Dataset: {imagenet_path}/val")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
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
    print("Loading dataset...")
    dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform=transform)
    
    # Split into train (80%) and val (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    num_workers = 4 if device == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=(device=='cuda'))
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print()
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load('checkpoints/qat_calibrated.pth', map_location='cpu', weights_only=False)
    model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    model = deit_tiny_patch16_224(pretrained=False, num_classes=1000)
    ms = model.state_dict()
    ld = {k: v for k, v in model_dict.items()
          if k in ms and getattr(v, 'shape', None) == ms[k].shape}
    model.load_state_dict(ld, strict=False)
    model = model.to(device)
    model.train()
    
    print("  Model loaded and moved to", device)
    print()
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"Starting training for {epochs} epoch(s)...")
    print()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 釋放不需要的 GPU 記憶體
            del outputs, loss
            torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*correct/total:.2f}% "
                      f"Time: {elapsed:.1f}s")
        
        train_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        print()
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Train Acc: {train_acc:.2f}%")
        print()
        
        # Validation
        print("Validating...")
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Val Acc: {val_acc:.2f}%")
        print()
    
    # Save checkpoint
    print("Saving checkpoint...")
    os.makedirs('output_gpu', exist_ok=True)
    
    # Move model back to CPU for saving
    model = model.cpu()
    
    torch.save({
        'model': model.state_dict(),
        'epoch': epochs,
        'train_acc': train_acc,
        'val_acc': val_acc,
    }, 'output_gpu/checkpoint.pth')
    print("  Saved to output_gpu/checkpoint.pth")
    print()
    
    # Save training results
    with open('output_gpu/training_results.txt', 'w') as f:
        f.write(f"Training Results\n")
        f.write(f"================\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {lr}\n\n")
        f.write(f"Train Accuracy: {train_acc:.2f}%\n")
        f.write(f"Val Accuracy: {val_acc:.2f}%\n")
        f.write(f"Training time: {epoch_time:.1f}s\n")
    
    print("="*80)
    print("Training completed!")
    print("="*80)
    print()
    print(f"Results:")
    print(f"  Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")
    print(f"  Time: {epoch_time:.1f}s")
    print()
    print("Next steps:")
    print("  1. cd TVM_benchmark")
    print("  2. python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12")
    print("  3. python generate_calibrated_scales.py --model-path '../output_gpu/checkpoint.pth' --output calibrated_scales_gpu.npy --image '../../data/test_image.JPEG'")
    print("  4. python test_100_images_gpu.py")
    print()

if __name__ == '__main__':
    main()
