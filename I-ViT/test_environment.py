#!/usr/bin/env python3
"""
環境測試腳本
檢查 GPU、PyTorch 和模型是否正確安裝
"""

import sys

def test_environment():
    print("="*60)
    print("環境測試")
    print("="*60)
    print()
    
    # Test 1: PyTorch
    print("Test 1: 檢查 PyTorch...")
    try:
        import torch
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch 未安裝: {e}")
        return False
    
    # Test 2: CUDA
    print("\nTest 2: 檢查 CUDA...")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA 可用")
        print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  ❌ CUDA 不可用")
        print(f"  提示: 請確認已安裝 CUDA 版本的 PyTorch")
        return False
    
    # Test 3: 其他依賴
    print("\nTest 3: 檢查其他依賴...")
    try:
        import torchvision
        print(f"  ✅ torchvision: {torchvision.__version__}")
    except ImportError:
        print(f"  ❌ torchvision 未安裝")
        return False
    
    try:
        import timm
        print(f"  ✅ timm: {timm.__version__}")
    except ImportError:
        print(f"  ❌ timm 未安裝")
        return False
    
    try:
        import numpy
        print(f"  ✅ numpy: {numpy.__version__}")
    except ImportError:
        print(f"  ❌ numpy 未安裝")
        return False
    
    try:
        from PIL import Image
        print(f"  ✅ Pillow 已安裝")
    except ImportError:
        print(f"  ❌ Pillow 未安裝")
        return False
    
    # Test 4: 模型導入
    print("\nTest 4: 檢查模型...")
    try:
        from models.vit_quant import deit_tiny_patch16_224
        print(f"  ✅ 模型導入成功")
    except ImportError as e:
        print(f"  ❌ 模型導入失敗: {e}")
        print(f"  提示: 請確認在 I-ViT 目錄下執行此腳本")
        return False
    
    # Test 5: 檢查 checkpoint
    print("\nTest 5: 檢查預訓練模型...")
    import os
    checkpoint_path = 'checkpoints/qat_calibrated.pth'
    if os.path.exists(checkpoint_path):
        print(f"  ✅ Checkpoint 存在: {checkpoint_path}")
        # 檢查檔案大小
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"  ✅ 檔案大小: {size_mb:.2f} MB")
    else:
        print(f"  ⚠️  Checkpoint 不存在: {checkpoint_path}")
        print(f"  提示: 訓練前需要此檔案")
    
    # Test 6: 檢查 ImageNet
    print("\nTest 6: 檢查 ImageNet 數據集...")
    imagenet_path = '../ImageNet/val'
    if os.path.exists(imagenet_path):
        print(f"  ✅ ImageNet 路徑存在: {imagenet_path}")
        # 計算類別數量
        try:
            num_classes = len([d for d in os.listdir(imagenet_path) 
                             if os.path.isdir(os.path.join(imagenet_path, d))])
            print(f"  ✅ 類別數量: {num_classes}")
            if num_classes != 1000:
                print(f"  ⚠️  預期 1000 個類別，實際 {num_classes} 個")
        except Exception as e:
            print(f"  ⚠️  無法讀取類別: {e}")
    else:
        print(f"  ❌ ImageNet 路徑不存在: {imagenet_path}")
        print(f"  提示: 請確認 ImageNet 在正確位置或修改 train_gpu.py 中的路徑")
        return False
    
    # Test 7: GPU 記憶體測試
    print("\nTest 7: GPU 記憶體測試...")
    try:
        # 創建一個小張量測試 GPU
        x = torch.randn(100, 100).cuda()
        y = x @ x
        print(f"  ✅ GPU 計算測試通過")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ GPU 計算測試失敗: {e}")
        return False
    
    print()
    print("="*60)
    print("✅ 所有測試通過！環境設置正確")
    print("="*60)
    print()
    print("下一步:")
    print("  python train_gpu.py")
    print()
    return True

if __name__ == '__main__':
    success = test_environment()
    sys.exit(0 if success else 1)
