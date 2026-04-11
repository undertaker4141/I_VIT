#!/usr/bin/env python3
"""簡易 CUDA 與 PyTorch 測試腳本"""

import sys

def test_pytorch():
    print("=" * 50)
    print("PyTorch 環境測試")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安裝")
        return False
    
    # CUDA 測試
    print(f"✓ CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"✓ GPU 數量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    - 記憶體: {props.total_memory / 1024**3:.1f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
        
        # 簡單運算測試
        print("\n執行 GPU 運算測試...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Warm up
        _ = torch.mm(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(100):
            z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"✓ 矩陣乘法 (1000x1000) x100: {elapsed*1000:.2f} ms")
        print(f"✓ GPU 運算正常!")
    else:
        print("✗ CUDA 不可用，將使用 CPU")
        return False
    
    return True


def test_timm():
    print("\n" + "=" * 50)
    print("timm 模型測試")
    print("=" * 50)
    
    try:
        import timm
        print(f"✓ timm 版本: {timm.__version__}")
        
        # 測試載入 DeiT 模型
        print("載入 DeiT-Tiny 模型...")
        model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
        print(f"✓ 模型參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # 測試 forward pass
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass 成功! 輸出 shape: {output.shape}")
        return True
        
    except ImportError:
        print("✗ timm 未安裝")
        return False
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return False


def test_ivit():
    print("\n" + "=" * 50)
    print("I-ViT 量化模組測試")
    print("=" * 50)
    
    try:
        sys.path.insert(0, '/home/undertaker4141/I_VIT/I-ViT')
        from models.quantization_utils.quant_modules import QuantLinear, QuantAct
        print("✓ 量化模組載入成功")
        
        import torch
        # 測試 QuantAct
        quant_act = QuantAct(activation_bit=8)
        x = torch.randn(1, 197, 192)
        if torch.cuda.is_available():
            quant_act = quant_act.cuda()
            x = x.cuda()
        
        output, scale = quant_act(x)
        print(f"✓ QuantAct 測試成功! Scale: {scale.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ I-ViT 模組測試失敗: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 開始測試...\n")
    
    results = {
        "PyTorch + CUDA": test_pytorch(),
        "timm": test_timm(),
        "I-ViT": test_ivit(),
    }
    
    print("\n" + "=" * 50)
    print("測試結果摘要")
    print("=" * 50)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("🎉 所有測試通過!")
        sys.exit(0)
    else:
        print("⚠️  部分測試失敗，請檢查上方錯誤訊息")
        sys.exit(1)
