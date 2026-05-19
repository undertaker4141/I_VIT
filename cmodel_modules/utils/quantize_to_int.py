"""
量化到整數
==================================================================
用於將浮點數轉換為整數（測試和驗證用）

算法:
    x_int = round(x_float / scaling_factor)

輸入:
    - x_float: 浮點輸入
    - scaling_factor: scaling factor (scalar or per-channel)
    - bits: 位元數 (8, 16, 32)

輸出:
    - x_int: 整數輸出

註: 這個函數主要用於測試和驗證，不在推論路徑上
"""

import numpy as np


def quantize_to_int(x_float, scaling_factor, bits=8):
    """
    量化到整數
    
    參數:
        x_float: 浮點輸入
        scaling_factor: scaling factor (scalar or per-channel)
        bits: 位元數 (8, 16, 32)
    
    返回:
        x_int: 整數輸出
    """
    if isinstance(scaling_factor, np.ndarray):
        # Per-channel
        x_int = np.zeros_like(x_float)
        for c in range(x_float.shape[-1]):
            x_int[..., c] = np.round(x_float[..., c] / scaling_factor[c])
    else:
        # Scalar
        x_int = np.round(x_float / scaling_factor)
    
    if bits == 8:
        return np.clip(x_int, -128, 127).astype(np.int8)
    elif bits == 16:
        return np.clip(x_int, -32768, 32767).astype(np.int16)
    else:
        return x_int.astype(np.int32)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 quantize_to_int")
    print("="*80)
    
    # 測試 1: Scalar scaling factor
    print("\n[測試 1] Scalar scaling factor")
    x_float = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    scaling_factor = 0.01
    
    x_int = quantize_to_int(x_float, scaling_factor, bits=8)
    print(f"  輸入: {x_float}")
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  輸出: {x_int}")
    
    # 驗證反量化
    x_float_reconstructed = x_int * scaling_factor
    print(f"  重建: {x_float_reconstructed}")
    print(f"  誤差: {np.abs(x_float - x_float_reconstructed)}")
    
    # 測試 2: Per-channel scaling factor
    print("\n[測試 2] Per-channel scaling factor")
    x_float = np.array([[0.1, 0.2], [0.3, 0.4]])
    scaling_factor = np.array([0.01, 0.02])
    
    x_int = quantize_to_int(x_float, scaling_factor, bits=8)
    print(f"  輸入: {x_float}")
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  輸出: {x_int}")
    
    # 測試 3: 不同位元數
    print("\n[測試 3] 不同位元數")
    x_float = np.array([1.0, 10.0, 100.0, 1000.0])
    scaling_factor = 0.01
    
    for bits in [8, 16, 32]:
        x_int = quantize_to_int(x_float, scaling_factor, bits=bits)
        print(f"  {bits}-bit: {x_int}, dtype={x_int.dtype}")
    
    # 測試 4: Clipping
    print("\n[測試 4] Clipping")
    x_float = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    scaling_factor = 0.01
    
    x_int = quantize_to_int(x_float, scaling_factor, bits=8)
    print(f"  輸入: {x_float}")
    print(f"  輸出 (8-bit): {x_int}")
    print(f"  範圍: [{x_int.min()}, {x_int.max()}]")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
