"""
反量化到浮點
==================================================================
用於將整數轉換為浮點數（測試和驗證用）

算法:
    x_float = x_int * scaling_factor

輸入:
    - x_int: 整數輸入
    - scaling_factor: scaling factor (scalar or per-channel)

輸出:
    - x_float: 浮點輸出

註: 這個函數主要用於測試和驗證，不在推論路徑上
"""

import numpy as np


def dequantize_to_float(x_int, scaling_factor):
    """
    反量化到浮點
    
    參數:
        x_int: 整數輸入
        scaling_factor: scaling factor (scalar or per-channel)
    
    返回:
        x_float: 浮點輸出
    """
    if isinstance(scaling_factor, np.ndarray):
        # Per-channel
        x_float = np.zeros_like(x_int, dtype=np.float32)
        for c in range(x_int.shape[-1]):
            x_float[..., c] = x_int[..., c].astype(np.float32) * scaling_factor[c]
        return x_float
    else:
        # Scalar
        return x_int.astype(np.float32) * scaling_factor


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 dequantize_to_float")
    print("="*80)
    
    # 測試 1: Scalar scaling factor
    print("\n[測試 1] Scalar scaling factor")
    x_int = np.array([10, 50, 100, 150, 200], dtype=np.int8)
    scaling_factor = 0.01
    
    x_float = dequantize_to_float(x_int, scaling_factor)
    print(f"  輸入: {x_int}")
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  輸出: {x_float}")
    
    # 測試 2: Per-channel scaling factor
    print("\n[測試 2] Per-channel scaling factor")
    x_int = np.array([[10, 20], [30, 40]], dtype=np.int8)
    scaling_factor = np.array([0.01, 0.02])
    
    x_float = dequantize_to_float(x_int, scaling_factor)
    print(f"  輸入: {x_int}")
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  輸出: {x_float}")
    
    # 測試 3: 不同整數類型
    print("\n[測試 3] 不同整數類型")
    scaling_factor = 0.01
    
    for dtype in [np.int8, np.int16, np.int32]:
        x_int = np.array([100, 200, 300], dtype=dtype)
        x_float = dequantize_to_float(x_int, scaling_factor)
        print(f"  {dtype.__name__}: {x_int} -> {x_float}")
    
    # 測試 4: 驗證量化-反量化循環
    print("\n[測試 4] 驗證量化-反量化循環")
    from .quantize_to_int import quantize_to_int
    
    x_original = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    scaling_factor = 0.01
    
    x_int = quantize_to_int(x_original, scaling_factor, bits=8)
    x_reconstructed = dequantize_to_float(x_int, scaling_factor)
    
    print(f"  原始: {x_original}")
    print(f"  量化: {x_int}")
    print(f"  重建: {x_reconstructed}")
    print(f"  誤差: {np.abs(x_original - x_reconstructed)}")
    print(f"  最大誤差: {np.max(np.abs(x_original - x_reconstructed)):.6f}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
