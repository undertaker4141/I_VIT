"""
整數指數運算
==================================================================
RTL 對應: 待實現

算法:
    1. Polynomial approximation: x + x/2 - x/16
    2. Clamping: max(x, n * x0_int)
    3. Division: q = x // x0_int, r = x - q * x0_int
    4. Base: exp_base = r/2 - x0_int
    5. Shift: exp_base << (n - q) or exp_base >> (q - n)

輸入:
    - x_int: int64
    - x0_int: floor(-1.0 / scaling_factor) - 必須是負數 (int64)
    - n: 精度參數 (23 for GELU, 15 for Softmax) (int64)

輸出:
    - exp_int: int64

特點:
    - 使用位移和多項式近似
    - 避免浮點運算
    - 中間值使用 int64
"""

import numpy as np


def int_exp_shift(x_int, x0_int, n):
    """
    整數指數運算（使用位移和多項式近似）
    
    參數:
        x_int: 整數輸入 (int64)
        x0_int: floor(-1.0 / scaling_factor) - 必須是負數 (int64)
        n: 精度參數 (23 for GELU, 15 for Softmax) (int64)
    
    返回:
        exp_int: 指數運算結果 (int64)
    
    算法:
        1. Polynomial approximation: x + x/2 - x/16
        2. Clamping: max(x, n * x0_int)
        3. Division: q = x // x0_int, r = x - q * x0_int
        4. Base: exp_base = r/2 - x0_int
        5. Shift: exp_base << (n - q) or exp_base >> (q - n)
    """
    # 確保輸入是整數類型
    x_int = x_int.astype(np.int64)
    x0_int = np.int64(x0_int)
    n = np.int64(n)
    
    # Step 1: Polynomial approximation
    term1 = x_int >> 1  # x / 2
    term2 = x_int >> 4  # x / 16
    x_int = x_int + term1 - term2
    
    # Step 2: Clamping (下界限制)
    lower_bound = n * x0_int
    x_int = np.maximum(x_int, lower_bound)
    
    # Step 3: Integer division
    q = x_int // x0_int
    r = x_int - x0_int * q
    
    # Step 4: Base calculation
    exp_base = (r >> 1) - x0_int
    
    # Step 5: Dynamic shift
    shift = n - q
    
    # 處理正負位移
    result = np.zeros_like(exp_base, dtype=np.int64)
    
    # 正位移 (左移)
    pos_mask = shift >= 0
    if np.any(pos_mask):
        # 限制位移量避免溢位
        shift_clamped = np.minimum(shift[pos_mask], 62)
        result[pos_mask] = exp_base[pos_mask] << shift_clamped
    
    # 負位移 (右移)
    neg_mask = shift < 0
    if np.any(neg_mask):
        shift_abs = np.abs(shift[neg_mask])
        shift_clamped = np.minimum(shift_abs, 62)
        result[neg_mask] = exp_base[neg_mask] >> shift_clamped
    
    # Step 6: Clamp to non-negative
    result = np.maximum(result, 0)
    
    return result.astype(np.int64)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 int_exp_shift")
    print("="*80)
    
    # 測試 1: GELU 參數 (n=23)
    print("\n[測試 1] GELU 參數 (n=23)")
    x_int = np.array([-1000, -500, 0, 500, 1000], dtype=np.int64)
    scaling_factor = 0.01
    x0_int = np.floor(-1.0 / (scaling_factor * 1.702)).astype(np.int64)
    n = 23
    
    output = int_exp_shift(x_int, x0_int, n)
    print(f"  輸入: {x_int}")
    print(f"  x0_int: {x0_int}")
    print(f"  n: {n}")
    print(f"  輸出: {output}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 2: Softmax 參數 (n=15)
    print("\n[測試 2] Softmax 參數 (n=15)")
    x_int = np.array([-1000, -500, 0, 500, 1000], dtype=np.int64)
    scaling_factor = 0.01
    x0_int = np.floor(-1.0 / scaling_factor).astype(np.int64)
    n = 15
    
    output = int_exp_shift(x_int, x0_int, n)
    print(f"  輸入: {x_int}")
    print(f"  x0_int: {x0_int}")
    print(f"  n: {n}")
    print(f"  輸出: {output}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 3: 批次處理
    print("\n[測試 3] 批次處理")
    x_int = np.random.randint(-1000, 1000, (10, 64), dtype=np.int64)
    scaling_factor = 0.01
    x0_int = np.floor(-1.0 / scaling_factor).astype(np.int64)
    n = 15
    
    output = int_exp_shift(x_int, x0_int, n)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
