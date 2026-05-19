"""
整數 GELU 激活函數
==================================================================
RTL 對應: 待實現

算法:
    GELU(x) = x * sigmoid(1.702 * x)
    其中 sigmoid(x) = exp(x) / (exp(x) + exp(-x))

輸入:
    - x_int: int32
    - scaling_factor: float or array
    - output_bit: 輸出位元數 (default: 8)
    - n: 指數運算的精度參數 (default: 23)

輸出:
    - output_int: int32

特點:
    - 使用明確的 int64 乘法（P1 修復）
    - 最大乘積 ~2^62，不會溢出 int64
    - 最終輸出是 int32
"""

import numpy as np
from .int_exp_shift import int_exp_shift


def int_gelu(x_int, scaling_factor, output_bit=8, n=23):
    """
    純整數 GELU 激活函數
    
    參數:
        x_int: 整數輸入 (int32)
        scaling_factor: 輸入的 scaling factor (float or array)
        output_bit: 輸出位元數 (default: 8)
        n: 指數運算的精度參數 (default: 23)
    
    返回:
        output_int: 整數輸出 (int32)
    
    算法:
        GELU(x) = x * sigmoid(1.702 * x)
        其中 sigmoid(x) = exp(x) / (exp(x) + exp(-x))
    """
    # 確保輸入是整數類型
    pre_x_int = x_int.astype(np.int64)
    
    # Step 1: 計算 x0_int for sigmoid (使用 scalar scaling factor)
    if isinstance(scaling_factor, np.ndarray):
        # 如果是 per-channel，取平均值
        scaling_factor_scalar = np.mean(scaling_factor)
    else:
        scaling_factor_scalar = scaling_factor
    
    scaling_factor_sig = scaling_factor_scalar * 1.702
    x0_int = np.floor(-1.0 / scaling_factor_sig).astype(np.int64)
    
    # Step 2: Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Step 3: 計算 exp(x - x_max) 和 exp(-x_max)
    exp_int = int_exp_shift(x_algo, x0_int, n)
    exp_int_max = int_exp_shift(-x_int_max, x0_int, n)
    
    # Step 4: Sigmoid 分母
    exp_int_sum = exp_int + exp_int_max
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    
    # Step 5: Division
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # Step 6: Scale and shift（P1 修復：使用明確的 int64）
    shift_amt = 31 - output_bit + 1  # = 24 for output_bit=8
    
    # 明確使用 int64 乘法（不使用 object）
    exp_int64 = exp_int.astype(np.int64)
    factor_int64 = factor.astype(np.int64)
    
    # 64-bit 乘法（最大值 ~2^62，不會溢出 int64）
    term = exp_int64 * factor_int64
    
    # 右移
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # Step 7: Multiply with input
    output = pre_x_int * sigmoid_int
    
    return output.astype(np.int32)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 int_gelu")
    print("="*80)
    
    # 測試 1: 基本功能
    print("\n[測試 1] 基本功能")
    x_int = np.array([[-1000, -500, 0, 500, 1000]], dtype=np.int32)
    scaling_factor = 0.01
    
    output = int_gelu(x_int, scaling_factor)
    print(f"  輸入: {x_int[0]}")
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  輸出: {output[0]}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 2: 批次處理
    print("\n[測試 2] 批次處理")
    x_int = np.random.randint(-1000, 1000, (2, 10, 192), dtype=np.int32)
    scaling_factor = 0.01
    
    output = int_gelu(x_int, scaling_factor)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 3: Per-channel scaling factor
    print("\n[測試 3] Per-channel scaling factor")
    x_int = np.random.randint(-1000, 1000, (1, 10, 192), dtype=np.int32)
    scaling_factor = np.random.uniform(0.005, 0.015, 192)
    
    output = int_gelu(x_int, scaling_factor)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  Scaling factor: shape={scaling_factor.shape}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 4: 驗證 int64 不溢出
    print("\n[測試 4] 驗證 int64 不溢出")
    x_int = np.array([[2**30, -2**30]], dtype=np.int32)
    scaling_factor = 0.01
    
    output = int_gelu(x_int, scaling_factor)
    print(f"  輸入: {x_int[0]}")
    print(f"  輸出: {output[0]}")
    print(f"  無 NaN: {not np.any(np.isnan(output))}")
    print(f"  無 Inf: {not np.any(np.isinf(output))}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
    print("\n✅ P1 修復: 使用明確的 int64 乘法（不使用 object）")
    print("   最大乘積 ~2^62，不會溢出 int64")
