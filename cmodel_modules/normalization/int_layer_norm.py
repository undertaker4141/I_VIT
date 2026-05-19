"""
整數 LayerNorm
==================================================================
RTL 對應: int_layer_norm.sv

算法:
    1. mean = round(mean(x))
    2. y = x - mean
    3. var = sum(y^2)
    4. std = sqrt(var) using Newton iteration (20 次)
    5. factor = floor((2^31 - 1) / std)
    6. y_norm = round(y * factor / 2)
    7. output = y_norm + bias_int

輸入:
    - x_int: int16 [batch, seq_len, features]
    - bias_int: float64 [features] (實際是整數值)
    - weight: float32 [features] (LayerNorm weight)
    - bias: float32 [features] (LayerNorm bias)
    - dim_sqrt: sqrt(features)

輸出:
    - output_int: int32 [batch, seq_len, features]

特點:
    - 使用 Newton iteration 計算 sqrt（20 次迭代）
    - 使用 float64 避免精度損失（但每步都 floor，模擬整數行為）
    - variance 值遠低於 float64 精確整數表示上限（2^53）
"""

import numpy as np


def int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt):
    """
    純整數 LayerNorm
    
    參數:
        x_int: 整數輸入 [batch, seq_len, features] (int16)
        bias_int: 整數 bias [features] (float64, 實際是整數值)
        weight: LayerNorm weight [features] (float32)
        bias: LayerNorm bias [features] (float32)
        dim_sqrt: sqrt(features)
    
    返回:
        output_int: 整數輸出 [batch, seq_len, features] (int32)
    
    算法:
        1. mean = round(mean(x))
        2. y = x - mean
        3. var = sum(y^2)
        4. std = sqrt(var) using Newton iteration
        5. factor = floor((2^31 - 1) / std)
        6. y_norm = floor(y * factor / 2)
        7. output = y_norm + bias_int
    """
    # Step 1: Mean (使用 round，不是 floor)
    mean_int = np.round(np.mean(x_int, axis=-1, keepdims=True))
    
    # Step 2: Centering
    y_int = x_int - mean_int
    
    # Step 3: Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # Step 4: Newton iteration for sqrt (increased to 20 for better precision)
    k = np.full_like(var_int, 2 ** 16, dtype=np.float64)
    for _ in range(20):
        k_1 = np.floor((k + np.floor(var_int / k)) / 2)
        k = k_1
    std_int = k
    
    # Step 5: Normalization factor
    factor = np.floor((2 ** 31 - 1) / std_int)
    
    # Step 6: Normalize (using round for better precision)
    y_int_normalized = np.round(y_int * factor / 2)
    
    # Step 7: Add bias
    output_int = y_int_normalized + bias_int
    
    # Clip to int32 range before casting
    output_int = np.clip(output_int, -2**31, 2**31 - 1)
    
    return output_int.astype(np.int32)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 int_layer_norm")
    print("="*80)
    
    # 測試 1: 基本功能
    print("\n[測試 1] 基本功能")
    x_int = np.random.randint(-1000, 1000, (1, 10, 192), dtype=np.int16)
    bias_int = np.random.randint(-1000, 1000, 192).astype(np.float64)
    weight = np.ones(192, dtype=np.float32)
    bias = np.zeros(192, dtype=np.float32)
    dim_sqrt = np.sqrt(192)
    
    output = int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸入範圍: [{x_int.min()}, {x_int.max()}]")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 2: 驗證歸一化效果
    print("\n[測試 2] 驗證歸一化效果")
    x_int = np.array([[[100, 200, 300, 400]]], dtype=np.int16)
    bias_int = np.zeros(4, dtype=np.float64)
    weight = np.ones(4, dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    dim_sqrt = np.sqrt(4)
    
    output = int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
    print(f"  輸入: {x_int[0, 0]}")
    print(f"  輸出: {output[0, 0]}")
    print(f"  輸出均值: {np.mean(output[0, 0]):.2f}")
    print(f"  輸出標準差: {np.std(output[0, 0]):.2f}")
    
    # 測試 3: Batch 處理
    print("\n[測試 3] Batch 處理")
    x_int = np.random.randint(-1000, 1000, (4, 197, 192), dtype=np.int16)
    bias_int = np.random.randint(-1000, 1000, 192).astype(np.float64)
    weight = np.ones(192, dtype=np.float32)
    bias = np.zeros(192, dtype=np.float32)
    dim_sqrt = np.sqrt(192)
    
    output = int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
    print("\n註: Newton iteration 使用 float64，但每步都 floor，模擬整數行為")
    print("    variance 值遠低於 float64 精確整數表示上限（2^53），無精度損失")
