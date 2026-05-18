"""
PyTorch 整數 C-Model
==================================================================
這個 C-Model 完全匹配 PyTorch IntLayerNorm 的算法，
不包含 TVM 的 int16 溢位模擬。

目的：驗證我們能否重現 PyTorch 的整數輸出。
"""

import numpy as np


def pytorch_int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt):
    """
    完全匹配 PyTorch IntLayerNorm 的算法
    
    參數:
        x_int: 整數輸入 [batch, seq_len, features]
        bias_int: 整數 bias [features]
        weight: LayerNorm weight [features] (用於計算 bias_int，這裡已經給定)
        bias: LayerNorm bias [features] (用於計算 bias_int，這裡已經給定)
        dim_sqrt: sqrt(features)
    
    返回:
        output_integer: 整數輸出 [batch, seq_len, features]
    """
    # Step 1: Mean (使用 round，不是 floor)
    mean_int = np.round(np.mean(x_int, axis=-1, keepdims=True))
    
    # Step 2: Centering (不使用 int16 溢位)
    y_int = x_int - mean_int
    
    # Step 3: Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # Step 4: Newton iteration for sqrt
    k = np.full_like(var_int, 2 ** 16, dtype=np.float64)
    for _ in range(10):
        k_1 = np.floor((k + np.floor(var_int / k)) / 2)
        k = k_1
    std_int = k
    
    # Step 5: Normalization factor
    factor = np.floor((2 ** 31 - 1) / std_int)
    
    # Step 6: Normalize
    y_int_normalized = np.floor(y_int * factor / 2)
    
    # Step 7: Add bias
    output_int = y_int_normalized + bias_int
    
    return output_int.astype(np.int32)


def pytorch_int_dense(x_int, weight_int, bias_int):
    """
    完全匹配 PyTorch QuantLinear 的整數運算
    
    參數:
        x_int: 整數輸入 [batch, ..., in_features]
        weight_int: 整數權重 [out_features, in_features]
        bias_int: 整數 bias [out_features]
    
    返回:
        output_int: 整數輸出 [batch, ..., out_features]
    """
    # 保存原始形狀
    original_shape = x_int.shape
    
    # Reshape 為 2D
    if len(original_shape) > 2:
        x_int = x_int.reshape(-1, original_shape[-1])
    
    # 整數矩陣乘法 (int8 @ int8 -> int32)
    output_int = np.matmul(x_int.astype(np.int32), weight_int.T.astype(np.int32))
    
    # 加上 bias
    if bias_int is not None:
        output_int = output_int + bias_int.astype(np.int32)
    
    # Reshape 回原始形狀
    if len(original_shape) > 2:
        output_int = output_int.reshape(*original_shape[:-1], -1)
    
    return output_int.astype(np.int32)


def pytorch_int_matmul(A_int, B_int):
    """
    整數矩陣乘法
    
    參數:
        A_int: 整數矩陣 A
        B_int: 整數矩陣 B
    
    返回:
        output_int: A @ B
    """
    return np.matmul(A_int.astype(np.int32), B_int.astype(np.int32)).astype(np.int32)


def int_exp_shift_pytorch(x_int, x0_int, n):
    """
    完全匹配 PyTorch IntGELU/IntSoftmax 的指數運算
    
    參數:
        x_int: 整數輸入
        x0_int: floor(-1.0 / scaling_factor) - 必須是負數
        n: 精度參數 (23 for GELU, 15 for Softmax)
    
    返回:
        exp_int: 指數運算結果
    
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


def pytorch_int_gelu(x_int, scaling_factor, output_bit=8, n=23):
    """
    完全匹配 PyTorch IntGELU 的算法
    
    參數:
        x_int: 整數輸入
        scaling_factor: 輸入的 scaling factor
        output_bit: 輸出位元數 (default: 8)
        n: 指數運算的精度參數 (default: 23)
    
    返回:
        output_int: 整數輸出
    
    算法:
        GELU(x) = x * sigmoid(1.702 * x)
        其中 sigmoid(x) = exp(x) / (exp(x) + exp(-x))
    """
    # 確保輸入是整數類型
    pre_x_int = x_int.astype(np.int64)
    
    # Step 1: 計算 x0_int for sigmoid
    scaling_factor_sig = scaling_factor * 1.702
    x0_int = np.floor(-1.0 / scaling_factor_sig).astype(np.int64)
    
    # Step 2: Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Step 3: 計算 exp(x - x_max) 和 exp(-x_max)
    exp_int = int_exp_shift_pytorch(x_algo, x0_int, n)
    exp_int_max = int_exp_shift_pytorch(-x_int_max, x0_int, n)
    
    # Step 4: Sigmoid 分母
    exp_int_sum = exp_int + exp_int_max
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    
    # Step 5: Division
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # Step 6: Scale and shift
    shift_amt = 31 - output_bit + 1  # = 24 for output_bit=8
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # Step 7: Multiply with input
    output = pre_x_int * sigmoid_int
    
    return output.astype(np.int32)


def pytorch_int_softmax(x_int, scaling_factor, output_bit=8, n=15):
    """
    完全匹配 PyTorch IntSoftmax 的算法
    
    參數:
        x_int: 整數輸入
        scaling_factor: 輸入的 scaling factor
        output_bit: 輸出位元數 (default: 8)
        n: 指數運算的精度參數 (default: 15)
    
    返回:
        output_int: 整數輸出
    
    算法:
        Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    # 確保輸入是整數類型
    pre_x_int = x_int.astype(np.int64)
    
    # Step 1: 計算 x0_int
    x0_int = np.floor(-1.0 / scaling_factor).astype(np.int64)
    
    # Step 2: Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Step 3: 計算 exp(x - x_max)
    exp_int = int_exp_shift_pytorch(x_algo, x0_int, n)
    
    # Step 4: Sum
    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    
    # Step 5: Division
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # Step 6: Scale and shift
    shift_amt = 31 - output_bit + 1  # = 24 for output_bit=8
    term = exp_int.astype(object) * factor.astype(object)
    output = (term >> shift_amt).astype(np.int64)
    
    return output.astype(np.int32)


if __name__ == '__main__':
    # 簡單測試
    print("PyTorch 整數 C-Model 測試")
    print("="*80)
    
    # 測試 LayerNorm
    x_int = np.random.randint(-1000, 1000, (1, 10, 192), dtype=np.int32)
    bias_int = np.random.randint(-1000, 1000, 192, dtype=np.int32)
    weight = np.ones(192, dtype=np.float32)
    bias = np.zeros(192, dtype=np.float32)
    dim_sqrt = np.sqrt(192)
    
    output = pytorch_int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
    print(f"LayerNorm 輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 Dense
    x_int = np.random.randint(-128, 127, (1, 10, 192), dtype=np.int8)
    weight_int = np.random.randint(-128, 127, (768, 192), dtype=np.int8)
    bias_int = np.random.randint(-1000, 1000, 768, dtype=np.int32)
    
    output = pytorch_int_dense(x_int, weight_int, bias_int)
    print(f"Dense 輸出範圍: [{output.min()}, {output.max()}]")
    
    print("\n✓ 測試完成")
