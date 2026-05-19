"""
純 NumPy C-Model - 完全底層實現
==================================================================
目標：使用純 NumPy 實現完整的 ViT 推論，不依賴 PyTorch

特點：
1. 所有運算都是純整數（int8, int16, int32；GELU/Softmax 中間計算使用 int64）
2. 不依賴 PyTorch，只使用 NumPy
3. 完全匹配 PyTorch IntLayerNorm/IntGELU/IntSoftmax 的算法
4. 可以直接轉換為 RTL 實現

註：int64 只用於 GELU 和 Softmax 的指數運算中間值，最終輸出仍是 int32。
    詳見 docs/cmodel/INT64_USAGE_EXPLANATION.md

模組：
- LayerNorm: 純整數實現（int32）
- Dense/Linear: int8 @ int8 -> int32
- MatMul: int8 @ int8 -> int32
- GELU: 整數指數運算 + sigmoid（中間值 int64，輸出 int32）
- Softmax: 整數指數運算 + 歸一化（中間值 int64，輸出 int32）
- QuantAct: 量化激活函數（requantization）
"""

import numpy as np


# ==================================================================
# 基礎運算模組
# ==================================================================

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


def int_dense(x_int, weight_int, bias_int):
    """
    純整數 Dense/Linear 層
    
    參數:
        x_int: 整數輸入 [batch, ..., in_features] (int8)
        weight_int: 整數權重 [out_features, in_features] (int8)
        bias_int: 整數 bias [out_features] (int32)
    
    返回:
        output_int: 整數輸出 [batch, ..., out_features] (int32)
    
    算法:
        output = (x @ W^T) + bias
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


def int_matmul(A_int, B_int):
    """
    純整數矩陣乘法
    
    參數:
        A_int: 整數矩陣 A (int8)
        B_int: 整數矩陣 B (int8)
    
    返回:
        output_int: A @ B (int32)
    """
    return np.matmul(A_int.astype(np.int32), B_int.astype(np.int32)).astype(np.int32)


# ==================================================================
# 非線性運算模組
# ==================================================================

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
    
    # Step 6: Scale and shift
    shift_amt = 31 - output_bit + 1  # = 24 for output_bit=8
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # Step 7: Multiply with input
    output = pre_x_int * sigmoid_int
    
    return output.astype(np.int32)


def int_softmax(x_int, scaling_factor, output_bit=8, n=15):
    """
    純整數 Softmax
    
    參數:
        x_int: 整數輸入 (int32)
        scaling_factor: 輸入的 scaling factor (float or array)
        output_bit: 輸出位元數 (default: 8)
        n: 指數運算的精度參數 (default: 15)
    
    返回:
        output_int: 整數輸出 (int32)
    
    算法:
        Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    # 確保輸入是整數類型
    pre_x_int = x_int.astype(np.int64)
    
    # Step 1: 計算 x0_int (使用 scalar scaling factor)
    if isinstance(scaling_factor, np.ndarray):
        # 如果是 per-channel，取平均值
        scaling_factor_scalar = np.mean(scaling_factor)
    else:
        scaling_factor_scalar = scaling_factor
    
    x0_int = np.floor(-1.0 / scaling_factor_scalar).astype(np.int64)
    
    # Step 2: Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Step 3: 計算 exp(x - x_max)
    exp_int = int_exp_shift(x_algo, x0_int, n)
    
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


# ==================================================================
# 量化/反量化模組
# ==================================================================

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


def requantize(x_int, input_sf, output_sf, output_bits=8):
    """
    Requantization: 從一個量化範圍轉換到另一個量化範圍
    
    參數:
        x_int: 整數輸入
        input_sf: 輸入 scaling factor
        output_sf: 輸出 scaling factor
        output_bits: 輸出位元數
    
    返回:
        output_int: 重新量化的整數輸出
    
    算法:
        1. 計算 scale = input_sf / output_sf
        2. output = round(x * scale)
        3. Clip to output range
    """
    # 計算縮放比例
    if isinstance(input_sf, np.ndarray) or isinstance(output_sf, np.ndarray):
        # Per-channel
        scale = input_sf / output_sf
        output = np.zeros_like(x_int, dtype=np.float32)
        for c in range(x_int.shape[-1]):
            if isinstance(scale, np.ndarray):
                output[..., c] = np.round(x_int[..., c].astype(np.float32) * scale[c])
            else:
                output[..., c] = np.round(x_int[..., c].astype(np.float32) * scale)
    else:
        # Scalar
        scale = input_sf / output_sf
        output = np.round(x_int.astype(np.float32) * scale)
    
    # Clip to output range
    if output_bits == 8:
        return np.clip(output, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(output, -32768, 32767).astype(np.int16)
    else:
        return output.astype(np.int32)


def quant_act_residual(x1_int, x1_sf, x2_int, x2_sf, output_sf, output_bits=16):
    """
    QuantAct with Residual: 量化激活函數 + 殘差連接
    
    參數:
        x1_int: 第一個輸入（主路徑）
        x1_sf: 第一個輸入的 scaling factor
        x2_int: 第二個輸入（殘差路徑）
        x2_sf: 第二個輸入的 scaling factor
        output_sf: 輸出 scaling factor
        output_bits: 輸出位元數
    
    返回:
        output_int: 重新量化的整數輸出
    
    算法:
        1. 反量化: x1_float = x1_int * x1_sf, x2_float = x2_int * x2_sf
        2. 相加: y_float = x1_float + x2_float
        3. 量化: y_int = round(y_float / output_sf)
    """
    # 反量化
    x1_float = dequantize_to_float(x1_int, x1_sf)
    x2_float = dequantize_to_float(x2_int, x2_sf)
    
    # 相加
    y_float = x1_float + x2_float
    
    # 量化
    y_int = quantize_to_int(y_float, output_sf, bits=output_bits)
    
    return y_int


# ==================================================================
# 測試程式
# ==================================================================

if __name__ == '__main__':
    print("純 NumPy C-Model 測試")
    print("="*80)
    
    # 測試 LayerNorm
    print("\n[1] 測試 LayerNorm")
    x_int = np.random.randint(-1000, 1000, (1, 10, 192), dtype=np.int16)
    bias_int = np.random.randint(-1000, 1000, 192).astype(np.float64)
    weight = np.ones(192, dtype=np.float32)
    bias = np.zeros(192, dtype=np.float32)
    dim_sqrt = np.sqrt(192)
    
    output = int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 Dense
    print("\n[2] 測試 Dense")
    x_int = np.random.randint(-128, 127, (1, 10, 192), dtype=np.int8)
    weight_int = np.random.randint(-128, 127, (768, 192), dtype=np.int8)
    bias_int = np.random.randint(-1000, 1000, 768, dtype=np.int32)
    
    output = int_dense(x_int, weight_int, bias_int)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  權重: shape={weight_int.shape}, dtype={weight_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 MatMul
    print("\n[3] 測試 MatMul")
    A_int = np.random.randint(-128, 127, (10, 192), dtype=np.int8)
    B_int = np.random.randint(-128, 127, (192, 64), dtype=np.int8)
    
    output = int_matmul(A_int, B_int)
    print(f"  A: shape={A_int.shape}, dtype={A_int.dtype}")
    print(f"  B: shape={B_int.shape}, dtype={B_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 GELU
    print("\n[4] 測試 GELU")
    x_int = np.random.randint(-1000, 1000, (1, 10, 192), dtype=np.int32)
    scaling_factor = 0.01
    
    output = int_gelu(x_int, scaling_factor)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 Softmax
    print("\n[5] 測試 Softmax")
    x_int = np.random.randint(-1000, 1000, (1, 10, 64), dtype=np.int32)
    scaling_factor = 0.01
    
    output = int_softmax(x_int, scaling_factor)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 Requantize
    print("\n[6] 測試 Requantize")
    x_int = np.random.randint(-32768, 32767, (1, 10, 192), dtype=np.int16)
    input_sf = 0.01
    output_sf = 0.001
    
    output = requantize(x_int, input_sf, output_sf, output_bits=8)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 測試 QuantAct Residual
    print("\n[7] 測試 QuantAct Residual")
    x1_int = np.random.randint(-32768, 32767, (1, 10, 192), dtype=np.int16)
    x1_sf = 0.01
    x2_int = np.random.randint(-32768, 32767, (1, 10, 192), dtype=np.int16)
    x2_sf = 0.015
    output_sf = 0.012
    
    output = quant_act_residual(x1_int, x1_sf, x2_int, x2_sf, output_sf)
    print(f"  輸入1: shape={x1_int.shape}, dtype={x1_int.dtype}, sf={x1_sf}")
    print(f"  輸入2: shape={x2_int.shape}, dtype={x2_int.dtype}, sf={x2_sf}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}], sf={output_sf}")
    
    print("\n" + "="*80)
    print("✓ 所有測試完成")
