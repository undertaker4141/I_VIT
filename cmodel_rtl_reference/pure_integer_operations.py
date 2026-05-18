"""
純整數運算模組
==================================================================
所有運算都使用整數，不使用浮點數（除了最後的 dequantization）

精度規則：
- int8: [-128, 127] - 用於量化後的激活值
- int16: [-32768, 32767] - 用於殘差連接和中間結果
- int32: [-2^31, 2^31-1] - 用於累加和大範圍計算
- int64: 用於防止溢位的中間計算
"""

import numpy as np


# ============================================================
# 1. 量化/反量化
# ============================================================

def quantize_to_int8(x_float, scaling_factor):
    """量化到 int8"""
    x_int = np.round(x_float / scaling_factor).astype(np.int64)
    x_int = np.clip(x_int, -128, 127).astype(np.int8)
    return x_int


def quantize_to_int16(x_float, scaling_factor):
    """量化到 int16"""
    x_int = np.round(x_float / scaling_factor).astype(np.int64)
    x_int = np.clip(x_int, -32768, 32767).astype(np.int16)
    return x_int


def quantize_to_int32(x_float, scaling_factor):
    """量化到 int32（不截斷）"""
    x_int = np.round(x_float / scaling_factor).astype(np.int64)
    # int32 範圍很大，通常不需要 clip
    return x_int.astype(np.int32)


def dequantize(x_int, scaling_factor):
    """反量化"""
    return x_int.astype(np.float32) * scaling_factor


# ============================================================
# 2. 線性層 (Dense)
# ============================================================

def int_dense(x_int8, weight_int8, bias_int32):
    """
    純整數線性層
    
    輸入:
        x_int8: [batch, ..., in_features] int8
        weight_int8: [out_features, in_features] int8
        bias_int32: [out_features] int32
    
    輸出:
        output_int32: [batch, ..., out_features] int32
    
    計算:
        output = (x @ weight.T) + bias
        int8 @ int8 -> int32 (累加防止溢位)
    """
    # 保存原始形狀
    original_shape = x_int8.shape
    
    # Reshape 為 2D
    if len(original_shape) > 2:
        x_int8 = x_int8.reshape(-1, original_shape[-1])
    
    # 整數矩陣乘法 (int8 @ int8 -> int32)
    x_int32 = x_int8.astype(np.int32)
    weight_int32 = weight_int8.astype(np.int32)
    output_int32 = np.matmul(x_int32, weight_int32.T)
    
    # 加上 bias
    if bias_int32 is not None:
        output_int32 = output_int32 + bias_int32.astype(np.int32)
    
    # Reshape 回原始形狀
    if len(original_shape) > 2:
        output_int32 = output_int32.reshape(*original_shape[:-1], -1)
    
    return output_int32


# ============================================================
# 3. 矩陣乘法 (MatMul)
# ============================================================

def int_matmul(A_int, B_int):
    """
    純整數矩陣乘法
    
    輸入:
        A_int: int8 或 int16
        B_int: int8 或 int16
    
    輸出:
        output_int32: int32
    """
    A_int32 = A_int.astype(np.int32)
    B_int32 = B_int.astype(np.int32)
    return np.matmul(A_int32, B_int32).astype(np.int32)


# ============================================================
# 4. 量化激活值 (QuantAct)
# ============================================================

def quantize_activation(x_int, input_scale, output_scale, output_bits=8, identity=None, identity_scale=None):
    """
    量化激活值（模擬 PyTorch QuantAct）
    
    輸入:
        x_int: 整數輸入（可能是 int32）
        input_scale: 輸入 scaling factor（可以是 scalar 或 array）
        output_scale: 輸出 scaling factor（通常是 scalar）
        output_bits: 輸出位元數（8 或 16）
        identity: 殘差連接的輸入（可選）
        identity_scale: 殘差連接的 scaling factor（可選）
    
    輸出:
        output_int: 量化後的整數（int8 或 int16）
    
    計算:
        如果有 identity:
            x_float = x_int * input_scale + identity * identity_scale
        否則:
            x_float = x_int * input_scale
        output_int = round(x_float / output_scale)
    
    純整數版本:
        使用定點數近似 scaling factor
    """
    # 使用 int64 防止溢位
    x_int64 = x_int.astype(np.int64)
    
    # 處理 input_scale（可能是 scalar 或 array）
    if isinstance(input_scale, np.ndarray):
        # Per-channel scaling: 需要逐 channel 處理
        if input_scale.ndim == 1 and input_scale.shape[0] == x_int.shape[-1]:
            # 逐 channel rescale
            x_rescaled = np.zeros_like(x_int64)
            for c in range(x_int.shape[-1]):
                rescale = input_scale[c] / output_scale
                rescale_fixed = np.int64(np.round(rescale * (2**16)))
                x_rescaled[..., c] = (x_int64[..., c] * rescale_fixed) >> 16
        else:
            # Broadcast
            rescale = input_scale / output_scale
            rescale_fixed = np.round(rescale * (2**16)).astype(np.int64)
            x_rescaled = (x_int64 * rescale_fixed) >> 16
    else:
        # Scalar scaling
        rescale = input_scale / output_scale
        rescale_fixed = np.int64(np.round(rescale * (2**16)))
        x_rescaled = (x_int64 * rescale_fixed) >> 16
    
    # 如果有 identity（殘差連接）
    if identity is not None:
        identity_int64 = identity.astype(np.int64)
        
        # 處理 identity_scale
        if isinstance(identity_scale, np.ndarray):
            if identity_scale.ndim == 1 and identity_scale.shape[0] == identity.shape[-1]:
                identity_rescaled = np.zeros_like(identity_int64)
                for c in range(identity.shape[-1]):
                    identity_rescale = identity_scale[c] / output_scale
                    identity_rescale_fixed = np.int64(np.round(identity_rescale * (2**16)))
                    identity_rescaled[..., c] = (identity_int64[..., c] * identity_rescale_fixed) >> 16
            else:
                identity_rescale = identity_scale / output_scale
                identity_rescale_fixed = np.round(identity_rescale * (2**16)).astype(np.int64)
                identity_rescaled = (identity_int64 * identity_rescale_fixed) >> 16
        else:
            identity_rescale = identity_scale / output_scale
            identity_rescale_fixed = np.int64(np.round(identity_rescale * (2**16)))
            identity_rescaled = (identity_int64 * identity_rescale_fixed) >> 16
        
        x_rescaled = x_rescaled + identity_rescaled
    
    # 截斷到目標精度
    if output_bits == 8:
        return np.clip(x_rescaled, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(x_rescaled, -32768, 32767).astype(np.int16)
    else:
        return x_rescaled.astype(np.int32)


# ============================================================
# 5. 殘差連接 (Residual Add)
# ============================================================

def int_residual_add_with_requantize(x1_int, x1_scale, x2_int, x2_scale, output_scale):
    """
    殘差連接 + 重新量化
    
    這是 PyTorch QuantAct 的 identity addition 邏輯
    
    輸入:
        x1_int: 主路徑整數值
        x1_scale: 主路徑 scaling factor
        x2_int: 殘差路徑整數值
        x2_scale: 殘差路徑 scaling factor
        output_scale: 輸出 scaling factor
    
    輸出:
        output_int16: int16 (殘差連接通常使用 int16)
    
    計算:
        x1_float = x1_int * x1_scale
        x2_float = x2_int * x2_scale
        output_float = x1_float + x2_float
        output_int = round(output_float / output_scale)
    
    純整數版本:
        output_int = round(x1_int * (x1_scale / output_scale) + x2_int * (x2_scale / output_scale))
    """
    # 計算 rescale 因子
    rescale_1 = x1_scale / output_scale
    rescale_2 = x2_scale / output_scale
    
    # 使用 int64 防止溢位
    x1_rescaled = (x1_int.astype(np.int64) * np.int64(np.round(rescale_1 * 2**16))) >> 16
    x2_rescaled = (x2_int.astype(np.int64) * np.int64(np.round(rescale_2 * 2**16))) >> 16
    
    # 相加
    output_int64 = x1_rescaled + x2_rescaled
    
    # 截斷到 int16
    output_int16 = np.clip(output_int64, -32768, 32767).astype(np.int16)
    
    return output_int16


# ============================================================
# 5. LayerNorm (已實現)
# ============================================================

def int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt):
    """
    純整數 LayerNorm
    
    輸入:
        x_int: int16 或 int32
        bias_int: int32
        weight: float (用於計算 bias_int，這裡已經給定)
        bias: float (用於計算 bias_int，這裡已經給定)
        dim_sqrt: sqrt(features)
    
    輸出:
        output_int32: int32
    """
    # 使用 float64 保持精度
    x_val = x_int.astype(np.float64)
    
    # 1. Mean
    mean_int = np.round(np.mean(x_val, axis=-1, keepdims=True))
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Newton iteration for sqrt
    k = np.full_like(var_int, 2 ** 16, dtype=np.float64)
    for _ in range(10):
        k_1 = np.floor((k + np.floor(var_int / k)) / 2)
        k = k_1
    std_int = k
    
    # 5. Normalization
    factor = np.floor((2 ** 31 - 1) / std_int)
    y_int_normalized = np.floor(y_int * factor / 2)
    
    # 6. Add bias
    output_int = y_int_normalized + bias_int.astype(np.float64)
    
    return output_int.astype(np.int32)


# ============================================================
# 6. GELU (已實現)
# ============================================================

def int_exp_shift(x_int, x0_int, n):
    """指數運算"""
    x_int = x_int.astype(np.int64)
    x0_int = np.int64(x0_int)
    n = np.int64(n)
    
    # Polynomial approximation
    term1 = x_int >> 1
    term2 = x_int >> 4
    x_int = x_int + term1 - term2
    
    # Clamping
    lower_bound = n * x0_int
    x_int = np.maximum(x_int, lower_bound)
    
    # Division
    q = x_int // x0_int
    r = x_int - x0_int * q
    
    # Base
    exp_base = (r >> 1) - x0_int
    
    # Dynamic shift
    shift = n - q
    result = np.zeros_like(exp_base, dtype=np.int64)
    
    pos_mask = shift >= 0
    if np.any(pos_mask):
        shift_clamped = np.minimum(shift[pos_mask], 62)
        result[pos_mask] = exp_base[pos_mask] << shift_clamped
    
    neg_mask = shift < 0
    if np.any(neg_mask):
        shift_abs = np.abs(shift[neg_mask])
        shift_clamped = np.minimum(shift_abs, 62)
        result[neg_mask] = exp_base[neg_mask] >> shift_clamped
    
    return np.maximum(result, 0).astype(np.int64)


def int_gelu(x_int32, scaling_factor, output_bit=8, n=23):
    """
    純整數 GELU
    
    輸入:
        x_int32: int32
        scaling_factor: float
        output_bit: 8
        n: 23
    
    輸出:
        output_int32: int32
    """
    pre_x_int = x_int32.astype(np.int64)
    
    # 計算 x0_int
    scaling_factor_sig = scaling_factor * 1.702
    x0_int = np.floor(-1.0 / scaling_factor_sig).astype(np.int64)
    
    # Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Exponential
    exp_int = int_exp_shift(x_algo, x0_int, n)
    exp_int_max = int_exp_shift(-x_int_max, x0_int, n)
    
    # Sigmoid
    exp_int_sum = exp_int + exp_int_max
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    
    factor = (2**31 - 1) // exp_int_sum_safe
    
    shift_amt = 31 - output_bit + 1
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # Multiply
    output = pre_x_int * sigmoid_int
    
    return output.astype(np.int32)


# ============================================================
# 7. Softmax (已實現)
# ============================================================

def int_softmax(x_int16, scaling_factor, output_bit=8, n=15):
    """
    純整數 Softmax
    
    輸入:
        x_int16: int16
        scaling_factor: float
        output_bit: 8
        n: 15
    
    輸出:
        output_int16: int16
    """
    pre_x_int = x_int16.astype(np.int64)
    
    # 計算 x0_int
    x0_int = np.floor(-1.0 / scaling_factor).astype(np.int64)
    
    # Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # Exponential
    exp_int = int_exp_shift(x_algo, x0_int, n)
    
    # Sum
    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    
    # Division
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # Scale and shift
    shift_amt = 31 - output_bit + 1
    term = exp_int.astype(object) * factor.astype(object)
    output = (term >> shift_amt).astype(np.int64)
    
    # Softmax 輸出通常是 int16
    return np.clip(output, 0, 32767).astype(np.int16)


# ============================================================
# 測試
# ============================================================

if __name__ == '__main__':
    print("純整數運算模組測試")
    print("="*80)
    
    # 測試 Dense
    print("\n測試 Dense:")
    x_int8 = np.random.randint(-128, 127, (2, 10, 192), dtype=np.int8)
    weight_int8 = np.random.randint(-128, 127, (768, 192), dtype=np.int8)
    bias_int32 = np.random.randint(-1000, 1000, 768, dtype=np.int32)
    
    output = int_dense(x_int8, weight_int8, bias_int32)
    print(f"  輸入: {x_int8.shape} int8")
    print(f"  權重: {weight_int8.shape} int8")
    print(f"  輸出: {output.shape} int32")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 MatMul
    print("\n測試 MatMul:")
    A_int8 = np.random.randint(-128, 127, (2, 3, 64), dtype=np.int8)
    B_int8 = np.random.randint(-128, 127, (2, 64, 197), dtype=np.int8)
    
    output = int_matmul(A_int8, B_int8)
    print(f"  A: {A_int8.shape} int8")
    print(f"  B: {B_int8.shape} int8")
    print(f"  輸出: {output.shape} int32")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 QuantAct
    print("\n測試 QuantAct:")
    x_int32 = np.random.randint(-100000, 100000, (2, 10, 192), dtype=np.int32)
    input_scale = 0.001
    output_scale = 0.01
    
    output = quantize_activation(x_int32, input_scale, output_scale, output_bits=8)
    print(f"  輸入: {x_int32.shape} int32, range=[{x_int32.min()}, {x_int32.max()}]")
    print(f"  輸出: {output.shape} int8, range=[{output.min()}, {output.max()}]")
    
    # 測試 QuantAct with identity
    print("\n測試 QuantAct (with identity):")
    identity_int16 = np.random.randint(-10000, 10000, (2, 10, 192), dtype=np.int16)
    identity_scale = 0.0005
    
    output = quantize_activation(x_int32, input_scale, output_scale, output_bits=16, 
                                 identity=identity_int16, identity_scale=identity_scale)
    print(f"  主路徑: {x_int32.shape} int32")
    print(f"  殘差路徑: {identity_int16.shape} int16")
    print(f"  輸出: {output.shape} int16, range=[{output.min()}, {output.max()}]")
    
    print("\n✓ 所有測試通過")


# ============================================================
# Attention 層
# ============================================================

def int_attention(x_int16, 
                 qkv_weight_int8, qkv_bias_int32, qkv_scale,
                 proj_weight_int8, proj_bias_int32, proj_scale,
                 num_heads, scale,
                 input_scale, qkv_output_scale, attn_output_scale):
    """
    純整數 Attention 層
    
    輸入:
        x_int16: [B, N, C] int16
        qkv_weight_int8: [3*C, C] int8
        qkv_bias_int32: [3*C] int32
        proj_weight_int8: [C, C] int8
        proj_bias_int32: [C] int32
        num_heads: 注意力頭數
        scale: 注意力縮放因子
        各種 scaling factors
    
    輸出:
        output_int16: [B, N, C] int16
    """
    B, N, C = x_int16.shape
    head_dim = C // num_heads
    
    # 1. QKV projection (int16 -> int32)
    qkv_int32 = int_dense(x_int16, qkv_weight_int8, qkv_bias_int32)
    
    # 2. Quantize to int8
    qkv_int8 = quantize_activation(qkv_int32, qkv_scale, qkv_output_scale, output_bits=8)
    
    # 3. Reshape to Q, K, V
    qkv_reshaped = qkv_int8.reshape(B, N, 3, num_heads, head_dim)
    qkv_transposed = np.transpose(qkv_reshaped, (2, 0, 3, 1, 4))  # [3, B, num_heads, N, head_dim]
    q_int8 = qkv_transposed[0]  # [B, num_heads, N, head_dim]
    k_int8 = qkv_transposed[1]
    v_int8 = qkv_transposed[2]
    
    # 4. Q @ K^T (int8 @ int8 -> int32)
    k_transposed = np.transpose(k_int8, (0, 1, 3, 2))  # [B, num_heads, head_dim, N]
    attn_scores_int32 = int_matmul(q_int8, k_transposed)  # [B, num_heads, N, N]
    
    # 5. Scale (乘以 scale 因子)
    scale_fixed = np.int32(np.round(scale * (2**16)))
    attn_scores_int32 = (attn_scores_int32.astype(np.int64) * scale_fixed) >> 16
    attn_scores_int32 = attn_scores_int32.astype(np.int32)
    
    # 6. Quantize to int16
    attn_scores_int16 = quantize_activation(attn_scores_int32, qkv_output_scale * qkv_output_scale * scale,
                                            qkv_output_scale, output_bits=16)
    
    # 7. Softmax (int16 -> int16)
    attn_softmax_int16 = int_softmax(attn_scores_int16, qkv_output_scale)
    
    # 8. Attn @ V (int16 @ int8 -> int32)
    attn_output_int32 = int_matmul(attn_softmax_int16, v_int8)  # [B, num_heads, N, head_dim]
    
    # 9. Transpose and reshape
    attn_output_transposed = np.transpose(attn_output_int32, (0, 2, 1, 3))  # [B, N, num_heads, head_dim]
    attn_output_int32 = attn_output_transposed.reshape(B, N, C)
    
    # 10. Quantize (int32 -> int16)
    attn_output_int16 = quantize_activation(attn_output_int32, qkv_output_scale * qkv_output_scale,
                                           qkv_output_scale, output_bits=16)
    
    # 11. Output projection (int16 -> int32)
    output_int32 = int_dense(attn_output_int16, proj_weight_int8, proj_bias_int32)
    
    # 12. Quantize to int16
    output_int16 = quantize_activation(output_int32, proj_scale, attn_output_scale, output_bits=16)
    
    return output_int16


# ============================================================
# MLP 層
# ============================================================

def int_mlp(x_int8,
           fc1_weight_int8, fc1_bias_int32, fc1_scale,
           fc2_weight_int8, fc2_bias_int32, fc2_scale,
           input_scale, gelu_output_scale, mlp_output_scale):
    """
    純整數 MLP 層
    
    輸入:
        x_int8: [B, N, C] int8
        fc1_weight_int8: [hidden_dim, C] int8
        fc1_bias_int32: [hidden_dim] int32
        fc2_weight_int8: [C, hidden_dim] int8
        fc2_bias_int32: [C] int32
        各種 scaling factors
    
    輸出:
        output_int16: [B, N, C] int16
    """
    # 1. FC1 (int8 -> int32)
    x_int32 = int_dense(x_int8, fc1_weight_int8, fc1_bias_int32)
    
    # 2. GELU (int32 -> int32)
    x_gelu_int32 = int_gelu(x_int32, fc1_scale)
    
    # 3. Quantize (int32 -> int32, 保持 int32 因為 GELU 輸出範圍大)
    # 這裡不需要量化，直接傳遞
    
    # 4. FC2 (int32 -> int32)
    # 需要先量化到合適的範圍
    x_gelu_quantized = quantize_activation(x_gelu_int32, gelu_output_scale, gelu_output_scale, output_bits=32)
    output_int32 = int_dense(x_gelu_quantized.astype(np.int8), fc2_weight_int8, fc2_bias_int32)
    
    # 5. Quantize to int16
    output_int16 = quantize_activation(output_int32, fc2_scale, mlp_output_scale, output_bits=16)
    
    return output_int16


# ============================================================
# Transformer Block
# ============================================================

def int_transformer_block(x_int16, block_params):
    """
    純整數 Transformer Block
    
    輸入:
        x_int16: [B, N, C] int16
        block_params: 包含所有權重和 scaling factors 的字典
    
    輸出:
        output_int16: [B, N, C] int16
    """
    B, N, C = x_int16.shape
    dim_sqrt = np.sqrt(C)
    
    # 保存輸入用於殘差連接
    residual_1_int16 = x_int16.copy()
    
    # 1. Norm1 (int16 -> int32)
    x_norm1_int32 = int_layer_norm(x_int16, 
                                   block_params['norm1_bias_int'],
                                   block_params['norm1_weight'],
                                   block_params['norm1_bias'],
                                   dim_sqrt)
    
    # 2. Quantize to int16
    x_norm1_int16 = quantize_activation(x_norm1_int32, 
                                       block_params['norm1_output_scale'],
                                       block_params['attn_input_scale'],
                                       output_bits=16)
    
    # 3. Attention (int16 -> int16)
    attn_out_int16 = int_attention(x_norm1_int16,
                                   block_params['qkv_weight_int8'],
                                   block_params['qkv_bias_int32'],
                                   block_params['qkv_scale'],
                                   block_params['proj_weight_int8'],
                                   block_params['proj_bias_int32'],
                                   block_params['proj_scale'],
                                   block_params['num_heads'],
                                   block_params['attn_scale'],
                                   block_params['attn_input_scale'],
                                   block_params['qkv_output_scale'],
                                   block_params['attn_output_scale'])
    
    # 4. Residual 1 (int16 + int16 -> int16)
    x_int16 = quantize_activation(attn_out_int16,
                                  block_params['attn_output_scale'],
                                  block_params['residual1_output_scale'],
                                  output_bits=16,
                                  identity=residual_1_int16,
                                  identity_scale=block_params['residual1_input_scale'])
    
    # 保存輸入用於殘差連接
    residual_2_int16 = x_int16.copy()
    
    # 5. Norm2 (int16 -> int32)
    x_norm2_int32 = int_layer_norm(x_int16,
                                   block_params['norm2_bias_int'],
                                   block_params['norm2_weight'],
                                   block_params['norm2_bias'],
                                   dim_sqrt)
    
    # 6. Quantize to int8
    x_norm2_int8 = quantize_activation(x_norm2_int32,
                                      block_params['norm2_output_scale'],
                                      block_params['mlp_input_scale'],
                                      output_bits=8)
    
    # 7. MLP (int8 -> int16)
    mlp_out_int16 = int_mlp(x_norm2_int8,
                           block_params['fc1_weight_int8'],
                           block_params['fc1_bias_int32'],
                           block_params['fc1_scale'],
                           block_params['fc2_weight_int8'],
                           block_params['fc2_bias_int32'],
                           block_params['fc2_scale'],
                           block_params['mlp_input_scale'],
                           block_params['gelu_output_scale'],
                           block_params['mlp_output_scale'])
    
    # 8. Residual 2 (int16 + int16 -> int16)
    x_int16 = quantize_activation(mlp_out_int16,
                                  block_params['mlp_output_scale'],
                                  block_params['residual2_output_scale'],
                                  output_bits=16,
                                  identity=residual_2_int16,
                                  identity_scale=block_params['residual2_input_scale'])
    
    return x_int16
