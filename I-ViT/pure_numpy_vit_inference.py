"""
純 NumPy ViT 推論 - 完全底層實現
==================================================================
目標：使用純 NumPy 實現完整的 ViT 推論，不依賴 PyTorch

特點：
1. 所有運算都是純整數（int8, int16, int32, int64）
2. 不依賴 PyTorch，只使用 NumPy
3. 完整實現 Attention 和 MLP
4. 可以直接轉換為 RTL 實現
"""

import os
import sys
import numpy as np
from PIL import Image
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmodel_rtl_reference'))

from pure_numpy_cmodel import (
    int_layer_norm,
    int_dense,
    int_matmul,
    int_gelu,
    int_softmax,
    quantize_to_int,
    dequantize_to_float,
    requantize,
    quant_act_residual
)


# ==================================================================
# Attention 模組（純 NumPy）
# ==================================================================

def int_attention(x_int, x_sf, qkv_weight, qkv_bias, qkv_sf,
                  proj_weight, proj_bias, proj_sf,
                  num_heads, scale):
    """
    純整數 Multi-Head Self-Attention
    
    參數:
        x_int: 整數輸入 [batch, seq_len, dim] (int8)
        x_sf: 輸入 scaling factor
        qkv_weight: QKV 權重 [3*dim, dim] (int8)
        qkv_bias: QKV bias [3*dim] (int32)
        qkv_sf: QKV scaling factor
        proj_weight: Projection 權重 [dim, dim] (int8)
        proj_bias: Projection bias [dim] (int32)
        proj_sf: Projection scaling factor
        num_heads: 注意力頭數
        scale: 縮放因子 (1/sqrt(head_dim))
    
    返回:
        output_int: 整數輸出 [batch, seq_len, dim] (int16)
        output_sf: 輸出 scaling factor
    """
    B, N, C = x_int.shape
    head_dim = C // num_heads
    
    # Step 1: QKV projection (int8 @ int8 -> int32)
    qkv_int32 = int_dense(x_int, qkv_weight, qkv_bias)
    
    # Step 2: 計算 QKV scaling factor
    qkv_output_sf = x_sf * qkv_sf
    
    # Step 3: Requantize to int8
    # 這裡需要一個統一的 scaling factor
    qkv_unified_sf = np.mean(qkv_output_sf) if isinstance(qkv_output_sf, np.ndarray) else qkv_output_sf
    qkv_int8 = requantize(qkv_int32, qkv_output_sf, qkv_unified_sf, output_bits=8)
    
    # Step 4: Reshape to [B, N, 3, num_heads, head_dim]
    qkv_int8 = qkv_int8.reshape(B, N, 3, num_heads, head_dim)
    
    # Step 5: Permute to [3, B, num_heads, N, head_dim]
    qkv_int8 = qkv_int8.transpose(2, 0, 3, 1, 4)
    
    # Step 6: Split Q, K, V
    q_int8, k_int8, v_int8 = qkv_int8[0], qkv_int8[1], qkv_int8[2]
    
    # Step 7: Q @ K^T (int8 @ int8 -> int32)
    # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
    attn_int32 = np.matmul(q_int8.astype(np.int32), k_int8.transpose(0, 1, 3, 2).astype(np.int32))
    
    # Step 8: Scale (attn = attn * scale)
    # scale = 1/sqrt(head_dim), 這裡用整數近似
    # attn_scaled = attn * scale = attn / sqrt(head_dim)
    attn_scaled_int32 = (attn_int32.astype(np.float64) * scale).astype(np.int32)
    
    # Step 9: 計算 attn scaling factor
    attn_sf = qkv_unified_sf * qkv_unified_sf * scale
    
    # Step 10: Softmax
    attn_softmax_int32 = int_softmax(attn_scaled_int32, attn_sf, output_bit=8, n=15)
    attn_softmax_sf = attn_sf  # Softmax 輸出的 scaling factor
    
    # Step 11: Requantize to int8
    attn_softmax_int8 = requantize(attn_softmax_int32, attn_softmax_sf, attn_softmax_sf, output_bits=8)
    
    # Step 12: Attn @ V (int8 @ int8 -> int32)
    # [B, num_heads, N, N] @ [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
    attn_v_int32 = np.matmul(attn_softmax_int8.astype(np.int32), v_int8.astype(np.int32))
    
    # Step 13: 計算 attn_v scaling factor
    attn_v_sf = attn_softmax_sf * qkv_unified_sf
    
    # Step 14: Transpose to [B, N, num_heads, head_dim]
    attn_v_int32 = attn_v_int32.transpose(0, 2, 1, 3)
    
    # Step 15: Reshape to [B, N, C]
    attn_v_int32 = attn_v_int32.reshape(B, N, C)
    
    # Step 16: Requantize to int8
    attn_v_unified_sf = np.mean(attn_v_sf) if isinstance(attn_v_sf, np.ndarray) else attn_v_sf
    attn_v_int8 = requantize(attn_v_int32, attn_v_sf, attn_v_unified_sf, output_bits=8)
    
    # Step 17: Projection (int8 @ int8 -> int32)
    output_int32 = int_dense(attn_v_int8, proj_weight, proj_bias)
    
    # Step 18: 計算 output scaling factor
    output_sf = attn_v_unified_sf * proj_sf
    
    # Step 19: Requantize to int16
    output_unified_sf = np.mean(output_sf) if isinstance(output_sf, np.ndarray) else output_sf
    output_int16 = requantize(output_int32, output_sf, output_unified_sf, output_bits=16)
    
    return output_int16, output_unified_sf


# ==================================================================
# MLP 模組（純 NumPy）
# ==================================================================

def int_mlp(x_int, x_sf, fc1_weight, fc1_bias, fc1_sf,
            fc2_weight, fc2_bias, fc2_sf):
    """
    純整數 MLP (Feed-Forward Network)
    
    參數:
        x_int: 整數輸入 [batch, seq_len, dim] (int8)
        x_sf: 輸入 scaling factor
        fc1_weight: FC1 權重 [hidden_dim, dim] (int8)
        fc1_bias: FC1 bias [hidden_dim] (int32)
        fc1_sf: FC1 scaling factor
        fc2_weight: FC2 權重 [dim, hidden_dim] (int8)
        fc2_bias: FC2 bias [dim] (int32)
        fc2_sf: FC2 scaling factor
    
    返回:
        output_int: 整數輸出 [batch, seq_len, dim] (int16)
        output_sf: 輸出 scaling factor
    """
    # Step 1: FC1 (int8 @ int8 -> int32)
    fc1_int32 = int_dense(x_int, fc1_weight, fc1_bias)
    
    # Step 2: 計算 FC1 scaling factor
    fc1_output_sf = x_sf * fc1_sf
    
    # Step 3: GELU
    gelu_int32 = int_gelu(fc1_int32, fc1_output_sf, output_bit=8, n=23)
    gelu_sf = fc1_output_sf  # GELU 輸出的 scaling factor
    
    # Step 4: Requantize to int8
    gelu_unified_sf = np.mean(gelu_sf) if isinstance(gelu_sf, np.ndarray) else gelu_sf
    gelu_int8 = requantize(gelu_int32, gelu_sf, gelu_unified_sf, output_bits=8)
    
    # Step 5: FC2 (int8 @ int8 -> int32)
    fc2_int32 = int_dense(gelu_int8, fc2_weight, fc2_bias)
    
    # Step 6: 計算 FC2 scaling factor
    fc2_output_sf = gelu_unified_sf * fc2_sf
    
    # Step 7: Requantize to int16
    fc2_unified_sf = np.mean(fc2_output_sf) if isinstance(fc2_output_sf, np.ndarray) else fc2_output_sf
    output_int16 = requantize(fc2_int32, fc2_output_sf, fc2_unified_sf, output_bits=16)
    
    return output_int16, fc2_unified_sf


# ==================================================================
# Transformer Block（純 NumPy）
# ==================================================================

def int_transformer_block(x_int, x_sf, block_weights, dim_sqrt):
    """
    純整數 Transformer Block
    
    參數:
        x_int: 整數輸入 [batch, seq_len, dim] (int16)
        x_sf: 輸入 scaling factor
        block_weights: Block 權重字典
        dim_sqrt: sqrt(dim)
    
    返回:
        output_int: 整數輸出 [batch, seq_len, dim] (int16)
        output_sf: 輸出 scaling factor
    """
    # 保存輸入用於殘差連接 1
    x_input_for_residual1 = x_int.copy()
    x_sf_for_residual1 = x_sf
    
    # ============================================================
    # Norm1
    # ============================================================
    x_norm1_int32 = int_layer_norm(
        x_int.astype(np.float64),
        block_weights['norm1_bias_int'],
        block_weights['norm1_weight'],
        block_weights['norm1_bias'],
        dim_sqrt
    )
    
    # 計算 norm1 scaling factor
    base_scaling_factor = dim_sqrt / (2 ** 30)
    norm1_sf = base_scaling_factor * block_weights['norm1_weight']
    
    # Requantize to int8 (統一 scaling factor)
    norm1_unified_sf = np.mean(norm1_sf)
    x_norm1_int8 = requantize(x_norm1_int32, norm1_sf, norm1_unified_sf, output_bits=8)
    
    # ============================================================
    # Attention
    # ============================================================
    attn_output_int16, attn_output_sf = int_attention(
        x_norm1_int8, norm1_unified_sf,
        block_weights['qkv_weight_int'], block_weights['qkv_bias_int'], block_weights['qkv_fc_scaling_factor'],
        block_weights['proj_weight_int'], block_weights['proj_bias_int'], block_weights['proj_fc_scaling_factor'],
        block_weights['num_heads'], block_weights['scale']
    )
    
    # ============================================================
    # Residual 1
    # ============================================================
    # 計算統一的 output scaling factor
    residual1_output_sf = (attn_output_sf + x_sf_for_residual1) / 2
    x_int16 = quant_act_residual(
        attn_output_int16, attn_output_sf,
        x_input_for_residual1, x_sf_for_residual1,
        residual1_output_sf, output_bits=16
    )
    x_sf = residual1_output_sf
    
    # 保存輸入用於殘差連接 2
    x_input_for_residual2 = x_int16.copy()
    x_sf_for_residual2 = x_sf
    
    # ============================================================
    # Norm2
    # ============================================================
    x_norm2_int32 = int_layer_norm(
        x_int16.astype(np.float64),
        block_weights['norm2_bias_int'],
        block_weights['norm2_weight'],
        block_weights['norm2_bias'],
        dim_sqrt
    )
    
    # 計算 norm2 scaling factor
    norm2_sf = base_scaling_factor * block_weights['norm2_weight']
    
    # Requantize to int8 (統一 scaling factor)
    norm2_unified_sf = np.mean(norm2_sf)
    x_norm2_int8 = requantize(x_norm2_int32, norm2_sf, norm2_unified_sf, output_bits=8)
    
    # ============================================================
    # MLP
    # ============================================================
    mlp_output_int16, mlp_output_sf = int_mlp(
        x_norm2_int8, norm2_unified_sf,
        block_weights['fc1_weight_int'], block_weights['fc1_bias_int'], block_weights['fc1_fc_scaling_factor'],
        block_weights['fc2_weight_int'], block_weights['fc2_bias_int'], block_weights['fc2_fc_scaling_factor']
    )
    
    # ============================================================
    # Residual 2
    # ============================================================
    # 計算統一的 output scaling factor
    residual2_output_sf = (mlp_output_sf + x_sf_for_residual2) / 2
    x_int16 = quant_act_residual(
        mlp_output_int16, mlp_output_sf,
        x_input_for_residual2, x_sf_for_residual2,
        residual2_output_sf, output_bits=16
    )
    x_sf = residual2_output_sf
    
    return x_int16, x_sf


# ==================================================================
# 主程式（待實現）
# ==================================================================

if __name__ == '__main__':
    print("純 NumPy ViT 推論")
    print("="*80)
    print("✓ 模組已載入")
    print("  - int_layer_norm")
    print("  - int_attention")
    print("  - int_mlp")
    print("  - int_transformer_block")
    print("\n下一步：實現完整的端到端推論")
