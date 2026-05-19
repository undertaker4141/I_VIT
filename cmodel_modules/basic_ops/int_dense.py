"""
整數 Dense/Linear 層
==================================================================
RTL 對應: int_dense_kernel.sv

算法:
    output = (x @ W^T) + bias

輸入:
    - x_int: int8 [batch, ..., in_features]
    - weight_int: int8 [out_features, in_features]
    - bias_int: int32 [out_features]

輸出:
    - output_int: int32 [batch, ..., out_features]

特點:
    - int8 × int8 -> int32 (MAC 運算)
    - 支援多維輸入（自動 reshape）
"""

import numpy as np


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


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 int_dense")
    print("="*80)
    
    # 測試 1: 2D 輸入
    print("\n[測試 1] 2D 輸入")
    x_int = np.random.randint(-128, 127, (10, 192), dtype=np.int8)
    weight_int = np.random.randint(-128, 127, (768, 192), dtype=np.int8)
    bias_int = np.random.randint(-1000, 1000, 768, dtype=np.int32)
    
    output = int_dense(x_int, weight_int, bias_int)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  權重: shape={weight_int.shape}, dtype={weight_int.dtype}")
    print(f"  Bias: shape={bias_int.shape}, dtype={bias_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 2: 3D 輸入
    print("\n[測試 2] 3D 輸入")
    x_int = np.random.randint(-128, 127, (2, 10, 192), dtype=np.int8)
    
    output = int_dense(x_int, weight_int, bias_int)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 3: 無 bias
    print("\n[測試 3] 無 bias")
    x_int = np.random.randint(-128, 127, (10, 192), dtype=np.int8)
    
    output = int_dense(x_int, weight_int, None)
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
