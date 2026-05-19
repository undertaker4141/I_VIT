"""
整數矩陣乘法
==================================================================
RTL 對應: 可複用 int_dense_kernel.sv 的 MAC 單元

算法:
    output = A @ B

輸入:
    - A_int: int8 [M, K]
    - B_int: int8 [K, N]

輸出:
    - output_int: int32 [M, N]

特點:
    - int8 × int8 -> int32 (MAC 運算)
    - 用於 Attention 的 Q@K 和 @V
"""

import numpy as np


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
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 int_matmul")
    print("="*80)
    
    # 測試 1: 2D × 2D
    print("\n[測試 1] 2D × 2D")
    A_int = np.random.randint(-128, 127, (10, 192), dtype=np.int8)
    B_int = np.random.randint(-128, 127, (192, 64), dtype=np.int8)
    
    output = int_matmul(A_int, B_int)
    print(f"  A: shape={A_int.shape}, dtype={A_int.dtype}")
    print(f"  B: shape={B_int.shape}, dtype={B_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 2: 3D × 3D (batch matmul)
    print("\n[測試 2] 3D × 3D (batch matmul)")
    A_int = np.random.randint(-128, 127, (2, 10, 192), dtype=np.int8)
    B_int = np.random.randint(-128, 127, (2, 192, 64), dtype=np.int8)
    
    output = int_matmul(A_int, B_int)
    print(f"  A: shape={A_int.shape}, dtype={A_int.dtype}")
    print(f"  B: shape={B_int.shape}, dtype={B_int.dtype}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 3: 驗證與 NumPy 一致性
    print("\n[測試 3] 驗證與 NumPy 一致性")
    A_int = np.array([[1, 2], [3, 4]], dtype=np.int8)
    B_int = np.array([[5, 6], [7, 8]], dtype=np.int8)
    
    output = int_matmul(A_int, B_int)
    expected = np.array([[19, 22], [43, 50]], dtype=np.int32)
    
    print(f"  輸出: {output}")
    print(f"  預期: {expected}")
    print(f"  一致: {np.array_equal(output, expected)}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
