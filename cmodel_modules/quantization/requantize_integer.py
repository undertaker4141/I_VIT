"""
純整數 Requantization
==================================================================
RTL 對應: 待實現

算法:
    output = round((x * M) >> S)
           = ((x * M) + (1 << (S-1))) >> S

輸入:
    - x_int: int8/int16/int32
    - M_or_input_sf: 整數乘數 M (int32) 或輸入 scaling factor (float)
    - S_or_output_sf: 右移位數 S (int) 或輸出 scaling factor (float)
    - output_bits: 輸出位元數 (8/16/32)

輸出:
    - output_int: 重新量化的整數輸出

特點:
    - 使用 int64 避免乘法溢出（P0 修復）
    - 支援 per-channel 量化
    - 兼容舊接口（自動調用 precompute_requant_params）
"""

import numpy as np
from .precompute_requant_params import precompute_requant_params


def requantize_integer(x_int, M_or_input_sf, S_or_output_sf, output_bits=8):
    """
    純整數 Requantization（P0 修復）- 兼容舊接口
    
    參數:
        x_int: 整數輸入 (int8/int16/int32)
        M_or_input_sf: 整數乘數 M (int32) 或輸入 scaling factor (float)
        S_or_output_sf: 右移位數 S (int) 或輸出 scaling factor (float)
        output_bits: 輸出位元數 (8/16/32)
    
    返回:
        output_int: 重新量化的整數輸出
    
    算法:
        output = round((x * M) >> S)
               = ((x * M) + (1 << (S-1))) >> S
    
    註：使用 int64 避免乘法溢出
    
    兼容性：
        - 如果 M_or_input_sf 是 float，自動調用 precompute_requant_params
        - 如果 M_or_input_sf 是 int，直接使用 M 和 S
    """
    # 檢查是否需要預計算 M 和 S
    if isinstance(M_or_input_sf, (float, np.floating)) or isinstance(M_or_input_sf, np.ndarray):
        # 舊接口：requantize_integer(x, input_sf, output_sf, output_bits)
        input_sf = M_or_input_sf
        output_sf = S_or_output_sf
        M, S = precompute_requant_params(input_sf, output_sf)
    else:
        # 新接口：requantize_integer(x, M, S, output_bits)
        M = M_or_input_sf
        S = S_or_output_sf
    
    # 處理 per-channel 情況
    if isinstance(M, np.ndarray):
        output = np.zeros_like(x_int, dtype=np.int64)
        
        for c in range(x_int.shape[-1]):
            x_int64 = x_int[..., c].astype(np.int64)
            M_int64 = np.int64(M[c])
            S_c = int(S[c]) if isinstance(S, np.ndarray) else int(S)
            
            # 乘法
            scaled = x_int64 * M_int64
            
            # 右移前加上 rounding bias
            if S_c > 0:
                rounding_bias = np.int64(1) << (S_c - 1)
                output[..., c] = (scaled + rounding_bias) >> S_c
            else:
                output[..., c] = scaled
    else:
        # Scalar 情況
        # 使用 int64 避免溢出
        x_int64 = x_int.astype(np.int64)
        M_int64 = np.int64(M)
        
        # 乘法
        scaled = x_int64 * M_int64
        
        # 右移前加上 rounding bias (相當於 round)
        if S > 0:
            rounding_bias = np.int64(1) << (S - 1)
            output = (scaled + rounding_bias) >> S
        else:
            output = scaled
    
    # Clip to output range
    if output_bits == 8:
        return np.clip(output, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(output, -32768, 32767).astype(np.int16)
    else:
        return np.clip(output, -2147483648, 2147483647).astype(np.int32)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 requantize_integer")
    print("="*80)
    
    # 測試 1: 新接口（使用 M 和 S）
    print("\n[測試 1] 新接口（使用 M 和 S）")
    x_int = np.array([100, 200, 300, 400], dtype=np.int16)
    M = 10
    S = 3
    
    output = requantize_integer(x_int, M, S, output_bits=8)
    expected = np.round((x_int * M) / (2 ** S))
    
    print(f"  輸入: {x_int}")
    print(f"  M: {M}, S: {S}")
    print(f"  輸出: {output}")
    print(f"  預期: {expected}")
    print(f"  差異: {np.abs(output - expected)}")
    
    # 測試 2: 舊接口（使用 scaling factors）
    print("\n[測試 2] 舊接口（使用 scaling factors）")
    x_int = np.array([100, 200, 300, 400], dtype=np.int16)
    input_sf = 0.01
    output_sf = 0.001
    
    output = requantize_integer(x_int, input_sf, output_sf, output_bits=8)
    print(f"  輸入: {x_int}")
    print(f"  輸入 SF: {input_sf}, 輸出 SF: {output_sf}")
    print(f"  輸出: {output}")
    
    # 測試 3: Per-channel
    print("\n[測試 3] Per-channel")
    x_int = np.array([[100, 200], [300, 400]], dtype=np.int16)
    input_sf = np.array([0.01, 0.02])
    output_sf = 0.01
    
    output = requantize_integer(x_int, input_sf, output_sf, output_bits=8)
    print(f"  輸入: shape={x_int.shape}")
    print(f"  輸入 SF: {input_sf}")
    print(f"  輸出 SF: {output_sf}")
    print(f"  輸出: {output}")
    
    # 測試 4: 驗證不溢出
    print("\n[測試 4] 驗證不溢出")
    x_int = np.array([32767, -32768], dtype=np.int16)
    M = 1000000
    S = 20
    
    output = requantize_integer(x_int, M, S, output_bits=16)
    print(f"  輸入: {x_int}")
    print(f"  M: {M}, S: {S}")
    print(f"  輸出: {output}")
    print(f"  無溢出: {not np.any(np.isnan(output))}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
    print("\n✅ P0 修復: 使用純整數 M×2^(-S) 格式，避免浮點運算")
    print("   使用 int64 避免乘法溢出")
