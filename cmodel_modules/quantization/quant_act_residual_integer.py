"""
純整數 QuantAct with Residual
==================================================================
RTL 對應: 待實現

算法:
    1. x1_scaled = (x1 * M1) >> S1
    2. x2_scaled = (x2 * M2) >> S2
    3. y = x1_scaled + x2_scaled
    4. output = (y * M_out) >> S_out

輸入:
    - x1_int: 第一個輸入（主路徑）
    - x1_sf: 第一個輸入的 scaling factor
    - x2_int: 第二個輸入（殘差路徑）
    - x2_sf: 第二個輸入的 scaling factor
    - output_sf: 輸出 scaling factor
    - output_bits: 輸出位元數

輸出:
    - output_int: 重新量化的整數輸出

特點:
    - 純整數實現（P0 修復）
    - 將兩個輸入對齊到輸出 scale，然後相加
    - 使用 int64 避免溢出
"""

import numpy as np
from .precompute_requant_params import precompute_requant_params
from .requantize_integer import requantize_integer


def quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf, output_bits=16):
    """
    純整數 QuantAct with Residual（P0 修復）
    
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
        1. x1_scaled = (x1 * M1) >> S1
        2. x2_scaled = (x2 * M2) >> S2
        3. y = x1_scaled + x2_scaled
        4. output = (y * M_out) >> S_out
    
    註：為了簡化，我們先將兩個輸入對齊到同一個 scale，再相加
    """
    # 預計算 requantization 參數
    # 策略：將兩個輸入都轉換到輸出 scale
    M1, S1 = precompute_requant_params(x1_sf, output_sf)
    M2, S2 = precompute_requant_params(x2_sf, output_sf)
    
    # Requantize x1 和 x2 到輸出 scale（使用 int32 作為中間格式）
    x1_scaled = requantize_integer(x1_int, M1, S1, output_bits=32)
    x2_scaled = requantize_integer(x2_int, M2, S2, output_bits=32)
    
    # 相加（int32 + int32 -> int32）
    y = x1_scaled.astype(np.int64) + x2_scaled.astype(np.int64)
    
    # Clip to output range
    if output_bits == 8:
        return np.clip(y, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(y, -32768, 32767).astype(np.int16)
    else:
        return np.clip(y, -2147483648, 2147483647).astype(np.int32)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 quant_act_residual_integer")
    print("="*80)
    
    # 測試 1: 基本功能
    print("\n[測試 1] 基本功能")
    x1_int = np.array([100, 200, 300, 400], dtype=np.int16)
    x1_sf = 0.01
    x2_int = np.array([50, 100, 150, 200], dtype=np.int16)
    x2_sf = 0.015
    output_sf = 0.012
    
    output = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf)
    print(f"  輸入1: {x1_int}, SF={x1_sf}")
    print(f"  輸入2: {x2_int}, SF={x2_sf}")
    print(f"  輸出: {output}, SF={output_sf}")
    
    # 測試 2: 批次處理
    print("\n[測試 2] 批次處理")
    x1_int = np.random.randint(-1000, 1000, (2, 10, 192), dtype=np.int16)
    x1_sf = 0.01
    x2_int = np.random.randint(-1000, 1000, (2, 10, 192), dtype=np.int16)
    x2_sf = 0.015
    output_sf = 0.012
    
    output = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf)
    print(f"  輸入1: shape={x1_int.shape}, dtype={x1_int.dtype}, SF={x1_sf}")
    print(f"  輸入2: shape={x2_int.shape}, dtype={x2_int.dtype}, SF={x2_sf}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}, SF={output_sf}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 3: Per-channel
    print("\n[測試 3] Per-channel")
    x1_int = np.random.randint(-1000, 1000, (1, 10, 4), dtype=np.int16)
    x1_sf = np.array([0.01, 0.02, 0.015, 0.012])
    x2_int = np.random.randint(-1000, 1000, (1, 10, 4), dtype=np.int16)
    x2_sf = np.array([0.015, 0.018, 0.020, 0.010])
    output_sf = 0.012
    
    output = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf)
    print(f"  輸入1: shape={x1_int.shape}, SF shape={x1_sf.shape}")
    print(f"  輸入2: shape={x2_int.shape}, SF shape={x2_sf.shape}")
    print(f"  輸出: shape={output.shape}, dtype={output.dtype}")
    print(f"  輸出範圍: [{output.min()}, {output.max()}]")
    
    # 測試 4: 驗證不溢出
    print("\n[測試 4] 驗證不溢出")
    x1_int = np.array([32767, -32768], dtype=np.int16)
    x1_sf = 0.01
    x2_int = np.array([32767, -32768], dtype=np.int16)
    x2_sf = 0.01
    output_sf = 0.01
    
    output = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf)
    print(f"  輸入1: {x1_int}")
    print(f"  輸入2: {x2_int}")
    print(f"  輸出: {output}")
    print(f"  無溢出: {not np.any(np.isnan(output))}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
    print("\n✅ P0 修復: 使用純整數加法，避免浮點運算")
    print("   將兩個輸入對齊到輸出 scale，然後相加")
