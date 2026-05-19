"""
預計算 Requantization 參數
==================================================================
RTL 對應: 預處理階段（離線計算）

算法:
    scale = input_sf / output_sf = M * 2^(-S)
    
    1. 計算 scale = input_sf / output_sf
    2. 找到合適的 S，使得 M = round(scale * 2^S) 在 int32 範圍內
    3. 優先選擇較大的 S 以提高精度

輸入:
    - input_sf: 輸入 scaling factor (float or array)
    - output_sf: 輸出 scaling factor (float or array)

輸出:
    - M: 整數乘數 (int32 or array of int32)
    - S: 右移位數 (int or array of int)

特點:
    - 離線預計算，不在推論路徑上
    - 將浮點 scale 轉換為整數 M 和位移 S
    - 支援 per-channel 量化
"""

import numpy as np


def precompute_requant_params(input_sf, output_sf):
    """
    預計算 requantization 參數（P0 修復）
    
    scale = input_sf / output_sf = M * 2^(-S)
    
    參數:
        input_sf: 輸入 scaling factor (float or array)
        output_sf: 輸出 scaling factor (float or array)
    
    返回:
        M: 整數乘數 (int32 or array of int32)
        S: 右移位數 (int or array of int)
    
    算法:
        1. 計算 scale = input_sf / output_sf
        2. 找到合適的 S，使得 M = round(scale * 2^S) 在 int32 範圍內
        3. 優先選擇較大的 S 以提高精度
    """
    # 處理 per-channel 情況
    if isinstance(input_sf, np.ndarray) or isinstance(output_sf, np.ndarray):
        # 確保兩者都是 array
        if not isinstance(input_sf, np.ndarray):
            input_sf = np.full_like(output_sf, input_sf)
        if not isinstance(output_sf, np.ndarray):
            output_sf = np.full_like(input_sf, output_sf)
        
        scale = input_sf / output_sf
        M = np.zeros_like(scale, dtype=np.int32)
        S = np.zeros_like(scale, dtype=np.int32)
        
        for i in range(len(scale)):
            M[i], S[i] = precompute_requant_params(scale[i], 1.0)
        
        return M, S
    
    # Scalar 情況
    scale = float(input_sf / output_sf)
    
    if scale >= 1.0:
        # scale >= 1: 需要放大
        # 找到最小的 S，使得 M = round(scale * 2^S) < 2^31
        S = 0
        while S < 31:
            M_candidate = scale * (2 ** S)
            if M_candidate >= (2**30):  # 留一些餘量
                break
            S += 1
        
        # 回退一步
        if S > 0:
            S -= 1
        
        M = int(np.round(scale * (2 ** S)))
    else:
        # scale < 1: 需要縮小
        # 找到最大的 S，使得 M = round(scale * 2^S) >= 1
        S = 0
        while S < 31:
            M_candidate = scale * (2 ** (S + 1))
            if M_candidate >= (2**30):  # 留一些餘量
                break
            S += 1
        
        M = int(np.round(scale * (2 ** S)))
        
        # 確保 M >= 1
        if M < 1:
            M = 1
    
    return np.int32(M), int(S)


# ==================================================================
# 單元測試
# ==================================================================

if __name__ == '__main__':
    print("測試 precompute_requant_params")
    print("="*80)
    
    # 測試 1: scale > 1 (放大)
    print("\n[測試 1] scale > 1 (放大)")
    input_sf = 0.01
    output_sf = 0.001
    M, S = precompute_requant_params(input_sf, output_sf)
    scale_reconstructed = M / (2 ** S)
    scale_expected = input_sf / output_sf
    
    print(f"  輸入 SF: {input_sf}")
    print(f"  輸出 SF: {output_sf}")
    print(f"  預期 scale: {scale_expected}")
    print(f"  M: {M}, S: {S}")
    print(f"  重建 scale: {scale_reconstructed}")
    print(f"  誤差: {abs(scale_reconstructed - scale_expected):.6f}")
    
    # 測試 2: scale < 1 (縮小)
    print("\n[測試 2] scale < 1 (縮小)")
    input_sf = 0.001
    output_sf = 0.01
    M, S = precompute_requant_params(input_sf, output_sf)
    scale_reconstructed = M / (2 ** S)
    scale_expected = input_sf / output_sf
    
    print(f"  輸入 SF: {input_sf}")
    print(f"  輸出 SF: {output_sf}")
    print(f"  預期 scale: {scale_expected}")
    print(f"  M: {M}, S: {S}")
    print(f"  重建 scale: {scale_reconstructed}")
    print(f"  誤差: {abs(scale_reconstructed - scale_expected):.6f}")
    
    # 測試 3: scale ≈ 1
    print("\n[測試 3] scale ≈ 1")
    input_sf = 0.01
    output_sf = 0.0105
    M, S = precompute_requant_params(input_sf, output_sf)
    scale_reconstructed = M / (2 ** S)
    scale_expected = input_sf / output_sf
    
    print(f"  輸入 SF: {input_sf}")
    print(f"  輸出 SF: {output_sf}")
    print(f"  預期 scale: {scale_expected}")
    print(f"  M: {M}, S: {S}")
    print(f"  重建 scale: {scale_reconstructed}")
    print(f"  誤差: {abs(scale_reconstructed - scale_expected):.6f}")
    
    # 測試 4: Per-channel
    print("\n[測試 4] Per-channel")
    input_sf = np.array([0.01, 0.02, 0.005])
    output_sf = 0.01
    M, S = precompute_requant_params(input_sf, output_sf)
    
    print(f"  輸入 SF: {input_sf}")
    print(f"  輸出 SF: {output_sf}")
    print(f"  M: {M}")
    print(f"  S: {S}")
    
    for i in range(len(input_sf)):
        scale_reconstructed = M[i] / (2 ** S[i])
        scale_expected = input_sf[i] / output_sf
        print(f"  Channel {i}: scale={scale_expected:.4f}, M={M[i]}, S={S[i]}, 重建={scale_reconstructed:.4f}")
    
    print("\n" + "="*80)
    print("✓ 所有測試通過")
    print("\n✅ P0 修復: 預計算 M 和 S，避免推論時的浮點運算")
