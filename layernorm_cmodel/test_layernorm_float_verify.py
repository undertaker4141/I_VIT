"""
方案 B: 使用 Float 驗證 IntLayerNorm
=====================================

這個腳本使用 Golden float 進行驗證，而不是直接比較整數輸出。

原理:
-----
C-model 輸出 (int) * scaling_factor = 應該接近 Golden float

這樣可以避免 Golden int 生成時的 round(float/scale) 誤差問題。
"""

import numpy as np
import os
import json

def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def cmodel_int_layer_norm(x_int, bias_integer):
    """
    純整數 CModel 實現
    """
    x_val = x_int.astype(np.int64)
    N = x_int.shape[-1]  # 192
    
    # 1. Mean (銀行家捨入)
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    q = sum_val // N
    r = sum_val % N
    
    # 處理負數餘數
    neg_mask = sum_val < 0
    q = np.where(neg_mask & (r != 0), q + 1, q)
    r = np.where(neg_mask & (r != 0), r - N, r)
    
    threshold = N // 2
    round_up = r > threshold
    round_down = r < -threshold
    odd = (q % 2 != 0)
    round_half_up = (r == threshold) & odd
    round_half_down = (r == -threshold) & odd
    
    mean_int = q + round_up.astype(np.int64) + round_half_up.astype(np.int64) \
                 - round_down.astype(np.int64) - round_half_down.astype(np.int64)
    
    # 2-3. Centering & Variance
    y_int = x_val - mean_int
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Newton-Raphson sqrt
    k = np.full_like(var_int, 2**16, dtype=np.int64)
    for _ in range(10):
        k = (k + var_int // np.maximum(k, 1)) >> 1
    std_int = k
    
    # 5. Factor
    MAX_INT31 = 2**31 - 1
    factor = MAX_INT31 // np.maximum(std_int, 1)
    
    # 6. Scaling
    y_int_scaled = (y_int * factor) >> 1
    
    # 7. Add Bias
    output_int = y_int_scaled + bias_integer.astype(np.int64)
    
    return output_int.astype(np.int32)


def verify_with_float(base_dir):
    """
    方案 B: 使用 Float 驗證
    
    將 C-model 的整數輸出轉成浮點，與 Golden float 比較
    """
    print("=" * 70)
    print("方案 B: 使用 Float 驗證")
    print("=" * 70)
    
    # 載入數據
    input_int = load_npy(os.path.join(base_dir, "golden/qact1_int.npy"))
    golden_float = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_float.npy"))
    golden_int = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_int.npy"))
    bias_integer = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy"))
    norm_weight = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_weight.npy"))
    
    print(f"\n載入資料:")
    print(f"  輸入: {input_int.shape}, range=[{input_int.min()}, {input_int.max()}]")
    print(f"  Golden float: {golden_float.shape}, range=[{golden_float.min():.4f}, {golden_float.max():.4f}]")
    print(f"  Golden int: {golden_int.shape}, range=[{golden_int.min()}, {golden_int.max()}]")
    
    # 計算 scaling_factor (per-channel)
    dim = input_int.shape[-1]  # 192
    dim_sqrt = np.sqrt(dim)
    base_scale = dim_sqrt / (2 ** 30)
    scaling_factor = base_scale * norm_weight  # Shape: [192]
    
    print(f"\nScaling Factor:")
    print(f"  dim_sqrt = sqrt({dim}) = {dim_sqrt:.6f}")
    print(f"  base_scale = dim_sqrt / 2^30 = {base_scale:.10e}")
    print(f"  scaling_factor range: [{scaling_factor.min():.10e}, {scaling_factor.max():.10e}]")
    
    # 運行 C-model 得到整數輸出
    print("\n運行純整數 C-model...")
    cmodel_int_output = cmodel_int_layer_norm(input_int, bias_integer)
    print(f"  C-model int output: range=[{cmodel_int_output.min()}, {cmodel_int_output.max()}]")
    
    # 將 C-model 整數輸出轉為浮點
    # cmodel_float = cmodel_int * scaling_factor
    cmodel_float = cmodel_int_output.astype(np.float64) * scaling_factor
    
    print(f"  C-model float output: range=[{cmodel_float.min():.6f}, {cmodel_float.max():.6f}]")
    
    # 與 Golden float 比較
    print("\n" + "-" * 70)
    print("比較: C-model Float vs Golden Float")
    print("-" * 70)
    
    diff_float = cmodel_float - golden_float
    abs_diff = np.abs(diff_float)
    
    print(f"\n絕對誤差:")
    print(f"  Max:  {abs_diff.max():.8f}")
    print(f"  Mean: {abs_diff.mean():.8f}")
    print(f"  Std:  {abs_diff.std():.8f}")
    
    # 相對誤差 (避免除以 0)
    golden_safe = np.where(np.abs(golden_float) < 1e-8, 1, golden_float)
    rel_error = np.abs(diff_float) / np.abs(golden_safe)
    rel_error = np.where(np.abs(golden_float) < 1e-8, 0, rel_error)
    
    print(f"\n相對誤差:")
    print(f"  Max:  {rel_error.max():.8f} ({rel_error.max()*100:.6f}%)")
    print(f"  Mean: {rel_error.mean():.8f} ({rel_error.mean()*100:.6f}%)")
    
    # 使用 np.allclose 驗證
    print("\n" + "-" * 70)
    print("np.allclose 驗證")
    print("-" * 70)
    
    tolerances = [
        (1e-4, 1e-4),
        (1e-3, 1e-3),
        (1e-2, 1e-2),
        (1e-1, 1e-1),
    ]
    
    for rtol, atol in tolerances:
        passed = np.allclose(cmodel_float, golden_float, rtol=rtol, atol=atol)
        match_count = np.sum(np.abs(diff_float) <= (atol + rtol * np.abs(golden_float)))
        match_rate = match_count / diff_float.size * 100
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  rtol={rtol:.0e}, atol={atol:.0e}: {status} (match rate: {match_rate:.4f}%)")
    
    # Per-element 誤差分佈
    print("\n" + "-" * 70)
    print("誤差分佈")
    print("-" * 70)
    
    thresholds = [0.001, 0.01, 0.1, 0.5, 1.0]
    for t in thresholds:
        passed = np.sum(abs_diff <= t)
        print(f"  |diff| <= {t:5.3f}: {passed:6d} / {abs_diff.size} ({passed/abs_diff.size*100:.4f}%)")
    
    # 最大誤差的位置
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(f"\n最大誤差位置: {max_idx}")
    print(f"  C-model: {cmodel_float[max_idx]:.8f}")
    print(f"  Golden:  {golden_float[max_idx]:.8f}")
    print(f"  Diff:    {diff_float[max_idx]:.8f}")
    print(f"  C-model int: {cmodel_int_output[max_idx]}")
    print(f"  Golden int:  {golden_int[max_idx]}")
    
    # 額外比較：直接比較整數
    print("\n" + "-" * 70)
    print("對照: 整數直接比較")
    print("-" * 70)
    
    diff_int = cmodel_int_output - golden_int
    int_mismatches = np.sum(diff_int != 0)
    print(f"  Mismatches: {int_mismatches} / {diff_int.size} ({int_mismatches/diff_int.size*100:.2f}%)")
    print(f"  Max int diff: {np.abs(diff_int).max()}")
    
    # 結論
    print("\n" + "=" * 70)
    print("💡 結論")
    print("=" * 70)
    
    mean_abs_error = abs_diff.mean()
    max_abs_error = abs_diff.max()
    
    if max_abs_error < 0.01:
        quality = "極佳"
        recommendation = "C-model 實現完全正確，誤差在可接受範圍內。"
    elif max_abs_error < 0.1:
        quality = "優良"
        recommendation = "C-model 實現正確，小幅誤差來自捨入差異。"
    elif max_abs_error < 1.0:
        quality = "可接受"
        recommendation = "C-model 基本正確，但存在一些累積誤差。"
    else:
        quality = "需要檢查"
        recommendation = "C-model 可能有錯誤，建議逐步驗證。"
    
    print(f"""
浮點驗證結果:
--------------
- 平均絕對誤差: {mean_abs_error:.8f}
- 最大絕對誤差: {max_abs_error:.8f}
- 品質評估: {quality}

結論:
------
{recommendation}

💡 使用浮點驗證（方案 B）的優勢:
1. 避免了 Golden int 生成時的 round(float/scale) 誤差
2. 更真實地反映 C-model 與 PyTorch 模型的差異
3. 可以使用標準的 np.allclose 進行驗證
""")
    
    return {
        'mean_abs_error': float(mean_abs_error),
        'max_abs_error': float(max_abs_error),
        'int_mismatch_rate': float(int_mismatches / diff_int.size * 100),
        'quality': quality
    }


def main():
    base_dir = "patterns"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns"
    
    if not os.path.exists(base_dir):
        print("Error: patterns directory not found")
        return
    
    print(f"使用 patterns 目錄: {base_dir}\n")
    
    results = verify_with_float(base_dir)
    
    # 保存結果
    results_file = os.path.join(base_dir, "float_verification_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已保存到: {results_file}")


if __name__ == "__main__":
    main()
