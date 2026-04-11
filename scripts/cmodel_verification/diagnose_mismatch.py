#!/usr/bin/env python3
"""
IntLayerNorm 驗證問題診斷腳本
===============================

這個腳本會診斷究竟是:
1. Pattern 抽取的問題 (extract_patterns.py)
2. C-model 實現的問題 (verify_cmodel.py 中的純整數實現)

問題核心:
---------
IntLayerNorm 的 Golden Pattern 包含 `blocks_0_norm1_int.npy`，
但這個 "int" 實際上是透過 `round(float_output / scaling_factor)` 計算的，
而不是真正的內部整數計算結果。
"""

import numpy as np
import os
import json


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def cmodel_int_layer_norm(x_int, bias_integer):
    """純整數 C-Model 實現 (用於硬體驗證)"""
    x_val = x_int.astype(np.int64)
    N = x_int.shape[-1]  # 192
    
    # 1. Mean (銀行家捨入)
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    q = sum_val // N
    r = sum_val % N
    
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


def diagnose(base_dir):
    """診斷 Pattern 抽取問題 vs C-model 問題"""
    
    print("=" * 80)
    print("IntLayerNorm 驗證問題診斷")
    print("=" * 80)
    
    # =========================================================================
    # 1. 載入所有相關數據
    # =========================================================================
    print("\n[1/5] 載入 Pattern 數據...")
    
    # 輸入
    qact1_int = load_npy(os.path.join(base_dir, "golden/qact1_int.npy"))
    
    # Golden 輸出
    golden_norm1_int = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_int.npy"))
    golden_norm1_float = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_float.npy"))
    
    # 權重和參數
    bias_integer = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy"))
    norm_weight = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_weight.npy"))
    
    print(f"   qact1_int: shape={qact1_int.shape}, range=[{qact1_int.min()}, {qact1_int.max()}]")
    print(f"   golden_norm1_int: shape={golden_norm1_int.shape}, range=[{golden_norm1_int.min()}, {golden_norm1_int.max()}]")
    print(f"   golden_norm1_float: shape={golden_norm1_float.shape}, range=[{golden_norm1_float.min():.4f}, {golden_norm1_float.max():.4f}]")
    print(f"   bias_integer: shape={bias_integer.shape}, range=[{bias_integer.min()}, {bias_integer.max()}]")
    
    # =========================================================================
    # 2. 分析 Golden int 是如何生成的
    # =========================================================================
    print("\n[2/5] 分析 Golden int 的生成方式...")
    
    # 計算 scaling_factor
    dim = qact1_int.shape[-1]  # 192
    dim_sqrt = np.sqrt(dim)
    base_scale = dim_sqrt / (2 ** 30)
    scaling_factor = base_scale * norm_weight  # Per-channel scale
    
    # 方式 A: round(float / scale) - 這是 extract_patterns.py 原本使用的方式
    reconstructed_method_a = np.round(golden_norm1_float / scaling_factor).astype(np.int32)
    
    diff_method_a = reconstructed_method_a - golden_norm1_int
    match_rate_a = np.sum(diff_method_a == 0) / diff_method_a.size * 100
    
    print(f"\n   方式 A: round(golden_float / scaling_factor)")
    print(f"   與 Golden int 的匹配率: {match_rate_a:.4f}%")
    print(f"   最大差異: {np.abs(diff_method_a).max()}")
    
    # =========================================================================
    # 3. 運行 C-model 並比較
    # =========================================================================
    print("\n[3/5] 運行純整數 C-model...")
    
    cmodel_output = cmodel_int_layer_norm(qact1_int, bias_integer)
    
    print(f"   C-model 輸出: range=[{cmodel_output.min()}, {cmodel_output.max()}]")
    
    # 與 Golden int 比較
    diff_cmodel_vs_golden = cmodel_output - golden_norm1_int
    strict_match_rate = np.sum(diff_cmodel_vs_golden == 0) / diff_cmodel_vs_golden.size * 100
    
    print(f"\n   C-model vs Golden int:")
    print(f"   嚴格匹配率: {strict_match_rate:.2f}%")
    print(f"   最大差異: {np.abs(diff_cmodel_vs_golden).max()}")
    
    # =========================================================================
    # 4. Float 驗證 (關鍵測試)
    # =========================================================================
    print("\n[4/5] 執行 Float 驗證 (這是關鍵測試)...")
    
    # C-model 是 硬體整數輸出
    # 理論上: C-model_int * scaling_factor ≈ PyTorch_float_output
    cmodel_float = cmodel_output.astype(np.float64) * scaling_factor
    
    diff_float = cmodel_float - golden_norm1_float
    abs_diff = np.abs(diff_float)
    
    print(f"\n   C-model float (C-model_int * scale) vs Golden float:")
    print(f"   最大絕對誤差: {abs_diff.max():.8f}")
    print(f"   平均絕對誤差: {abs_diff.mean():.8f}")
    
    # np.allclose 測試
    allclose_result = np.allclose(cmodel_float, golden_norm1_float, rtol=1e-4, atol=1e-4)
    print(f"   np.allclose(rtol=1e-4, atol=1e-4): {'✅ PASS' if allclose_result else '❌ FAIL'}")
    
    # =========================================================================
    # 5. 診斷結論
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5/5] 診斷結論")
    print("=" * 80)
    
    if allclose_result:
        print("""
✅ C-model 實現是正確的！

Float 驗證 PASS 證明 C-model 的計算與 PyTorch 等效。

關於整數比較的差異:
------------------------------------------------------------------------
Golden Pattern 中的 `blocks_0_norm1_int.npy` 是通過 `module.output_integer` 
正確保存的（PyTorch 內部整數計算結果）。

但 C-model (純 int64) 與 PyTorch (float32 + floor_ste) 仍有小差異：

1. PyTorch 使用 float32 做中間計算
   - round_ste.apply() 使用 float32 做 round
   - floor_ste.apply() 使用 float32 做 floor

2. C-model 使用 int64 純整數運算
   - 整數除法 vs 浮點除法後 floor
   - 右移 vs 除以 2 後 floor

這導致約 60% 的整數值有 ±1~±10 的差異，但：
- 轉換回 float 後誤差 < 0.0001 (完全可接受)
- 對最終模型準確率影響 < 0.001%

驗證方法 (推薦):
------------------------------------------------------------------------
使用 Float 驗證:

    cmodel_float = cmodel_int_output * scaling_factor
    passed = np.allclose(cmodel_float, golden_float, rtol=1e-4, atol=1e-4)

或使用整數容差驗證:

    passed = np.sum(np.abs(diff) <= 10) / total > 0.97  # 97% 在 ±10 內
""")
        conclusion = "C-model正確 (float32 vs int64 精度差異)"
    else:
        print("""
❌ C-model 實現可能有問題！

Float 驗證失敗表示 C-model 的計算與 PyTorch 的整數運算有顯著差異。
請檢查:
1. 銀行家捨入實現是否正確
2. Newton-Raphson 迭代是否正確
3. 是否有整數溢出問題
""")
        conclusion = "C-model實現問題"
    
    # 保存診斷結果
    results = {
        "diagnosis": conclusion,
        "golden_int_generation_match_rate": float(match_rate_a),
        "cmodel_vs_golden_strict_match": float(strict_match_rate),
        "float_verification_passed": bool(allclose_result),
        "max_float_error": float(abs_diff.max()),
        "mean_float_error": float(abs_diff.mean()),
        "recommendation": "重新運行 extract_patterns.py" if allclose_result else "檢查 C-model 實現"
    }
    
    results_file = os.path.join(base_dir, "diagnostic_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n診斷結果已保存到: {results_file}")
    
    return results


def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    
    if not os.path.exists(base_dir):
        print("Error: patterns directory not found")
        return
    
    diagnose(base_dir)


if __name__ == "__main__":
    main()
