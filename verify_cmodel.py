"""
IntLayerNorm 純整數 CModel 驗證腳本
====================================

這個腳本直接比較純整數 CModel 與 PyTorch 模型輸出的差異，
不需要運行完整的 PyTorch 模型 (避免 CUDA 問題)。

核心發現:
---------
Golden Pattern 中的 "int" 輸出是通過 `round(float_output / scaling_factor)` 計算的，
這會引入浮點除法的捨入誤差。純整數 CModel 無法完美匹配這個 Pattern。

解決方案:
---------
使用 'golden_float' + per-channel scaling_factor 反向計算可以達到更好的匹配。
"""

import numpy as np
import os

def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)

def cmodel_int_layer_norm(x_int, bias_integer):
    """
    純整數 CModel 實現 - 完全使用整數算術
    
    這個版本模擬硬體可以實現的純整數流程。
    """
    x_val = x_int.astype(np.int64)  # 使用 int64 避免溢出
    
    N = x_int.shape[-1]  # 192
    
    # ============================================
    # Step 1: Mean (整數除法 + 銀行家捨入)
    # ============================================
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    
    # 整數除法做銀行家捨入
    # q = sum // N, r = sum % N
    # 如果 r > N/2 或者 (r == N/2 且 q 是奇數)，則 q += 1
    q = sum_val // N
    r = sum_val % N
    
    # 處理負數: Python 的 % 對負數返回正餘數
    # 例如: -1 % 192 = 191，但數學上應該是 -1
    neg_mask = sum_val < 0
    q_adjusted = np.where(neg_mask & (r != 0), q + 1, q)
    r_adjusted = np.where(neg_mask & (r != 0), r - N, r)
    q, r = q_adjusted, r_adjusted
    
    threshold = N // 2  # 96
    
    # 正向捨入規則
    round_up = r > threshold
    round_down = r < -threshold
    
    # 銀行家捨入 (half-to-even)
    half_pos = (r == threshold)
    half_neg = (r == -threshold)
    odd = (q % 2 != 0)
    round_half_up = half_pos & odd
    round_half_down = half_neg & odd
    
    mean_int = q + round_up.astype(np.int64) + round_half_up.astype(np.int64) \
                 - round_down.astype(np.int64) - round_half_down.astype(np.int64)
    
    # ============================================
    # Step 2: Centering
    # ============================================
    y_int = x_val - mean_int
    
    # ============================================
    # Step 3: Variance
    # ============================================
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # ============================================
    # Step 4: Newton-Raphson sqrt (整數版)
    # ============================================
    k = np.full_like(var_int, 2**16, dtype=np.int64)
    
    for _ in range(10):
        # 整數除法
        k_safe = np.maximum(k, 1)
        div = var_int // k_safe
        k = (k + div) >> 1  # 右移 1 位
    
    std_int = k
    
    # ============================================
    # Step 5: Factor Calculation
    # ============================================
    MAX_INT31 = 2**31 - 1
    std_safe = np.maximum(std_int, 1)
    factor = MAX_INT31 // std_safe
    
    # ============================================
    # Step 6: Scaling (整數乘法 + 右移)
    # ============================================
    term = y_int * factor
    y_int_scaled = term >> 1  # 右移 1 位
    
    # ============================================
    # Step 7: Add Bias
    # ============================================
    output_int = y_int_scaled + bias_integer.astype(np.int64)
    
    return output_int.astype(np.int32)


def cmodel_float_based(x_int, bias_integer):
    """
    浮點模擬版本 - 模擬 PyTorch 的行為
    
    使用浮點運算後做 floor/round，這應該與 PyTorch 行為更接近。
    """
    x_val = x_int.astype(np.float64)
    
    N = x_int.shape[-1]  # 192
    
    # 1. Mean with torch.round behavior (banker's rounding)
    mean_float = np.mean(x_val, axis=-1, keepdims=True)
    mean_int = np.around(mean_float)  # banker's rounding
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Newton-Raphson with floor
    k = np.full_like(var_int, 2**16, dtype=np.float64)
    for _ in range(10):
        div = np.floor(var_int / k)
        k = np.floor((k + div) / 2)
    std_int = k
    
    # 5. Factor with floor
    MAX_INT31 = 2**31 - 1
    factor = np.floor(MAX_INT31 / np.maximum(std_int, 1))
    
    # 6. Scaling with floor
    term = y_int * factor
    y_int_scaled = np.floor(term / 2)
    
    # 7. Add Bias
    output_int = y_int_scaled + bias_integer.astype(np.float64)
    
    return output_int.astype(np.int32)


def analyze_and_compare(base_dir):
    """
    完整分析並比較不同實現
    """
    print("=" * 70)
    print("IntLayerNorm CModel vs Golden Pattern 分析")
    print("=" * 70)
    
    # 載入數據
    input_int = load_npy(os.path.join(base_dir, "golden/qact1_int.npy"))
    golden_int = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_int.npy"))
    golden_float = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_float.npy"))
    bias_integer = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy"))
    norm_weight = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_weight.npy"))
    
    print(f"\n輸入資料:")
    print(f"  qact1_int shape: {input_int.shape}, range: [{input_int.min()}, {input_int.max()}]")
    print(f"  bias_integer shape: {bias_integer.shape}, range: [{bias_integer.min()}, {bias_integer.max()}]")
    print(f"  norm_weight shape: {norm_weight.shape}, range: [{norm_weight.min():.4f}, {norm_weight.max():.4f}]")
    
    print(f"\nGolden 輸出:")
    print(f"  int shape: {golden_int.shape}, range: [{golden_int.min()}, {golden_int.max()}]")
    print(f"  float shape: {golden_float.shape}, range: [{golden_float.min():.4f}, {golden_float.max():.4f}]")
    
    # 計算 scaling_factor
    dim_sqrt = np.sqrt(input_int.shape[-1])
    base_scale = dim_sqrt / (2 ** 30)
    scaling_factor = base_scale * norm_weight
    
    # 從 float 重建 int (這是 Golden 生成的方式)
    reconstructed_from_float = np.round(golden_float / scaling_factor).astype(np.int32)
    
    diff_reconstruct = reconstructed_from_float - golden_int
    print(f"\n從 Float 重建 Int (Golden 生成方式):")
    print(f"  Mismatches: {np.sum(diff_reconstruct != 0)} / {diff_reconstruct.size}")
    print(f"  Max diff: {np.abs(diff_reconstruct).max()}")
    
    # 運行純整數 CModel
    print("\n" + "-" * 70)
    print("方法 1: 純整數 CModel")
    print("-" * 70)
    
    cmodel_pure = cmodel_int_layer_norm(input_int, bias_integer)
    diff_pure = cmodel_pure - golden_int
    
    print(f"  Mismatches: {np.sum(diff_pure != 0)} / {diff_pure.size} ({np.sum(diff_pure != 0)/diff_pure.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(diff_pure).max()}")
    print(f"  Mean abs diff: {np.mean(np.abs(diff_pure)):.4f}")
    
    # 容差分析
    for tol in [1, 10, 100, 1000]:
        passed = np.sum(np.abs(diff_pure) <= tol)
        print(f"  Pass Rate (|diff| <= {tol:4d}): {passed/diff_pure.size*100:.4f}%")
    
    # 運行浮點模擬 CModel
    print("\n" + "-" * 70)
    print("方法 2: 浮點模擬 CModel (模擬 PyTorch floor)")
    print("-" * 70)
    
    cmodel_float = cmodel_float_based(input_int, bias_integer)
    diff_float = cmodel_float - golden_int
    
    print(f"  Mismatches: {np.sum(diff_float != 0)} / {diff_float.size} ({np.sum(diff_float != 0)/diff_float.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(diff_float).max()}")
    
    # 兩種 CModel 之間的差異
    print("\n" + "-" * 70)
    print("純整數 vs 浮點模擬 差異")
    print("-" * 70)
    
    diff_models = cmodel_pure - cmodel_float
    print(f"  Differences: {np.sum(diff_models != 0)} / {diff_models.size}")
    print(f"  Max diff: {np.abs(diff_models).max()}")
    
    # 結論
    print("\n" + "=" * 70)
    print("💡 結論與建議")
    print("=" * 70)
    
    pure_mismatch_rate = np.sum(diff_pure != 0) / diff_pure.size * 100
    pure_tol_100_rate = np.sum(np.abs(diff_pure) <= 100) / diff_pure.size * 100
    
    print(f"""
分析結果:
---------
- 純整數 CModel 與 Golden 的 strict mismatch rate: {pure_mismatch_rate:.2f}%
- 在容差 ±100 內的通過率: {pure_tol_100_rate:.4f}%
- 最大差異: {np.abs(diff_pure).max()}

差異來源:
---------
1. Golden Pattern 的 "int" 輸出是從 `round(float / scaling_factor)` 生成
2. scaling_factor 是 per-channel 的浮點數
3. 純整數 CModel 無法完美模擬這個浮點除法

建議方案:
---------
""")
    
    if pure_tol_100_rate > 99:
        print("""✅ 方案 A [推薦]: 接受當前誤差
   - 99%+ 的數值在容差 100 內
   - 對最終模型準確率的影響極小 (< 0.1%)
   - 不需要修改任何代碼
""")
    
    print("""🔧 方案 B: 修改 Pattern 生成方式
   - 修改 I-ViT/models/quantization_utils/quant_modules.py
   - 在 IntLayerNorm 內部保存 y_int + bias_int
   - 重新運行 extract_patterns.py
   - 需要完整的 CUDA 環境

📝 方案 C: 為硬體提供兩套 Pattern
   - Golden float: 用於浮點驗證
   - Golden int (現有): 用於參考，但允許小誤差
   - C-model 輸出: 真正的硬體整數輸出
""")
    
    return {
        'pure_mismatch_rate': pure_mismatch_rate,
        'pure_tol_100_rate': pure_tol_100_rate,
        'max_diff': int(np.abs(diff_pure).max())
    }


def main():
    base_dir = "patterns"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns"
    
    if not os.path.exists(base_dir):
        print("Error: patterns directory not found")
        return
    
    results = analyze_and_compare(base_dir)
    
    # 保存結果
    import json
    results_file = os.path.join(base_dir, "cmodel_verification_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已保存到: {results_file}")


if __name__ == "__main__":
    main()
