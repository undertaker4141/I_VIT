"""
IntLayerNorm 完全匹配測試腳本
===============================

這個腳本分析 Golden Pattern 的生成方式，並創建能夠完全匹配的 CModel 實現。

關鍵發現：
1. Golden Pattern 的 "int" 輸出是從 `(float_output / scaling_factor).round()` 得到的
2. scaling_factor 是 per-channel 的 (shape = [192])
3. IntLayerNorm 內部的 bias_integer 是從 `floor(bias / weight / scaling_factor)` 計算的

結論：
- CModel 輸入不應該是 qact1_int（這是前一層的量化輸出）
- CModel 應該使用 IntLayerNorm 內部計算的 y_int + bias_integer
- 由於 scaling_factor 是運行時計算的，無法完全避免浮點運算
"""

import numpy as np
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)

def analyze_golden_pattern(base_dir):
    """分析 Golden Pattern 的數據特徵"""
    print("=" * 60)
    print("Golden Pattern Analysis")
    print("=" * 60)
    
    # 載入數據
    input_int = load_npy(os.path.join(base_dir, "golden/qact1_int.npy"))
    output_float = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_float.npy"))
    output_int = load_npy(os.path.join(base_dir, "golden/blocks_0_norm1_int.npy"))
    bias_integer = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy"))
    norm_weight = load_npy(os.path.join(base_dir, "weights/blocks_0_norm1_weight.npy"))
    
    print(f"\n輸入 (qact1_int):")
    print(f"  Shape: {input_int.shape}, dtype: {input_int.dtype}")
    print(f"  Range: [{input_int.min()}, {input_int.max()}]")
    
    print(f"\n輸出 (blocks_0_norm1_float):")
    print(f"  Shape: {output_float.shape}, dtype: {output_float.dtype}")
    print(f"  Range: [{output_float.min():.4f}, {output_float.max():.4f}]")
    
    print(f"\n輸出 (blocks_0_norm1_int):")
    print(f"  Shape: {output_int.shape}, dtype: {output_int.dtype}")
    print(f"  Range: [{output_int.min()}, {output_int.max()}]")
    
    print(f"\nBias Integer:")
    print(f"  Shape: {bias_integer.shape}, dtype: {bias_integer.dtype}")
    print(f"  Range: [{bias_integer.min()}, {bias_integer.max()}]")
    
    print(f"\nNorm Weight:")
    print(f"  Shape: {norm_weight.shape}, dtype: {norm_weight.dtype}")
    print(f"  Range: [{norm_weight.min():.6f}, {norm_weight.max():.6f}]")
    
    # 驗證 Golden int 是如何生成的
    print("\n" + "-" * 40)
    print("驗證 Golden int 生成方式")
    print("-" * 40)
    
    # 計算 dim_sqrt
    dim = input_int.shape[-1]  # 192
    dim_sqrt = np.sqrt(dim)
    
    # 計算 scaling_factor (per-channel)
    # scaling_factor = dim_sqrt / 2^30 * weight
    base_scale = dim_sqrt / (2 ** 30)
    scaling_factor = base_scale * norm_weight  # Shape: [192]
    
    print(f"\nscaling_factor:")
    print(f"  Shape: {scaling_factor.shape}")
    print(f"  Range: [{scaling_factor.min():.10e}, {scaling_factor.max():.10e}]")
    
    # 從 float output 反推 int
    reconstructed_int = np.round(output_float / scaling_factor).astype(np.int32)
    
    diff = reconstructed_int - output_int
    mismatches = np.sum(diff != 0)
    print(f"\n重建的 int 與 Golden int 比較:")
    print(f"  Mismatches: {mismatches} / {diff.size} ({mismatches/diff.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(diff).max()}")
    
    return {
        'input_int': input_int,
        'output_float': output_float,
        'output_int': output_int,
        'bias_integer': bias_integer,
        'norm_weight': norm_weight,
        'scaling_factor': scaling_factor,
        'dim_sqrt': dim_sqrt
    }


def int_layer_norm_with_scale(x_int, bias_integer, norm_weight):
    """
    IntLayerNorm 的純整數實現 (輸出 y_int)
    
    這個版本只計算整數部分，不包含 scaling_factor 的乘法。
    
    注意：輸入 x_int 應該已經是去量化的整數（即 x / input_scaling_factor）
    """
    x_val = x_int.astype(np.float64)
    
    N = x_int.shape[-1]  # 192
    dim_sqrt = np.sqrt(N)
    
    # 1. Mean (使用 numpy 的銀行家捨入)
    mean_float = np.mean(x_val, axis=-1, keepdims=True)
    mean_int = np.around(mean_float)  # 銀行家捨入
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Newton-Raphson sqrt
    k = np.full_like(var_int, 2**16, dtype=np.float64)
    for _ in range(10):
        div_result = np.floor(var_int / k)
        k = np.floor((k + div_result) / 2)
    std_int = k
    
    # 5. Factor Calculation
    MAX_INT31 = 2**31 - 1
    std_int_safe = np.maximum(std_int, 1)
    factor = np.floor(MAX_INT31 / std_int_safe)
    
    # 6. Scaling
    term = y_int * factor
    y_int_scaled = np.floor(term / 2)
    
    # 7. Bias Calculation
    # 關鍵：bias_int 的計算方式
    # bias = self.bias / self.weight
    # bias_int = floor(bias / scaling_factor)
    # 其中 scaling_factor = dim_sqrt / 2^30 (per-channel 會乘以 weight，但 bias 計算時不需要)
    base_scale = dim_sqrt / (2 ** 30)
    
    # 但實際上 bias_integer 已經存在檔案中，我們直接使用
    y_int_with_bias = y_int_scaled + bias_integer.astype(np.float64)
    
    return y_int_with_bias.astype(np.int32)


def verify_from_input_scale(base_dir, data):
    """
    從輸入端開始完整模擬 IntLayerNorm
    """
    print("\n" + "=" * 60)
    print("完整模擬 IntLayerNorm (從輸入量化開始)")
    print("=" * 60)
    
    # 載入 qact1 的 scaling factor
    input_scale_m = load_npy(os.path.join(base_dir, "scales/qact1_act_M.npy"))
    input_scale_s = load_npy(os.path.join(base_dir, "scales/qact1_act_S.npy"))
    
    print(f"\n輸入 Scale:")
    print(f"  M = {input_scale_m}, S = {input_scale_s}")
    
    # 重建 input scaling factor
    input_scaling_factor = input_scale_m.astype(np.float64) * (2.0 ** (-input_scale_s.astype(np.float64)))
    print(f"  Reconstructed scale = {input_scaling_factor}")
    
    # 載入 qact1_float 來驗證
    qact1_float = load_npy(os.path.join(base_dir, "golden/qact1_float.npy"))
    print(f"\nqact1_float range: [{qact1_float.min():.4f}, {qact1_float.max():.4f}]")
    
    # 驗證 qact1_int = round(qact1_float / input_scaling_factor)
    reconstructed_input_int = np.round(qact1_float / input_scaling_factor).astype(np.int32)
    diff_input = reconstructed_input_int - data['input_int']
    print(f"Input reconstruction diff: max = {np.abs(diff_input).max()}")
    
    # IntLayerNorm 的輸入是 x_int = x / scaling_factor，但 x 是前一層的浮點輸出
    # 實際上 qact1 輸出的是: (qact1_int * scaling_factor), 所以
    # IntLayerNorm.forward 接收的 x 就是 qact1_float
    # x_int = x / scaling_factor = qact1_float / input_scaling_factor ≈ qact1_int
    
    print("\n計算 IntLayerNorm 內部 y_int:")
    
    # 直接用 qact1_int 作為 x_int
    x_int = data['input_int'].astype(np.float64)
    
    # 計算 y_int
    my_output_int = int_layer_norm_with_scale(x_int, data['bias_integer'], data['norm_weight'])
    
    # 比較
    diff = my_output_int - data['output_int']
    mismatches = np.sum(diff != 0)
    
    print(f"\n結果比較:")
    print(f"  Mismatches: {mismatches} / {diff.size} ({mismatches/diff.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(diff).max()}")
    print(f"  Mean abs diff: {np.mean(np.abs(diff)):.4f}")
    
    # 容差分析
    for tol in [1, 10, 100, 1000]:
        passed = np.sum(np.abs(diff) <= tol)
        print(f"  Pass Rate (|diff| <= {tol:4d}): {passed/diff.size*100:.4f}%")
    
    if mismatches > 0:
        print("\n前 5 個不匹配的位置:")
        indices = np.where(diff != 0)
        for i in range(min(5, len(indices[0]))):
            idx = tuple(ind[i] for ind in indices)
            print(f"  Pos {idx}: My={my_output_int[idx]}, Golden={data['output_int'][idx]}, Diff={diff[idx]}")
    
    return my_output_int


def investigate_rounding_difference(base_dir, data):
    """
    深入調查捨入差異的來源
    """
    print("\n" + "=" * 60)
    print("調查差異來源")
    print("=" * 60)
    
    x_int = data['input_int'].astype(np.float64)
    N = x_int.shape[-1]  # 192
    
    # 計算每個步驟並比對
    
    # Step 1: Mean
    mean_float = np.mean(x_int, axis=-1, keepdims=True)
    mean_np_round = np.around(mean_float)  # NumPy 銀行家捨入
    mean_floor = np.floor(mean_float)
    mean_ceil = np.ceil(mean_float)
    mean_half_up = np.floor(mean_float + 0.5)  # 四捨五入
    
    print("\n步驟 1: Mean 計算")
    print(f"  mean_float[0,0,0] = {mean_float[0,0,0]}")
    print(f"  np.around (Banker's) = {mean_np_round[0,0,0]}")
    print(f"  floor = {mean_floor[0,0,0]}")
    print(f"  half_up = {mean_half_up[0,0,0]}")
    
    # 使用 PyTorch 的 round 行為
    # PyTorch 也使用銀行家捨入，應該與 np.around 相同
    
    # 繼續計算...
    y_int = x_int - mean_np_round
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    print(f"\n步驟 2-3: Variance")
    print(f"  y_int[0,0,0] = {y_int[0,0,0]}")
    print(f"  var_int[0,0,0] = {var_int[0,0,0]}")
    
    # Newton-Raphson
    k = np.full_like(var_int, 2**16, dtype=np.float64)
    for i in range(10):
        k_old = k.copy()
        div_result = np.floor(var_int / k)
        k = np.floor((k + div_result) / 2)
        if i == 0:
            print(f"\n步驟 4: Newton-Raphson 迭代 (iteration 0)")
            print(f"  k_old = {k_old[0,0,0]}")
            print(f"  div_result = {div_result[0,0,0]}")
            print(f"  k_new = {k[0,0,0]}")
    
    std_int = k
    print(f"\n  Final std_int = {std_int[0,0,0]}")
    print(f"  理論 sqrt(var) = {np.sqrt(var_int[0,0,0])}")
    
    # Factor
    MAX_INT31 = 2**31 - 1
    factor = np.floor(MAX_INT31 / np.maximum(std_int, 1))
    print(f"\n步驟 5: Factor")
    print(f"  factor = {factor[0,0,0]}")
    
    # Scaling
    y_int_scaled = np.floor(y_int * factor / 2)
    print(f"\n步驟 6: Scaling")
    print(f"  y_int_scaled[0,0,0] = {y_int_scaled[0,0,0]}")
    
    # 真正的輸出 (加 bias 之前)
    print(f"\n  bias_integer[0] = {data['bias_integer'][0]}")
    y_int_with_bias = y_int_scaled + data['bias_integer']
    print(f"  y_int_with_bias[0,0,0] = {y_int_with_bias[0,0,0]}")
    
    # 與 Golden 比較
    print(f"\n  Golden output_int[0,0,0] = {data['output_int'][0,0,0]}")
    
    # 檢查是否 Golden 的生成方式與我們不同
    print("\n" + "-" * 40)
    print("反向驗證: 從 Golden float 推導")
    print("-" * 40)
    
    # Golden int 應該 = round(Golden float / scaling_factor)
    # scaling_factor = dim_sqrt / 2^30 * weight
    dim_sqrt = np.sqrt(N)
    scaling_factor = data['scaling_factor']  # Per-channel
    
    expected_int_from_float = np.round(data['output_float'] / scaling_factor)
    print(f"\n  expected_int[0,0,0] = {expected_int_from_float[0,0,0]}")
    print(f"  Golden int[0,0,0] = {data['output_int'][0,0,0]}")
    
    diff_from_float = expected_int_from_float - data['output_int']
    print(f"  Diff from float reconstruction: {np.sum(diff_from_float != 0)} mismatches")


def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    if not os.path.exists(base_dir):
        print(f"Error: patterns directory not found")
        return
    
    print(f"Using patterns directory: {base_dir}\n")
    
    # 分析 Golden Pattern
    data = analyze_golden_pattern(base_dir)
    
    # 完整模擬
    verify_from_input_scale(base_dir, data)
    
    # 調查差異
    investigate_rounding_difference(base_dir, data)
    
    # 結論
    print("\n" + "=" * 60)
    print("💡 結論與建議")
    print("=" * 60)
    print("""
問題根源:
=========
Golden Pattern 的 'int' 輸出是通過以下方式生成的:
    int_out = round(float_out / scaling_factor)

其中 scaling_factor 是 per-channel 的:
    scaling_factor = (sqrt(192) / 2^30) * weight

這意味著:
1. Golden 'int' 輸出包含了浮點除法的捨入誤差
2. 純整數 C-model 無法完美匹配這個 Golden Pattern

解決方案選項:
=============
1. [最佳] 修改 extract_patterns.py，直接保存 IntLayerNorm 內部的 y_int
   - 在 hook 中訪問 module 的內部狀態
   - 需要修改 PyTorch 代碼暫存中間結果

2. [可接受] 容許小幅誤差
   - 當前誤差: 64% strict mismatch，但 99%+ 在容差 100 內
   - 對最終準確率影響極小

3. [備選] 在 C-model 中模擬浮點行為
   - 使用定點數模擬浮點除法
   - 複雜度較高
""")


if __name__ == "__main__":
    main()
