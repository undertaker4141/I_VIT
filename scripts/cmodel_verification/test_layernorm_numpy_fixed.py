import numpy as np
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)

def int_layer_norm_pytorch_equivalent(x_int, bias_int):
    """
    PyTorch-Equivalent Implementation of IntLayerNorm.
    
    這個版本完全模擬 PyTorch quant_modules.py 的 IntLayerNorm.forward()，
    包括使用浮點運算後做 floor/round。
    
    ⚠️ 重要差異：
    1. PyTorch 使用浮點 mean，再用 torch.round() (銀行家捨入)
    2. PyTorch 使用浮點除法後做 floor_ste (torch.floor)
    3. 所有中間運算在 PyTorch 中是 float32，而非純整數
    """
    
    # 轉換為 float64 模擬 PyTorch 的浮點行為
    x_val = x_int.astype(np.float64)
    bias_val = bias_int.astype(np.float64)
    
    N = x_int.shape[-1]  # 192
    
    # 1. Mean - 使用 PyTorch 的方式：浮點平均後做 round (銀行家捨入)
    # PyTorch: mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
    mean_float = np.mean(x_val, axis=-1, keepdims=True)
    # np.around 使用銀行家捨入法 (round half to even)，與 torch.round 相同
    mean_int = np.around(mean_float)
    
    # 2. Centering
    # PyTorch: y_int = x_int - mean_int
    y_int = x_val - mean_int
    
    # 3. Variance
    # PyTorch: y_sq_int = y_int ** 2
    # PyTorch: var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Integer Iteration (Newton-Raphson for sqrt)
    # PyTorch:
    #   k = 2 ** 16
    #   for _ in range(10):
    #       k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
    #       k = k_1
    #   std_int = k
    k = np.full_like(var_int, 2**16, dtype=np.float64)
    
    for _ in range(10):
        # 關鍵：PyTorch 使用浮點除法後做 floor
        # floor_ste.apply(var_int/k) 是 torch.floor(浮點除法)
        div_result = np.floor(var_int / k)
        k = np.floor((k + div_result) / 2)
    
    std_int = k
    
    # 5. Factor Calculation
    # PyTorch: factor = floor_ste.apply((2 ** 31-1) / std_int)
    MAX_INT31 = 2**31 - 1
    std_int_safe = np.maximum(std_int, 1)
    # 關鍵：PyTorch 使用浮點除法後做 floor
    factor = np.floor(MAX_INT31 / std_int_safe)
    
    # 6. Scaling
    # PyTorch: y_int = floor_ste.apply(y_int * factor / 2)
    # 關鍵：這是 (y_int * factor) / 2.0，結果再做 floor
    term = y_int * factor
    y_int_norm = np.floor(term / 2)
    
    # 7. Add Bias
    # PyTorch: y_int = y_int + bias_int
    output_int = y_int_norm + bias_val
    
    return output_int.astype(np.int32)


def int_layer_norm_pure_integer(x_int, bias_int):
    """
    Pure Integer Implementation of IntLayerNorm.
    
    這個版本嘗試完全避免浮點運算，使用純整數算術。
    這是硬體 C-model / RTL 應該實現的版本。
    
    ⚠️ 與 PyTorch 的差異來源：
    1. 整數除法 vs 浮點除法後 floor
    2. 右移 vs 浮點除以 2 後 floor
    
    對於正數，這些操作是等價的。
    對於負數，存在差異！
    """
    x_val = x_int.astype(np.int64)
    bias_val = bias_int.astype(np.int64)
    
    N = x_int.shape[-1]  # 192
    
    # 1. Mean - 銀行家捨入的純整數實現
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    
    # 整數除法做銀行家捨入
    q = sum_val // N
    r = sum_val % N
    
    # 處理負數餘數
    neg_mask = sum_val < 0
    if np.any(neg_mask):
        # Python 的 % 對負數返回正餘數，需要調整
        # 例如: -1 % 192 = 191 in Python, 但我們需要 -1 as remainder
        # 調整：如果原值是負的且餘數不為0，q 要加 1，r 要減 N
        adjust_mask = neg_mask & (r != 0)
        q = np.where(adjust_mask, q + 1, q)
        r = np.where(adjust_mask, r - N, r)
    
    threshold = N // 2  # 96
    
    round_up_mask = r > threshold
    round_down_mask = r < -threshold
    
    # 銀行家捨入規則
    half_way_positive = (r == threshold)
    half_way_negative = (r == -threshold)
    odd_mask = (q % 2 != 0)
    round_half_to_even_up = half_way_positive & odd_mask
    round_half_to_even_down = half_way_negative & odd_mask
    
    # 正數餘數：可能要向上捨入
    correction_up = (round_up_mask | round_half_to_even_up).astype(np.int64)
    # 負數餘數：可能要向下捨入
    correction_down = (round_down_mask | round_half_to_even_down).astype(np.int64)
    
    mean_int = q + correction_up - correction_down
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Integer Iteration
    # 純整數版本 - 使用整數除法和右移
    k = np.full_like(var_int, 2**16, dtype=np.int64)
    
    for _ in range(10):
        safe_k = np.maximum(k, 1)
        # 整數除法
        div_result = var_int // safe_k
        # 右移 1 位 (等價於整數除以 2)
        k = (k + div_result) >> 1
    
    std_int = k
    
    # 5. Factor Calculation
    MAX_INT31 = 2**31 - 1
    std_int_safe = np.maximum(std_int, 1)
    # 整數除法
    factor = MAX_INT31 // std_int_safe
    
    # 6. Scaling
    term = y_int * factor
    # 右移 1 位
    y_int_norm = term >> 1
    
    # 7. Add Bias
    output_int = y_int_norm + bias_val
    
    return output_int.astype(np.int32)


def analyze_difference(my_out, golden_out, name=""):
    """分析兩個輸出之間的差異"""
    diff = my_out - golden_out
    abs_diff = np.abs(diff)
    
    mismatches = np.sum(diff != 0)
    total_elements = diff.size
    
    print(f"\n{'='*50}")
    print(f"Analysis: {name}")
    print(f"{'='*50}")
    
    print(f"Strict Mismatches: {mismatches} / {total_elements} ({mismatches/total_elements*100:.2f}%)")
    print(f"Max Absolute Diff: {np.max(abs_diff)}")
    print(f"Mean Absolute Diff: {np.mean(abs_diff):.4f}")
    
    # 容差分析
    for tol in [1, 10, 100, 1000]:
        passed = np.sum(abs_diff <= tol)
        print(f"Pass Rate (|diff| <= {tol:4d}): {passed/total_elements*100:.4f}%")
    
    if mismatches > 0:
        indices = np.where(diff != 0)
        print(f"\nFirst 5 mismatches:")
        for i in range(min(5, len(indices[0]))):
            idx = tuple(ind[i] for ind in indices)
            print(f"  Pos {idx}: My={my_out[idx]}, Golden={golden_out[idx]}, Diff={diff[idx]}")


def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    if not os.path.exists(base_dir) and os.path.exists("../../patterns_pytorch"):
        base_dir = "../../patterns_pytorch"

    input_path = os.path.join(base_dir, "golden/qact1_int.npy")
    bias_path = os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy")
    golden_path = os.path.join(base_dir, "golden/blocks_0_norm1_int.npy")
    
    print(f"Loading from: {base_dir}")
    x_int = load_npy(input_path)
    bias_int = load_npy(bias_path)
    golden_out = load_npy(golden_path)
    
    print(f"Input shape: {x_int.shape}")
    print(f"Input dtype: {x_int.dtype}")
    print(f"Input range: [{x_int.min()}, {x_int.max()}]")
    print(f"Golden shape: {golden_out.shape}")
    print(f"Golden dtype: {golden_out.dtype}")
    
    # 測試 PyTorch 等效版本
    pytorch_equiv_out = int_layer_norm_pytorch_equivalent(x_int, bias_int)
    analyze_difference(pytorch_equiv_out, golden_out, "PyTorch-Equivalent (using float operations)")
    
    # 測試純整數版本
    pure_int_out = int_layer_norm_pure_integer(x_int, bias_int)
    analyze_difference(pure_int_out, golden_out, "Pure Integer (for hardware C-model)")
    
    # 比較兩個版本的差異
    print("\n" + "="*50)
    print("Comparison: PyTorch-Equivalent vs Pure Integer")
    print("="*50)
    diff_implementations = pytorch_equiv_out - pure_int_out
    mismatches_impl = np.sum(diff_implementations != 0)
    print(f"Differences between implementations: {mismatches_impl} / {diff_implementations.size}")
    
    if mismatches_impl > 0:
        print("\n💡 結論：")
        print("=" * 50)
        print("PyTorch 的 IntLayerNorm 使用浮點運算 (torch.floor, torch.round)，")
        print("這導致 Golden Pattern 無法用純整數算術完美復現。")
        print("\n建議方案：")
        print("1. [推薦] 修改 PyTorch 代碼使用純整數運算，重新生成 Pattern")
        print("2. [備選] 接受小幅誤差 (對最終準確率影響極小)")
        print("3. [備選] 在 C-model 中模擬浮點 floor 行為")


if __name__ == "__main__":
    main()
