"""
IViT CModel Golden Pattern 修正驗證腳本 (v3)
=============================================
針對 GELU 和 LayerNorm 修正 CModel 核心演算法使其
與 TVM 原始碼 (layers.py) bit-exact 對齊：

GELU 修正：
  - exp kernel 改為標準版: (r >> 1) - x0, shift = n - q
  - (原本用 delay-division: r - 2*x0, shift = n - q - 1)

LayerNorm 修正：
  - Newton 初始值改為固定 2^16 (TVM: relay.const(2**16, 'uint32'))
  - (原本用動態 2^(bit_len/2))
  - normalize 順序: (factor/std) * data / 2 (先算 factor/std 再乘 data)
  - (原本用 data * factor >> 1，先乘後移)
"""
import numpy as np
import os
import sys
import json
from datetime import datetime

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PARENT_DIR)
PATTERNS_DIR = os.path.join(PARENT_DIR, "patterns_tvm")
NUM_BLOCKS = 12

# ============================================================
# 修正版 CModel
# ============================================================

def int_exp_shift_kernel_standard(x_int, x0_int, n):
    """
    標準版 shift_exp，與 TVM layers.py 完全對齊：
      data = data + (data >> 1) - (data >> 4)
      x0 = int(-1.0/input_scale - 1)
      data = max(data, n * x0)
      q = data / x0
      r = data - q * x0
      exp_int = (r >> 1) - x0        ← 標準版
      exp_int = exp_int << (n - q)    ← 標準版
    """
    # 1. Polynomial approx: x * 1.5625
    term1 = x_int >> 1
    term2 = x_int >> 4
    x_int = x_int + term1 - term2

    # 2. Clamp
    lower_bound = n * x0_int
    x_int = np.maximum(x_int, lower_bound)

    # 3. Integer division
    q = x_int // x0_int

    # 4. Remainder
    r = x_int - x0_int * q

    # 5. Base (標準版)
    exp_int = (r >> 1) - x0_int

    # 6. Variable shift: n - q (標準版)
    shift = n - q

    res = np.zeros_like(exp_int)
    pos_mask = shift >= 0
    if np.any(pos_mask):
        res[pos_mask] = np.left_shift(exp_int[pos_mask], shift[pos_mask])
    neg_mask = shift < 0
    if np.any(neg_mask):
        res[neg_mask] = np.right_shift(exp_int[neg_mask], -shift[neg_mask])

    return np.maximum(res, 0)


def int_gelu_kernel_fixed(x_int, x0_int, output_bit=8, n=23):
    """修正版 GELU，使用標準版 exp kernel"""
    pre_x_int = x_int.astype(np.int64)

    # 1. Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max

    # 2. exp(x - x_max)
    exp_int = int_exp_shift_kernel_standard(x_algo, x0_int, n)

    # 3. exp(-x_max)
    exp_int_max = int_exp_shift_kernel_standard(-x_int_max, x0_int, n)

    # 4. Sigmoid = exp(x) / (exp(x) + exp(-x_max))
    exp_int_sum = exp_int + exp_int_max
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)

    factor = (2**31 - 1) // exp_int_sum_safe

    shift_amt = 31 - output_bit + 1  # = 24
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)

    # 5. gelu = x * sigmoid
    output = pre_x_int * sigmoid_int
    return output.astype(np.int32)


def int_softmax_kernel_fixed(x_int, x0_int, output_bit=8, n=16):
    """Softmax CModel (已驗證 bit-exact)"""
    pre_x_int = x_int.astype(np.int64)

    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max

    exp_int = int_exp_shift_kernel_standard(x_algo, x0_int, n)

    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)

    factor = (2**31 - 1) // exp_int_sum_safe

    shift_amt = 31 - output_bit + 1  # = 24
    term = exp_int.astype(object) * factor.astype(object)
    output = (term >> shift_amt).astype(np.int64)

    return output.astype(np.int32)


def int_layer_norm_fixed(x_int, bias_int):
    """
    修正版 LayerNorm，與 TVM layers.py 完全對齊：
    - Newton 初始值固定 2^16
    - normalize: (factor / std) * data / 2
    - 取用 TVM int16 累積的溢位特性與無條件捨去至零的截斷法
    """
    x_val = x_int.astype(np.int32)
    N = x_val.shape[-1]

    # 1. 為了模擬 TVM 中 sum 累積時的 int16 溢位
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    sum_wrapped = sum_val.astype(np.int16).astype(np.int32)
    
    # 2. Mean (truncated division towards zero)
    mean_int = np.fix(sum_wrapped / N).astype(np.int32)

    # 3. Center
    y_int = x_val - mean_int
    # 再次模擬 TVM 的 int16 溢位 (data - mean 在 TVM 轉型 int32 前為 int16 運算)
    y_int = y_int.astype(np.int16).astype(np.int32)

    # 4. Square → uint32
    data_sq = (y_int.astype(np.int32) * y_int.astype(np.int32)).astype(np.uint32)

    # 5. Variance sum (在 uint32 的空間下加總不會溢位 int32)
    var = np.sum(data_sq, axis=-1, keepdims=True).astype(np.uint32)

    # 6. Newton iteration for sqrt - 固定初始值 2^16 (TVM)
    std = np.full_like(var, 2**16, dtype=np.uint32)
    for _ in range(10):
        safe_std = np.maximum(std, np.uint32(1))
        # TVM 的 std = (std + var/std)/2，為正整數的情形向下取整除法
        std = (std + var // safe_std) // np.uint32(2)

    std_i32 = std.astype(np.int32)
    std_safe = np.maximum(std_i32, np.int32(1))

    # 7. Normalize: (factor / std) * data / 2  (TVM 順序)
    factor = np.int32(2**31 - 1)
    factor_div_std = (factor // std_safe).astype(np.int32)

    y_int32 = y_int.astype(np.int32)
    # (factor/std) * data → int64 避免溢位
    term = factor_div_std.astype(np.int64) * y_int32.astype(np.int64)
    # C++ truncated division by 2 towards zero (避免 Python 的 // 負數 floor 行為)
    y_norm = np.where(term >= 0, term // 2, -((-term) // 2)).astype(np.int32)
    
    # 8. Add bias
    output_int = y_norm + bias_int.astype(np.int32)
    return output_int


# ============================================================
# 工具函式
# ============================================================
def load(path):
    if not os.path.exists(path):
        return None
    return np.load(path)

def calc_metrics(my_out, golden_out):
    diff = my_out.astype(np.int64) - golden_out.astype(np.int64)
    abs_diff = np.abs(diff)
    total = diff.size
    exact_match = int(np.sum(diff == 0))
    within_1 = int(np.sum(abs_diff <= 1))
    within_5 = int(np.sum(abs_diff <= 5))
    mae = float(np.mean(abs_diff))
    max_diff = int(np.max(abs_diff))
    return {
        "total": total,
        "exact_match": exact_match,
        "exact_rate": exact_match / total * 100,
        "within_1": within_1,
        "within_1_rate": within_1 / total * 100,
        "within_5": within_5,
        "within_5_rate": within_5 / total * 100,
        "mae": mae,
        "max_diff": max_diff,
    }

def print_metrics(name, m):
    status = "[PASS]" if m["exact_rate"] > 99.0 else ("[WARN]" if m["within_1_rate"] > 95 else "[FAIL]")
    print(f"  {status} {name}")
    print(f"    exact: {m['exact_match']}/{m['total']} ({m['exact_rate']:.2f}%)")
    print(f"    +/-1:  {m['within_1']}/{m['total']} ({m['within_1_rate']:.2f}%)")
    print(f"    MAE: {m['mae']:.4f}, Max Diff: {m['max_diff']}")
    return status

# ============================================================
# 測試函式
# ============================================================
def test_layernorm(block_id):
    if block_id == 0:
        input_path = os.path.join(PATTERNS_DIR, "golden", "embed_add_pos.npy")
    else:
        input_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id - 1}_add2.npy")
    
    golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_norm1_int.npy")
    bias_path = os.path.join(PATTERNS_DIR, "weights", f"block_{block_id}_norm1_bias.npy")

    x_int = load(input_path)
    golden = load(golden_path)
    bias = load(bias_path)
    if any(v is None for v in [x_int, golden, bias]):
        return None, "SKIP"

    my_out = int_layer_norm_fixed(x_int.astype(np.int32), bias)
    return calc_metrics(my_out, golden), None


def test_gelu(block_id):
    input_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_fc1_int.npy")
    golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_gelu_int.npy")
    gelu_scale_path = os.path.join(PATTERNS_DIR, "scales", f"block_{block_id}_qconfig_gelu_input_scale_float.npy")
    fc1_out_float_path = os.path.join(PATTERNS_DIR, "scales", f"block_{block_id}_qconfig_fc1_output_scale_float.npy")

    x_int = load(input_path)
    golden = load(golden_path)
    gelu_scale_arr = load(gelu_scale_path)
    fc1_out_float = load(fc1_out_float_path)

    if any(v is None for v in [x_int, golden, gelu_scale_arr, fc1_out_float]):
        return None, "SKIP"

    input_scale = float(gelu_scale_arr.flatten()[0])

    # Per-channel requant
    ratio = fc1_out_float.flatten() / input_scale
    x_float = x_int.astype(np.float64)
    ratio_bc = ratio.reshape(1, 1, -1)
    x_req = np.round(x_float * ratio_bc).astype(np.int32)
    x_req = np.clip(x_req, -128, 127).astype(np.int32)

    x0_int = int(-1.0 / (input_scale * 1.702) - 1)
    x0_int = np.int64(x0_int)

    my_out = int_gelu_kernel_fixed(x_req, x0_int, output_bit=8, n=23)
    return calc_metrics(my_out, golden), None


def test_softmax(block_id):
    input_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_attn_matmul1_int.npy")
    golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_softmax_int.npy")
    m1_out_float_path = os.path.join(PATTERNS_DIR, "scales", f"block_{block_id}_qconfig_matmul_1_output_scale_float.npy")
    sm_scale_path = os.path.join(PATTERNS_DIR, "scales", f"block_{block_id}_qconfig_softmax_input_scale_float.npy")

    x_int = load(input_path)
    golden = load(golden_path)
    m1_out_float = load(m1_out_float_path)
    sm_scale_arr = load(sm_scale_path)

    if any(v is None for v in [x_int, golden, m1_out_float, sm_scale_arr]):
        return None, "SKIP"

    input_scale = float(sm_scale_arr.flatten()[0])
    m1_out_scale = float(m1_out_float.flatten()[0])
    qk_scale = (192 // 3) ** -0.5  # = 0.125

    ratio = (m1_out_scale * qk_scale) / input_scale
    x_req = np.round(x_int.astype(np.float64) * ratio).astype(np.int32)
    x_req = np.clip(x_req, -128, 127).astype(np.int32)

    x0_int = int(-1.0 / input_scale - 1)
    x0_int = np.int64(x0_int)

    my_out = int_softmax_kernel_fixed(x_req, x0_int, output_bit=8, n=16)
    return calc_metrics(my_out, golden.astype(np.int32)), None


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 70)
    print("IViT CModel Golden Pattern Verification (v3 - Fixed CModels)")
    print(f"Pattern: {PATTERNS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    for func_name, test_fn, label in [
        ("layernorm", test_layernorm, "[LayerNorm] FIXED: Newton init=2^16, norm=(factor/std)*data/2"),
        ("gelu",      test_gelu,      "[GELU] FIXED: standard exp kernel (r>>1)-x0, shift=n-q"),
        ("softmax",   test_softmax,   "[Softmax] standard exp kernel, n=16, ob=8"),
    ]:
        print(f"\n{'-'*50}")
        print(f"  {label}")
        print(f"{'-'*50}")
        results = []
        for blk in range(NUM_BLOCKS):
            metrics, skip = test_fn(blk)
            if skip:
                print(f"  Block {blk:2d}: {skip}")
                continue
            status = print_metrics(f"Block {blk:2d}", metrics)
            results.append({"block": blk, **metrics, "status": status})
        all_results[func_name] = results

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for name, results in all_results.items():
        if not results:
            continue
        avg_exact = np.mean([r["exact_rate"] for r in results])
        avg_mae = np.mean([r["mae"] for r in results])
        max_d = max(r["max_diff"] for r in results)
        pc = sum(1 for r in results if "PASS" in r["status"])
        print(f"  {name.upper():12s}: PASS {pc}/{len(results)}, "
              f"avg exact {avg_exact:.2f}%, avg MAE {avg_mae:.4f}, max diff {max_d}")

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_test_results_v3.json")
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=conv, ensure_ascii=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
