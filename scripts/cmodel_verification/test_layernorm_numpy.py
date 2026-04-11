import numpy as np
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)

def int_layer_norm_pure(x_int, bias_int):
    """
    Pure Integer Implementation of IntLayerNorm.
    Uses "Round Half To Even" (Banker's Rounding) for Mean calculation
    to better match PyTorch's float behavior, but keeps everything in INT64.
    """
    x_val = x_int.astype(np.int64) 
    bias_val = bias_int.astype(np.int64)
    
    N = x_int.shape[-1] # 192
    
    # 1. Mean
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    
    q = sum_val // N
    r = sum_val % N
    
    threshold = N // 2
    
    round_up_mask = (r > threshold)
    
    # PyTorch 使用的是 銀行家捨入法
    # 當 r == threshold 時，捨入到最近的偶數
    half_way_mask = (r == threshold)
    odd_mask = (q % 2 != 0)
    round_half_to_even_mask = half_way_mask & odd_mask
    
    correction = (round_up_mask | round_half_to_even_mask).astype(np.int64)
    mean_int = q + correction
    
    mean_int = mean_int.astype(np.int32)
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = (y_int ** 2)
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True) 
    
    # 4. Integer Iteration
    k = np.full_like(var_int, 65536, dtype=np.int64) # 65536 = 2^16
    
    for _ in range(10):
        safe_k = np.maximum(k, 1) # 避免除以 0
        div = var_int // safe_k
        k = (k + div) >> 1
        
    std_int = k.astype(np.int32)
    
    # 5. Factor Calculation
    MAX_INT31 = 2**31 - 1
    std_int_safe = np.maximum(std_int, 1)
    factor = MAX_INT31 // std_int_safe
    
    # 6. Scaling
    term = y_int * factor
    y_int_norm = term >> 1
    
    y_int_norm = y_int_norm.astype(np.int32)
    
    # 7. Add Bias
    output_int = y_int_norm + bias_val.astype(np.int32)
    
    return output_int

def main():
    base_dir = "patterns_pytorch"
    # if not os.path.exists(base_dir):
    #     base_dir = "I-ViT/patterns_pytorch"
    # if not os.path.exists(base_dir) and os.path.exists("../../patterns_pytorch"):
    #      base_dir = "../../patterns_pytorch"

    input_path = os.path.join(base_dir, "golden/qact1_int.npy")
    bias_path = os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy")
    golden_path = os.path.join(base_dir, "golden/blocks_0_norm1_int.npy")
    
    x_int = load_npy(input_path)
    bias_int = load_npy(bias_path)
    golden_out = load_npy(golden_path)
    
    my_out = int_layer_norm_pure(x_int, bias_int)
    
    diff = my_out - golden_out
    mismatches = np.sum(diff != 0)
    total_elements = diff.size
    
    print("-" * 40)
    print(f"Verification Results:")
    
    golden_safe = np.where(golden_out == 0, 1, golden_out).astype(np.float64)
    rel_error = np.abs(diff) / np.abs(golden_safe)
    
    atol = 100
    rtol = 1e-4
    
    match_mask = (np.abs(diff) <= atol) | (rel_error <= rtol)
    passed_elements = np.sum(match_mask)
    pass_rate = passed_elements / total_elements * 100
    
    print(f"Strict Mismatches:     {mismatches} / {total_elements} ({mismatches/total_elements*100:.2f}%)")
    print(f"Pass Rate (Tol-based): {passed_elements} / {total_elements} ({pass_rate:.6f}%)")
    
    if mismatches > 0:
        indices = np.where(~match_mask)
        if len(indices[0]) > 0:
            print(f"\nExample Major Mismatch (out of {len(indices[0])}):")
            idx = tuple(ind[0] for ind in indices)
            print(f"Pos {idx}: My={my_out[idx]}, Golden={golden_out[idx]}, Diff={diff[idx]}")
        else:
             print("\nAll mismatches are within tolerance.")

if __name__ == "__main__":
    main()
