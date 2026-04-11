import numpy as np
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)

def int_exp_shift_kernel(x_int, x0_int, n=15):
    """
    Pure Integer Exponential Approximation Kernel
    x_int: Input tensor (int64)
    x0_int: Pre-calculated parameter (int64) - Scalar or Broadcastable
    n: Constant shift amount (default 15 for Softmax)
    """
    # 1. Polynomial Approximation: x + x/2 - x/16
    term1 = x_int >> 1
    term2 = x_int >> 4
    x_int = x_int + term1 - term2
    
    # 2. Clamp Lower Bound (x >= n * x0)
    lower_bound = n * x0_int
    x_int = np.maximum(x_int, lower_bound)
    
    # 3. Integer Division (q = x // x0)
    # Handle zero division if x0 is 0 (unlikely for proper scales)
    if np.any(x0_int == 0):
        # Fallback or error? Assuming x0 != 0
        pass 
        
    q = x_int // x0_int
    
    # 4. Remainder (r = x - x0 * q)
    r = x_int - x0_int * q
    
    # 5. Exp Base Calculation
    exp_int = (r >> 1) - x0_int
    
    # 6. Variable Shift (exp_int * 2^(n-q))
    shift = n - q
    
    # Handle variable shift
    res = np.zeros_like(exp_int)
    
    # Positive (Left) Shift
    pos_mask = shift >= 0
    if np.any(pos_mask):
        res[pos_mask] = np.left_shift(exp_int[pos_mask], shift[pos_mask])
        
    # Negative (Right) Shift
    neg_mask = shift < 0
    if np.any(neg_mask):
        res[neg_mask] = np.right_shift(exp_int[neg_mask], -shift[neg_mask])
        
    return np.maximum(res, 0)

def int_softmax_kernel(x_int, x0_int, output_bit=8, n=15):
    """
    Pure Integer Softmax Kernel
    x_int: Input tensor (int32/int64)
    x0_int: Parameter scalar (int64)
    """
    # Ensure int64
    pre_x_int = x_int.astype(np.int64)
    
    # 1. Stability Shift (x - x_max)
    # Max across last dimension (softmax dimension)
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    
    # Debug
    print(f"DEBUG: x_int_max min={x_int_max.min()}, max={x_int_max.max()}")
    
    x_algo = pre_x_int - x_int_max
    print(f"DEBUG: x_algo (shifted) min={x_algo.min()}, max={x_algo.max()}")
    
    # 2. Calculate Exp(x_algo)
    exp_int = int_exp_shift_kernel(x_algo, x0_int, n)
    print(f"DEBUG: exp_int min={exp_int.min()}, max={exp_int.max()}")
    
    # 3. Sum Exponentials
    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    
    # Clamp Sum
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    
    # 4. Calculate Factor (Approximate Reciprocal)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # 5. Calculate Output Probability
    # exp_int * factor / 2^(31 - output_bit + 1)
    shift_amt = 31 - output_bit + 1
    term = exp_int.astype(object) * factor.astype(object)
    output = (term >> shift_amt).astype(np.int64)
    
    print(f"DEBUG: output min={output.min()}, max={output.max()}")
    
    return output.astype(np.int32)

def prepare_softmax_params(m, s):
    """
    Host-side parameter preparation
    """
    # Calculate Scaling Factor
    scaling_factor = m.astype(np.float64) * (2.0 ** (-s.astype(np.float64)))
    
    # Note: Softmax uses 'scaling_factor' directly for x0 calculation in code
    # BUT wait, look at quant_modules.py:
    # x0_int = torch.floor(-1.0 / scaling_factor)
    # Unlike GELU, it does NOT multiply by 1.702!
    
    # Calculate x0_int (Register Value)
    x0_int = np.floor(-1.0 / scaling_factor).astype(np.int64)
    
    print(f"DEBUG: Scale min={scaling_factor.min()}, max={scaling_factor.max()}")
    print(f"DEBUG: x0_int min={x0_int.min()}, max={x0_int.max()}")
    
    return x0_int

def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    
    print(f"Using patterns directory: {base_dir}")
    
    # Files
    input_path = os.path.join(base_dir, "golden/blocks_0_attn_qact_attn1_int.npy")
    m_path = os.path.join(base_dir, "scales/blocks_0_attn_qact_attn1_act_M.npy")
    s_path = os.path.join(base_dir, "scales/blocks_0_attn_qact_attn1_act_S.npy")
    golden_path = os.path.join(base_dir, "golden/blocks_0_attn_int_softmax_int.npy")
    
    if not os.path.exists(input_path):
         print(f"Input file not found: {input_path}")
         return

    print("Loading files...")
    x_int = load_npy(input_path)
    m = load_npy(m_path)
    s = load_npy(s_path)
    golden_out = load_npy(golden_path)
    
    print(f"Input shape: {x_int.shape}")
    print(f"M shape: {m.shape}")
    print(f"S shape: {s.shape}")
    print(f"Golden shape: {golden_out.shape}")
    
    print("Running IntSoftmax simulation...")
    
    # 1. Prepare Params
    x0_int = prepare_softmax_params(m, s)
    
    # 2. Run Kernel
    # n=15 for Softmax
    # output_bit=16 based on Golden Max ~32000
    my_out = int_softmax_kernel(x_int, x0_int, output_bit=16, n=15)
    
    # Analysis

    # Checking correlation and exact matches
    flat_my = my_out.flatten().astype(np.float64)
    flat_gold = golden_out.flatten().astype(np.float64)
    
    correlation = np.corrcoef(flat_my, flat_gold)[0, 1]
    print(f"Correlation: {correlation}")
    
    diff = my_out - golden_out
    abs_diff = np.abs(diff)
    
    tol = 20
    passed = np.sum(abs_diff <= tol)
    total = diff.size
    print(f"Pass Rate (AbsDiff <= {tol}): {passed} / {total} ({passed/total*100:.2f}%)")
    
    mismatches = np.sum(diff != 0)
    print(f"Strict Mismatches: {mismatches} / {total} ({mismatches/total*100:.2f}%)")
    
    if mismatches > 0:
        print(f"Max Diff: {np.max(abs_diff)}")
        indices = np.where(diff != 0)
        print("Example mismatch:")
        idx = tuple(ind[0] for ind in indices)
        print(f"Pos {idx}: My={my_out[idx]}, Golden={golden_out[idx]}, Diff={diff[idx]}")

if __name__ == "__main__":
    main()
