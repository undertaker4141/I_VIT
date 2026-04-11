import numpy as np
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)

def int_exp_shift_kernel(x_int, x0_int, n=23):
    """
    Pure Integer Exponential Approximation Kernel
    x_int: Input tensor (int64)
    x0_int: Pre-calculated parameter (int64)
    n: Constant shift amount (default 23)
    """
    # 1. Polynomial Approximation: x + x/2 - x/16
    term1 = x_int >> 1
    term2 = x_int >> 4
    x_int = x_int + term1 - term2
    
    # 2. Clamp Lower Bound (x >= n * x0)
    lower_bound = n * x0_int
    x_int = np.maximum(x_int, lower_bound)
    
    # 3. Integer Division (q = x // x0)
    q = x_int // x0_int
    
    # 4. Remainder (r = x - x0 * q)
    r = x_int - x0_int * q
    
    # 5. Exp Base Calculation
    exp_int = (r >> 1) - x0_int
    
    # 6. Variable Shift (exp_int * 2^(n-q))
    shift = n - q
    
    # Handle variable shift (Vectorized)
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

def int_gelu_kernel(x_int, x0_int, output_bit=8, n=23):
    """
    Pure Integer GELU Kernel
    x_int: Input tensor (int32/int64)
    x0_int: Parameter vector (int64)
    """
    # Ensure int64 for intermediate calculations
    pre_x_int = x_int.astype(np.int64)
    
    # 1. Stability Shift (x - x_max)
    # x_max per token (max across last dimension)
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    
    # Debug info
    print(f"DEBUG: x_int_max min={x_int_max.min()}, max={x_int_max.max()}")
    
    x_algo = pre_x_int - x_int_max
    print(f"DEBUG: x_algo (shifted) min={x_algo.min()}, max={x_algo.max()}")
    
    # 2. Calculate Exp(x_algo)
    exp_int = int_exp_shift_kernel(x_algo, x0_int, n)
    print(f"DEBUG: exp_int min={exp_int.min()}, max={exp_int.max()}")
    
    # 3. Calculate Exp(-x_max)
    # Note: Broadcasting x0_int (channel-wise) against x_max (token-wise)
    # x0_int: (1, 1, 768), x_max: (1, 197, 1) -> Output: (1, 197, 768)
    exp_int_max = int_exp_shift_kernel(-x_int_max, x0_int, n)
    
    # 4. Sigmoid Calculation
    exp_int_sum = exp_int + exp_int_max
    
    # Clamp Sum
    exp_int_sum = np.minimum(exp_int_sum, 2**31 - 1)
    
    # Calculate Factor (Approximate Reciprocal)
    exp_int_sum_safe = np.maximum(exp_int_sum, 1)
    factor = (2**31 - 1) // exp_int_sum_safe
    
    # Calculate Sigmoid (using object/float64 to avoid overflow)
    shift_amt = 31 - output_bit + 1
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    print(f"DEBUG: sigmoid_int min={sigmoid_int.min()}, max={sigmoid_int.max()}")
    
    # 5. Apply Sigmoid
    output = pre_x_int * sigmoid_int
    return output.astype(np.int32)

def prepare_gelu_params(m, s): # 可以先存入
    """
    Host-side parameter preparation
    """
    # Calculate Scaling Factor
    scaling_factor = m.astype(np.float64) * (2.0 ** (-s.astype(np.float64)))
    scaling_factor_sig = scaling_factor * 1.702
    
    # Calculate x0_int (Register Value)
    x0_int = np.floor(-1.0 / scaling_factor_sig).astype(np.int64)
    
    print(f"DEBUG: Scale min={scaling_factor.min()}, max={scaling_factor.max()}")
    print(f"DEBUG: x0_int min={x0_int.min()}, max={x0_int.max()}")
    
    return x0_int



def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    
    print(f"Using patterns directory: {base_dir}")
    
    # Files
    # Hypothesis: Input to GELU is the QuantAct output (Int8)
    input_path = os.path.join(base_dir, "golden/blocks_0_mlp_qact_gelu_int.npy")
    m_path = os.path.join(base_dir, "scales/blocks_0_mlp_qact_gelu_act_M.npy")
    s_path = os.path.join(base_dir, "scales/blocks_0_mlp_qact_gelu_act_S.npy")
    golden_path = os.path.join(base_dir, "golden/blocks_0_mlp_act_int.npy")

    
    if not os.path.exists(golden_path):
        print(f"Golden file not found: {golden_path}")
        golden_path = os.path.join(base_dir, "golden/blocks_0_mlp_qact_gelu_int.npy")
    
    print("Loading files...")
    x_int = load_npy(input_path)
    m = load_npy(m_path)
    s = load_npy(s_path)
    golden_out = load_npy(golden_path)
    
    print(f"Input shape: {x_int.shape}")
    print(f"M shape: {m.shape}")
    print(f"S shape: {s.shape}")
    print(f"Golden shape: {golden_out.shape}")
    
    print("Running IntGELU simulation...")
    # Reshape M/S for broadcasting: (1, 1, 768)
    if m.shape[0] == x_int.shape[-1]:
       m = m.reshape(1, 1, -1)
       s = s.reshape(1, 1, -1)
    
    # 1. Prepare Params (Host)
    x0_int = prepare_gelu_params(m, s)
    
    # 2. Run Kernel (Device/C-Model)
    my_out = int_gelu_kernel(x_int, x0_int, output_bit=8)

    
    # Analysis
    flat_my = my_out.flatten().astype(np.float64)
    flat_gold = golden_out.flatten().astype(np.float64)
    
    correlation = np.corrcoef(flat_my, flat_gold)[0, 1]
    print(f"Correlation: {correlation}")
    
    # Tolerance Check
    # Allow small relative error or absolute error
    diff = my_out - golden_out
    abs_diff = np.abs(diff)
    
    # Relaxed Check: Pass if Diff <= 100 ?
    # Golden values are around ~300. Diff 16 is 5%.
    # Let's show Pass Rate with tolerance
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
