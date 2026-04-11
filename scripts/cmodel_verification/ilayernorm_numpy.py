import numpy as np
import torch
import os
import sys

def load_npy(path):
    if not os.path.exists(path):
        print(f"Error: File not found {path}")
        sys.exit(1)
    return np.load(path)


def int_layer_norm_pure(x_int, bias_int):

    n = torch.tensor(x_int.shape[2], dtype=torch.float)
    dim_sqrt = torch.sqrt(n)
    mean_int = torch.round(x_int.float().mean(axis=2, keepdim=True))
    y_int = x_int - mean_int
    y_sq_int = y_int ** 2
    var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

    # Integer Iteration
    k = 2 ** 16
    for _ in range(10):
        k_1 = torch.floor((k + torch.floor(var_int/k))/2)
        k = k_1
    std_int = k

    factor = torch.floor((2 ** 31-1) / std_int)
    y_int = torch.floor(y_int * factor / 2)
    scaling_factor = dim_sqrt / 2 ** 30

    y_int = y_int + bias_int
    
    return y_int


def main():
    base_dir = "patterns_pytorch"
    if not os.path.exists(base_dir):
        base_dir = "I-ViT/patterns_pytorch"
    if not os.path.exists(base_dir) and os.path.exists("../../patterns_pytorch"):
         base_dir = "../../patterns_pytorch"

    input_path = os.path.join(base_dir, "golden/qact1_int.npy")
    bias_path = os.path.join(base_dir, "weights/blocks_0_norm1_bias_integer.npy")
    golden_path = os.path.join(base_dir, "golden/blocks_0_norm1_int.npy")
    
    x_int = load_npy(input_path)
    bias_int = load_npy(bias_path)
    golden_out = load_npy(golden_path)
    
    my_out = int_layer_norm_pure(torch.from_numpy(x_int), torch.from_numpy(bias_int))
    
    diff = my_out - torch.from_numpy(golden_out)
    mismatches = np.sum(diff.numpy() != 0)
    total_elements = diff.numpy().size
    
    print("-" * 40)
    print(f"Verification Results:")
    
    golden_safe = np.where(golden_out == 0, 1, golden_out).astype(np.float64)
    rel_error = np.abs(diff) / np.abs(golden_safe)
    
    atol = 0
    rtol = 1e-4
    
    # match_mask = (np.abs(diff) <= atol) | (rel_error <= rtol)
    match_mask = (np.abs(diff) <= atol)
    passed_elements = np.sum(match_mask.numpy())
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
