import numpy as np
import os
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PATTERNS_DIR = os.path.join(PARENT_DIR, "patterns_tvm")

# ============================================================
# C-Model: Linear Operations (Dense, MatMul)
# ============================================================

def int_dense_kernel(x_int, weight_int, bias_int=None):
    """
    純整數 C-Model 實現 - Dense/Linear
    x_int: (B, Seq, in_features)
    weight_int: (out_features, in_features)
    bias_int: (out_features,) optional
    """
    x_val = x_int.astype(np.int32)
    w_val = weight_int.astype(np.int32)
    
    # x_val @ w_val^T
    out_val = np.matmul(x_val, w_val.T)
    
    if bias_int is not None:
        out_val = out_val + bias_int.astype(np.int32)
        
    return out_val.astype(np.int32)

def int_matmul_kernel(x_int, y_int):
    """
    純整數 C-Model 實現 - Batch Matmul
    對應 Relay 的 batch_matmul: x @ y^T, 注意轉置只針對最後兩維
    x_int: (B, H, M, K)
    y_int: (B, H, N, K)
    output: (B, H, M, N)
    """
    x_val = x_int.astype(np.int32)
    y_val = y_int.astype(np.int32)
    
    # 進行矩陣相乘 x_val @ y_val.T
    return np.matmul(x_val, y_val.transpose(0, 1, 3, 2))


# ============================================================
# Verification
# ============================================================

def calc_metrics(my_out, golden_out):
    diff = my_out.astype(np.int64) - golden_out.astype(np.int64)
    abs_diff = np.abs(diff)
    total = diff.size
    exact_match = int(np.sum(diff == 0))
    mae = float(np.mean(abs_diff))
    max_diff = int(np.max(abs_diff))
    return {
        "total": total,
        "exact_match": exact_match,
        "exact_rate": exact_match / total * 100,
        "mae": mae,
        "max_diff": max_diff,
    }

def print_metrics(name, m):
    status = "[PASS]" if m["exact_rate"] > 99.9 else "[FAIL]"
    print(f"  {status} {name}")
    print(f"    exact: {m['exact_match']}/{m['total']} ({m['exact_rate']:.2f}%)")
    print(f"    MAE: {m['mae']:.4f}, Max Diff: {m['max_diff']}")
    return status

def test_qkv_dense(block_id):
    # Load input from previous requantize
    x_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_req_norm1_to_qkv_int.npy")
    w_path = os.path.join(PATTERNS_DIR, "weights", f"block_{block_id}_attn_qkv_weight.npy")
    b_path = os.path.join(PATTERNS_DIR, "weights", f"block_{block_id}_attn_qkv_bias.npy")
    golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_qkv_int.npy")
    
    # Fallback to pure .npy if _int doesn't exist
    if not os.path.exists(x_path):
        x_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_req_norm1_to_qkv.npy")
    if not os.path.exists(golden_path):
        golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_qkv.npy")

    if any(not os.path.exists(p) for p in [x_path, w_path, b_path, golden_path]):
        return None, "SKIP (Missing Files)"

    x = np.load(x_path).astype(np.int32)
    w = np.load(w_path).astype(np.int32)
    b = np.load(b_path).astype(np.int32)
    golden = np.load(golden_path).astype(np.int32)

    my_out = int_dense_kernel(x, w, b)
    return calc_metrics(my_out, golden), None

def test_attn_matmul1(block_id, num_heads=3):
    # req_qkv_to_matmul1 in TVM provides the flat vector (1, 197, 576)
    x_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_req_qkv_to_matmul1_int.npy")
    golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_attn_matmul1_int.npy")
    
    if not os.path.exists(x_path):
        x_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_req_qkv_to_matmul1.npy")
    if not os.path.exists(golden_path):
        golden_path = os.path.join(PATTERNS_DIR, "golden", f"block_{block_id}_attn_matmul1.npy")

    if any(not os.path.exists(p) for p in [x_path, golden_path]):
        return None, "SKIP (Missing Files)"

    qkv = np.load(x_path).astype(np.int32) # (1, 197, 576)
    golden = np.load(golden_path).astype(np.int32)
    
    # Emulate the split in TVM to extract q and k
    # qkv_reshape = relay.reshape(req2, [0, 0, 3, num_heads, -1])  # (1, 197, 3, 3, 64)
    qkv_reshape = qkv.reshape(1, qkv.shape[1], 3, num_heads, -1)
    
    # qkv_t = relay.transpose(qkv_reshape, [2, 0, 3, 1, 4])        # (3, 1, 3, 197, 64)
    qkv_t = qkv_reshape.transpose(2, 0, 3, 1, 4)
    
    # q, k
    q = qkv_t[0] # (1, 3, 197, 64)
    k = qkv_t[1] # (1, 3, 197, 64)
    
    # matmul
    my_out = int_matmul_kernel(q, k)
    return calc_metrics(my_out, golden), None

def main():
    print("=" * 70)
    print("IViT CModel Linear Layers Pattern Verification")
    print("=" * 70)
    
    print("\n--------------------------------------------------")
    print("  [Dense] QKV (x @ w.T + b)")
    print("--------------------------------------------------")
    for blk in range(12):
        metrics, skip = test_qkv_dense(blk)
        if skip:
            print(f"  Block {blk}: {skip}")
        else:
            print_metrics(f"Block {blk:2d}", metrics)
            
    print("\n--------------------------------------------------")
    print("  [MatMul] Attention MatMult 1 (q @ k.T)")
    print("--------------------------------------------------")
    for blk in range(12):
        metrics, skip = test_attn_matmul1(blk)
        if skip:
            print(f"  Block {blk}: {skip}")
        else:
            print_metrics(f"Block {blk:2d}", metrics)

if __name__ == "__main__":
    main()
