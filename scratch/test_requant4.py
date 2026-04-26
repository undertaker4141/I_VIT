import numpy as np

PATTERNS_DIR = "/home/undertaker4141/I_VIT/patterns_tvm"

def get_eff_scale_ms(in_scale, out_scale, max_bit=31):
    eff_scale = in_scale / out_scale
    mantissa, exponent = np.frexp(eff_scale)
    M = np.round(mantissa * (2 ** max_bit)).astype(np.int64)
    S = max_bit - exponent
    return M, S.astype(np.int64)

def check_qkv_req_hw():
    qkv_int = np.load(f"{PATTERNS_DIR}/golden/block_0_qkv_int.npy").astype(np.int64)
    req_gold = np.load(f"{PATTERNS_DIR}/golden/block_0_req_qkv_to_matmul1_int.npy").astype(np.int32)
    
    in_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_float.npy").flatten()
    out_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_matmul_1_input_scale_float.npy").flatten()
    
    M, S = get_eff_scale_ms(in_s, out_s)
    
    # Broadcast M, S
    M = M.reshape(1, 1, -1)
    S = S.reshape(1, 1, -1)
    
    # Hardware logic
    add_val = np.left_shift(np.int64(1), S - 1)
    req = np.right_shift((qkv_int * M) + add_val, S)
    req = np.clip(req, -128, 127).astype(np.int32)
    
    diff = req - req_gold
    print(f"Num mismatch (QKV -> Matmul1 HW): {np.sum(diff != 0)}")
    
    # Let me also test with round to even since sometimes np.round is what TVM uses
    eff_scale = in_s / out_s
    req_f = np.round(qkv_int * eff_scale.reshape(1, 1, -1))
    req_f = np.clip(req_f, -128, 127).astype(np.int32)
    
    diff_f = req_f - req_gold
    print(f"Num mismatch (QKV -> Matmul1 Float built-in): {np.sum(diff_f != 0)}")
    
    # Diff between HW req and Float req
    diff_h_f = req - req_f
    print(f"Num mismatch (HW vs Float builtin): {np.sum(diff_h_f != 0)}")

check_qkv_req_hw()
