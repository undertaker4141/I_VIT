import numpy as np

PATTERNS_DIR = "/home/undertaker4141/I_VIT/patterns_tvm"

# Test norm1 to qkv requant
def check_norm1_req():
    norm_int = np.load(f"{PATTERNS_DIR}/golden/block_0_norm1_int.npy").astype(np.int64)
    req_gold = np.load(f"{PATTERNS_DIR}/golden/block_0_req_norm1_to_qkv_int.npy").astype(np.int32)
    
    in_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_norm1_output_scale_float.npy").flatten()
    out_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_input_scale_float.npy").flatten()
    
    eff_scale = in_s / out_s
    req = np.round(norm_int * eff_scale)
    req = np.clip(req, -128, 127).astype(np.int32)
    
    diff = req - req_gold
    print(f"Num mismatch (Norm1 -> QKV float): {np.sum(diff != 0)}")

def check_qkv_req():
    # qkv output to matmul1 input
    # qkv output is int32. This is output of Dense. 
    # In TVM, the target scale of req after QKV is qconfig2.input_scale
    qkv_int = np.load(f"{PATTERNS_DIR}/golden/block_0_qkv_int.npy").astype(np.int64)
    req_gold = np.load(f"{PATTERNS_DIR}/golden/block_0_req_qkv_to_matmul1_int.npy").astype(np.int32)
    
    in_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_float.npy").flatten()
    out_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_matmul_1_input_scale_float.npy").flatten()
    
    eff_scale = in_s / out_s
    # qkv_int has shape (1, 197, 576), eff_scale has shape (576,)
    req = np.round(qkv_int * eff_scale.reshape(1, 1, -1))
    req = np.clip(req, -128, 127).astype(np.int32)

    diff = req - req_gold
    print(f"Num mismatch (QKV -> Matmul1 float): {np.sum(diff != 0)}")

check_norm1_req()
check_qkv_req()
