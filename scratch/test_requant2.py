import numpy as np
import os

PATTERNS_DIR = "/home/undertaker4141/I_VIT/patterns_tvm"

def requantize_test():
    qkv = np.load(f"{PATTERNS_DIR}/golden/block_0_qkv_int.npy").astype(np.int64)
    req_gold = np.load(f"{PATTERNS_DIR}/golden/block_0_req_qkv_to_matmul1_int.npy").astype(np.int32)
    
    in_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_input_scale_float.npy").flatten()
    wt_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_kernel_scale_float.npy").flatten()
    out_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_float.npy").flatten()
    
    # TVM effective scale:
    eff_scale = (in_s * wt_s) / out_s
    eff_scale = eff_scale.reshape(1, 1, -1)
    
    req_float = np.round(qkv * eff_scale)
    req_float = np.clip(req_float, -128, 127).astype(np.int32)
    
    diff_f = req_float - req_gold
    print(f"Num mismatch (Effective Float Requant vs TVM): {np.sum(diff_f != 0)} / {diff_f.size}")
    if np.sum(diff_f != 0) > 0:
        print(req_float[diff_f != 0][:5])
        print(req_gold[diff_f != 0][:5])

requantize_test()
