import numpy as np
import os

PATTERNS_DIR = "/home/undertaker4141/I_VIT/patterns_tvm"

def check_requant():
    qkv = np.load(f"{PATTERNS_DIR}/golden/block_0_qkv_int.npy").astype(np.int64)
    req_gold = np.load(f"{PATTERNS_DIR}/golden/block_0_req_qkv_to_matmul1_int.npy").astype(np.int32)
    
    M = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_M.npy").flatten()
    S = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_S.npy").flatten()
    
    # Broadcast M and S
    # qkv shape: (1, 197, 576), M shape: (576,)
    M = M.reshape(1, 1, -1).astype(np.int64)
    S = S.reshape(1, 1, -1).astype(np.int64)
    
    # Hardware Requant:
    add_val = np.left_shift(np.int64(1), S - 1)
    req_hw = np.right_shift((qkv * M) + add_val, S)
    req_hw = np.clip(req_hw, -128, 127)
    
    diff = req_hw - req_gold
    print(f"Num mismatch (HW Requant vs TVM): {np.sum(diff != 0)} / {diff.size}")
    if np.sum(diff != 0) > 0:
        print(req_hw[diff != 0][:5])
        print(req_gold[diff != 0][:5])
    
    # Float Requant:
    float_s = np.load(f"{PATTERNS_DIR}/scales/block_0_qconfig_qkv_output_scale_float.npy").flatten()
    float_s = float_s.reshape(1, 1, -1)
    
    # Wait, the TVM input to QKV was req, QKV outputs int32. The output scale of QKV is something, 
    # but the requantization to matmul relies on some specific in/out scales.
    # In TVM, the scale factor for requantizing an int32 back to int8 is usually:
    # out_scale / (in_scale * w_scale) --> but TVM just dumps the QKV output scale which IS this factor.
    # Actually, TVM pre-calculates the correct effective scale and stores it in _M and _S for output.
    req_float = np.round(qkv * float_s)
    req_float = np.clip(req_float, -128, 127)
    
    diff_f = req_float - req_gold
    print(f"Num mismatch (Float Requant vs TVM): {np.sum(diff_f != 0)} / {diff_f.size}")

check_requant()
