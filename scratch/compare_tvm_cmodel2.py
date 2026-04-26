import numpy as np
import os

TVM_DIR = "/home/undertaker4141/I_VIT/patterns_tvm/golden"
CMOD_DIR = "/home/undertaker4141/I_VIT/patterns_cmodel_golden/golden"

layers = [
    "embed_req",
]

for l in layers:
    tvm_file = os.path.join(TVM_DIR, f"{l}.npy")
    if not os.path.exists(tvm_file):
        tvm_file = os.path.join(TVM_DIR, f"{l}_int.npy")
    cmod_file = os.path.join(CMOD_DIR, f"{l}_int.npy")
    
    if not os.path.exists(tvm_file) or not os.path.exists(cmod_file):
        continue
        
    t = np.load(tvm_file).astype(np.float64) 
    c = np.load(cmod_file).astype(np.float64)
    
    if t.shape != c.shape:
        try:
            t = t.reshape(c.shape)
        except:
            pass
            
    diff = t - c
    max_diff = np.max(np.abs(diff))
    print(f"{l:<30} | Max: {max_diff:<10.1f}")
