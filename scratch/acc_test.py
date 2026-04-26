import numpy as np
import os

CMOD_DIR = "/home/undertaker4141/I_VIT/patterns_cmodel_golden/golden"
TVM_DIR = "/home/undertaker4141/I_VIT/patterns_tvm/golden"

cmod_head = np.load(f"{CMOD_DIR}/head_int.npy")
tvm_head = np.load(f"{TVM_DIR}/head_int.npy") if os.path.exists(f"{TVM_DIR}/head_int.npy") else np.load(f"{TVM_DIR}/head.npy")

print("C-Model Top-5 classes:")
cmod_flat = cmod_head.flatten()
tvm_flat = tvm_head.flatten()

cmod_idx = np.argsort(cmod_flat)[::-1][:5]
tvm_idx = np.argsort(tvm_flat)[::-1][:5]

print("C-Model:", cmod_idx, "Scores:", cmod_flat[cmod_idx])
print("TVM:", tvm_idx, "Scores:", tvm_flat[tvm_idx])
