import numpy as np
import os

TVM_DIR = "/home/undertaker4141/I_VIT/patterns_tvm/golden"
CMOD_DIR = "/home/undertaker4141/I_VIT/patterns_cmodel_golden/golden"

def compare_file(name):
    tvm_file = os.path.join(TVM_DIR, f"{name}.npy")
    cmod_file = os.path.join(CMOD_DIR, f"{name}_int.npy")
    
    if not os.path.exists(tvm_file) or not os.path.exists(cmod_file):
        return None
        
    t = np.load(tvm_file).astype(np.int64)
    c = np.load(cmod_file).astype(np.int64)
    
    if t.shape != c.shape:
        print(f"Shape mismatch for {name}: TVM={t.shape}, CMod={c.shape}")
        # Try to reshape or flatten if safe, but usually means different layout
        c = c.reshape(t.shape)
        
    diff = t - c
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    num_diff = np.sum(diff != 0)
    total = diff.size
    diff_ratio = num_diff / total * 100
    
    return max_diff, mean_diff, num_diff, total, diff_ratio

layers = [
    "embed_conv_int",
    "block_0_qkv_int",
    "block_0_attn_matmul1_int",
    "block_0_softmax_int",
    "block_0_attn_matmul2_int",
    "block_0_fc1_int",
    "block_0_gelu_int",
    "block_0_add1_int",
    "block_0_add2_int",
    "block_11_qkv_int",
    "block_11_attn_matmul1_int",
    "block_11_softmax_int",
    "block_11_attn_matmul2_int",
    "block_11_fc1_int",
    "block_11_gelu_int",
    "block_11_add1_int",
    "block_11_add2_int",
]

print("=== Layer-wise Difference Analysis ===")
print(f"{'Layer Name':<30} | {'Max Diff':<10} | {'Mean Diff':<10} | {'Diff Count / Total':<20} | {'Diff %':<10}")
print("-" * 90)

for l in layers:
    # Golden TVM has names like block_0_qkv_int.npy
    # But C-Model also generated with _int in the name?
    # Actually my cmodel_vit_infer saves "self._save_mid(qkv, 'block_0_qkv')", which saves as "block_0_qkv_int.npy"
    # TVM Golden also saves as "block_0_qkv_int.npy".
    
    # Let's remove _int for my function
    base_name = l.replace("_int", "")
    tvm_file = os.path.join(TVM_DIR, f"{base_name}_int.npy")
    cmod_file = os.path.join(CMOD_DIR, f"{base_name}_int.npy")
    
    if not os.path.exists(tvm_file):
        tvm_file = os.path.join(TVM_DIR, f"{base_name}.npy")
    if not os.path.exists(tvm_file) or not os.path.exists(cmod_file):
        continue
        
    t = np.load(tvm_file).astype(np.float64) # in case some tvm output is not perfectly int, but TVM golden is int
    c = np.load(cmod_file).astype(np.float64)
    
    # TVM's embed_add_pos might not exist in that exact name. Let's handle known names
    if t.shape != c.shape:
        try:
            t = t.reshape(c.shape)
        except:
            pass
            
    diff = t - c
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    num_diff = np.sum(diff != 0)
    total = diff.size
    
    print(f"{base_name:<30} | {max_diff:<10.1f} | {mean_diff:<10.4f} | {num_diff:>10} / {total:<7} | {num_diff/total*100:<6.2f}%")

print("-" * 90)

# Check classification accuracy
tvm_head_file = os.path.join(TVM_DIR, "head_int.npy")
cmod_head_file = os.path.join(CMOD_DIR, "head_int.npy")

if os.path.exists(tvm_head_file) and os.path.exists(cmod_head_file):
    tvm_head = np.load(tvm_head_file)
    cmod_head = np.load(cmod_head_file)
    
    tvm_pred = np.argmax(tvm_head)
    cmod_pred = np.argmax(cmod_head)
    
    print(f"TVM Prediction Class: {tvm_pred} (Score: {tvm_head[0, tvm_pred] if tvm_head.ndim == 2 else tvm_head[tvm_pred]})")
    print(f"C-Model Prediction Class: {cmod_pred} (Score: {cmod_head[0, cmod_pred] if cmod_head.ndim == 2 else cmod_head[cmod_pred]})")
    
    tvm_top5 = np.argsort(tvm_head.flatten())[-5:][::-1]
    cmod_top5 = np.argsort(cmod_head.flatten())[-5:][::-1]
    print(f"TVM Top-5: {tvm_top5}")
    print(f"CMod Top-5: {cmod_top5}")
