import numpy as np

t = np.load('/home/undertaker4141/I_VIT/patterns_tvm/golden/embed_add_pos.npy')
c = np.load('/home/undertaker4141/I_VIT/patterns_cmodel_golden/golden/embed_add_pos_int.npy')

print("TVM shape:", t.shape, " dtype:", t.dtype)
print("TVM values:", t.flatten()[:20])
print("TVM min:", t.min(), "max:", t.max())

print("\nC-Model shape:", c.shape, "dtype:", c.dtype)
print("C-Model values:", c.flatten()[:20])
print("C-Model min:", c.min(), "max:", c.max())

# Ensure t gets cast properly for diff
diff = t.flatten()[:20].astype(np.float64) - c.flatten()[:20].astype(np.float64)
print("\nDiff first 20:", diff)
