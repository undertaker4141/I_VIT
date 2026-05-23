import numpy as np

# 載入 golden patterns
data = np.load('golden_patterns/golden_patterns.npz', allow_pickle=True)

print("=" * 80)
print("Golden Patterns 詳細結構")
print("=" * 80)
print()

# 檢查 input
print("1. Input:")
input_data = data['input'].item()
if isinstance(input_data, dict):
    print("   Type: dict")
    for k, v in input_data.items():
        if isinstance(v, np.ndarray):
            print(f"   - {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min()}, {v.max()}]")
else:
    print(f"   Type: {type(input_data)}")
    if isinstance(input_data, np.ndarray):
        print(f"   shape={input_data.shape}, dtype={input_data.dtype}")

print()

# 檢查 block_0
print("2. Block 0:")
block_0 = data['block_0'].item()
if isinstance(block_0, dict):
    print("   Type: dict")
    for k, v in block_0.items():
        if isinstance(v, np.ndarray):
            print(f"   - {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min()}, {v.max()}]")
else:
    print(f"   Type: {type(block_0)}")

print()

# 檢查 final_norm_output
print("3. Final Norm Output:")
final_norm = data['final_norm_output'].item()
if isinstance(final_norm, np.ndarray):
    print(f"   Type: ndarray")
    print(f"   shape={final_norm.shape}, dtype={final_norm.dtype}, range=[{final_norm.min()}, {final_norm.max()}]")
else:
    print(f"   Type: {type(final_norm)}")

