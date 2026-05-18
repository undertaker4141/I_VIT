#!/usr/bin/env python3
"""
診斷 TVM 推理問題
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import os, sys

def float_to_int8(x_f32, input_scale):
    return np.round(x_f32.numpy() / input_scale).clip(-128, 127).astype('int8')

print("="*80)
print("TVM 推理診斷")
print("="*80)
print()

# 使用單張測試圖片
test_image = '../../data/test_image.JPEG'
if not os.path.exists(test_image):
    print(f"✗ 找不到測試圖片: {test_image}")
    sys.exit(1)

print(f"✓ 使用測試圖片: {test_image}")
print()

# Data transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

img = Image.open(test_image).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

print("Step 1: 加載 PyTorch 模型")
os.chdir('..')
sys.path.insert(0, os.getcwd())
from models.vit_quant import deit_tiny_patch16_224

checkpoint = torch.load('output_gpu/checkpoint.pth', map_location='cpu', weights_only=False)
model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
pt_model = deit_tiny_patch16_224(pretrained=False)
ms = pt_model.state_dict()
ld = {k: v for k, v in model_dict.items()
      if k in ms and getattr(v, 'shape', None) == ms[k].shape}
pt_model.load_state_dict(ld, strict=False)
pt_model.eval()

# Get input scale
calib_scales = {}
def hook_qact_input(mod, inp, out):
    calib_scales['qact_input.act_scaling_factor'] = float(out[1].flatten()[0])
pt_model.qact_input.register_forward_hook(hook_qact_input)

with torch.no_grad():
    pt_output = pt_model(img_tensor)
    pt_logits = pt_output[0].numpy()

input_scale = calib_scales['qact_input.act_scaling_factor']
pt_probs = F.softmax(torch.tensor(pt_logits), dim=-1).numpy()
pt_pred = np.argmax(pt_probs)

print(f"  Input scale: {input_scale:.8f}")
print(f"  PyTorch 預測: {pt_pred}")
print(f"  PyTorch 最高概率: {pt_probs[pt_pred]:.4f}")
print(f"  PyTorch logits 範圍: [{pt_logits.min():.2f}, {pt_logits.max():.2f}]")
print()

# Build TVM model
print("Step 2: 構建 TVM 模型")
os.chdir('TVM_benchmark')
sys.path.pop(0)
for k in [k for k in sys.modules if k.startswith('models')]:
    del sys.modules[k]

import tvm
from tvm import relay
import models.build_model as build_model
import convert_model

# 加載 calibrated scales
calib_scales_file = 'calibrated_scales_gpu.npy'
if not os.path.exists(calib_scales_file):
    calib_scales_file = 'calibrated_scales.npy'

print(f"  使用 calibrated scales: {calib_scales_file}")
convert_model.load_qconfig(model_dict, 12, calib_scales_file=calib_scales_file)

func, _ = build_model.get_workload(
    name='deit_tiny_patch16_224',
    batch_size=1, image_shape=(3, 224, 224),
    dtype='int8', data_layout='NCHW', kernel_layout='OIHW',
)

params_path = 'params.npy'
pretrained_params = np.load(params_path, allow_pickle=True)[()]

print("  編譯 TVM 模型...")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target='llvm', params={**pretrained_params})

dev = tvm.device('llvm', 0)
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
print("  ✓ TVM 模型編譯完成")
print()

# TVM inference
print("Step 3: TVM 推理")
x_int8 = float_to_int8(img_tensor, input_scale)
print(f"  輸入 int8 範圍: [{x_int8.min()}, {x_int8.max()}]")

runtime.set_input('data', x_int8)
runtime.run()

tvm_output = runtime.get_output(0).numpy()[0]
tvm_pred = np.argmax(tvm_output)

print(f"  TVM 預測: {tvm_pred}")
print(f"  TVM 最高概率: {tvm_output[tvm_pred]:.4f}")
print(f"  TVM 輸出範圍: [{tvm_output.min():.4f}, {tvm_output.max():.4f}]")
print(f"  TVM 輸出總和: {tvm_output.sum():.4f}")
print()

# 比較
print("="*80)
print("比較")
print("="*80)
print(f"PyTorch 預測: {pt_pred}, 概率: {pt_probs[pt_pred]:.4f}")
print(f"TVM 預測: {tvm_pred}, 概率: {tvm_output[tvm_pred]:.4f}")
print()

if pt_pred == tvm_pred:
    print("✓ 預測一致！")
else:
    print("✗ 預測不一致")
    print()
    print("PyTorch Top-5:")
    pt_top5 = np.argsort(pt_probs)[::-1][:5]
    for i, idx in enumerate(pt_top5):
        print(f"  {i+1}. Class {idx}: {pt_probs[idx]:.4f}")
    
    print()
    print("TVM Top-5:")
    tvm_top5 = np.argsort(tvm_output)[::-1][:5]
    for i, idx in enumerate(tvm_top5):
        print(f"  {i+1}. Class {idx}: {tvm_output[idx]:.4f}")

print()

# 診斷
if tvm_output.max() < 0.05:
    print("⚠️  TVM 輸出幾乎是均勻分佈（最大概率 <5%）")
    print("   這表示模型沒有學到任何東西，可能的原因：")
    print("   1. Calibrated scales 沒有正確應用")
    print("   2. 模型權重沒有正確加載")
    print("   3. TVM 編譯有問題")
    print()
    print("   建議檢查:")
    print("   - convert_model.load_qconfig() 是否正確執行")
    print("   - params.npy 是否包含正確的權重")
    print("   - 是否需要重新生成 params.npy")

print("="*80)
