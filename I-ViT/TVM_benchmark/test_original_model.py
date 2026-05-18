#!/usr/bin/env python3
"""
使用論文的原始預訓練模型測試 TVM
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os, sys, time

def float_to_int8(x_f32, input_scale):
    return np.round(x_f32.numpy() / input_scale).clip(-128, 127).astype('int8')

print("="*80)
print("測試論文原始模型 - 100 張圖片")
print("="*80)
print()

# 保存原始工作目錄
original_cwd = os.getcwd()

# Data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Step 1: 加載數據集...")
imagenet_path = os.path.join('..', '..', 'ImageNet')
if 'IMAGENET_PATH' in os.environ:
    imagenet_path = os.environ['IMAGENET_PATH']
else:
    imagenet_path = os.path.abspath(imagenet_path)
    
print(f"  ImageNet 路徑: {imagenet_path}")

dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform=transform)

# Use 100 images
indices = list(range(100))
subset = Subset(dataset, indices)
test_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

print(f"  使用 {len(subset)} 張圖片")
print()

# Load PyTorch model with ORIGINAL checkpoint
print("Step 2: 加載 PyTorch 模型（論文原始模型）...")
os.chdir('..')
sys.path.insert(0, os.getcwd())
from models.vit_quant import deit_tiny_patch16_224

# 使用論文的原始 checkpoint
checkpoint_path = os.path.join(original_cwd, '..', 'checkpoints', 'qat_calibrated.pth')
print(f"  使用 checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
pt_model = deit_tiny_patch16_224(pretrained=False)
ms = pt_model.state_dict()
ld = {k: v for k, v in model_dict.items()
      if k in ms and getattr(v, 'shape', None) == ms[k].shape}
pt_model.load_state_dict(ld, strict=False)
pt_model.eval()

# Get input scale
first_img, _ = subset[0]
first_img = first_img.unsqueeze(0)

calib_scales = {}
def hook_qact_input(mod, inp, out):
    calib_scales['qact_input.act_scaling_factor'] = float(out[1].flatten()[0])
pt_model.qact_input.register_forward_hook(hook_qact_input)

with torch.no_grad():
    _ = pt_model(first_img)
input_scale = calib_scales['qact_input.act_scaling_factor']

print(f"  Input scale: {input_scale:.8f}")
print()

# Build TVM model
print("Step 3: 構建 TVM 模型...")
os.chdir(os.path.join(original_cwd))
sys.path.pop(0)
for k in [k for k in sys.modules if k.startswith('models')]:
    del sys.modules[k]

import tvm
from tvm import relay
import models.build_model as build_model
import convert_model

# 檢查是否需要重新生成 params.npy 和 calibrated_scales.npy
params_file = 'params_original.npy'
scales_file = 'calibrated_scales_original.npy'

if not os.path.exists(params_file):
    print(f"  生成 {params_file}...")
    os.system(f'python convert_model.py --model-path ../checkpoints/qat_calibrated.pth --params-path . --depth 12')
    # 重命名
    if os.path.exists('params.npy'):
        os.rename('params.npy', params_file)

if not os.path.exists(scales_file):
    print(f"  生成 {scales_file}...")
    os.system(f"python generate_calibrated_scales.py --model-path '../checkpoints/qat_calibrated.pth' --output {scales_file} --image '../../data/test_image.JPEG'")

# 加載配置
convert_model.load_qconfig(model_dict, 12, calib_scales_file=scales_file)

func, _ = build_model.get_workload(
    name='deit_tiny_patch16_224',
    batch_size=1, image_shape=(3, 224, 224),
    dtype='int8', data_layout='NCHW', kernel_layout='OIHW',
)

pretrained_params = np.load(params_file, allow_pickle=True)[()]

print("  編譯中...")
t0 = time.time()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target='llvm', params={**pretrained_params})
print(f"  編譯完成 ({time.time()-t0:.1f}s)")
print()

dev = tvm.device('llvm', 0)
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

# Test on 100 images
print("Step 4: 測試 100 張圖片...")
print()

pt_correct = 0
tvm_correct = 0
both_correct = 0

for idx, (img, target) in enumerate(test_loader):
    target = target.item()
    
    # PyTorch inference
    with torch.no_grad():
        pt_logits = pt_model(img)[0].numpy()
    
    pt_probs = F.softmax(torch.tensor(pt_logits), dim=-1).numpy()
    pt_pred = np.argmax(pt_probs)
    
    # TVM inference
    x_int8 = float_to_int8(img, input_scale)
    runtime.set_input('data', x_int8)
    runtime.run()
    
    tvm_probs = runtime.get_output(0).numpy()[0]
    tvm_pred = np.argmax(tvm_probs)
    
    # Check accuracy
    if pt_pred == target:
        pt_correct += 1
    if tvm_pred == target:
        tvm_correct += 1
    if pt_pred == target and tvm_pred == target:
        both_correct += 1
    
    # Debug first image
    if idx == 0:
        print(f"  [DEBUG] 第一張圖片:")
        print(f"    Target: {target}")
        print(f"    PyTorch pred: {pt_pred}, prob={pt_probs[pt_pred]:.4f}")
        print(f"    TVM pred: {tvm_pred}, prob={tvm_probs[tvm_pred]:.4f}")
        print(f"    TVM probs range: [{tvm_probs.min():.4f}, {tvm_probs.max():.4f}]")
        print()
    
    if (idx + 1) % 10 == 0:
        print(f"  處理 {idx+1}/100 張圖片...")

print()
print("="*80)
print("結果（論文原始模型）")
print("="*80)
print()

print(f"PyTorch 結果:")
print(f"  Top-1 準確率: {pt_correct}/100 = {pt_correct}%")
print()

print(f"TVM 結果:")
print(f"  Top-1 準確率: {tvm_correct}/100 = {tvm_correct}%")
print()

print(f"一致性:")
print(f"  兩者都正確: {both_correct}/100 = {both_correct}%")
print()

print("="*80)
print("分析")
print("="*80)
print()

if tvm_correct >= 60:
    print(f"✅ 成功！TVM 準確率 {tvm_correct}% >= 60%")
    print()
    print("這證明:")
    print("  1. TVM 實現是正確的")
    print("  2. 論文的預訓練模型可以正常工作")
    print("  3. 問題在於 GPU 訓練的模型")
elif tvm_correct > 0:
    print(f"⚠️  TVM 準確率 {tvm_correct}% > 0%")
    print()
    print("TVM 有一些預測能力，但準確率不夠高")
else:
    print(f"❌ TVM 準確率 = 0%")
    print()
    print("即使使用論文的原始模型，TVM 仍然失敗")
    print("這表示 TVM 實現本身有問題")

print()
print("="*80)
