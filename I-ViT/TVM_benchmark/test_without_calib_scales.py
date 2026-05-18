#!/usr/bin/env python3
"""
測試不使用 calibrated scales（直接用 checkpoint 的 scales）
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
print("測試不使用 Calibrated Scales")
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
imagenet_path = os.path.abspath(imagenet_path)
print(f"  ImageNet 路徑: {imagenet_path}")

dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform=transform)
indices = list(range(100))
subset = Subset(dataset, indices)
test_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

print(f"  使用 {len(subset)} 張圖片")
print()

# Load PyTorch model
print("Step 2: 加載 PyTorch 模型...")
os.chdir('..')
sys.path.insert(0, os.getcwd())
from models.vit_quant import deit_tiny_patch16_224

checkpoint_path = os.path.join(original_cwd, '..', 'checkpoints', 'qat_calibrated.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
pt_model = deit_tiny_patch16_224(pretrained=False)
ms = pt_model.state_dict()
ld = {k: v for k, v in model_dict.items()
      if k in ms and getattr(v, 'shape', None) == ms[k].shape}
pt_model.load_state_dict(ld, strict=False)
pt_model.eval()

# Get input scale from checkpoint (not from calibration!)
input_scale = float(model_dict['qact_input.act_scaling_factor'].cpu().numpy())
print(f"  Input scale (from checkpoint): {input_scale:.8f}")
print()

# Build TVM model
print("Step 3: 構建 TVM 模型（不使用 calibrated scales）...")
os.chdir(os.path.join(original_cwd))
sys.path.pop(0)
for k in [k for k in sys.modules if k.startswith('models')]:
    del sys.modules[k]

import tvm
from tvm import relay
import models.build_model as build_model
import convert_model

# 🔥 關鍵：不傳入 calib_scales_file，讓它使用 checkpoint 的 scales
print("  使用 checkpoint 中的 scaling factors（不覆蓋）")

# 臨時重命名 calibrated_scales 文件，避免自動加載
if os.path.exists('calibrated_scales.npy'):
    os.rename('calibrated_scales.npy', 'calibrated_scales.npy.bak')
if os.path.exists('calibrated_scales_original.npy'):
    os.rename('calibrated_scales_original.npy', 'calibrated_scales_original.npy.bak')

try:
    convert_model.load_qconfig(model_dict, 12)  # 不傳入任何 calib_scales
finally:
    # 恢復文件名
    if os.path.exists('calibrated_scales.npy.bak'):
        os.rename('calibrated_scales.npy.bak', 'calibrated_scales.npy')
    if os.path.exists('calibrated_scales_original.npy.bak'):
        os.rename('calibrated_scales_original.npy.bak', 'calibrated_scales_original.npy')

func, _ = build_model.get_workload(
    name='deit_tiny_patch16_224',
    batch_size=1, image_shape=(3, 224, 224),
    dtype='int8', data_layout='NCHW', kernel_layout='OIHW',
)

pretrained_params = np.load('params_original.npy', allow_pickle=True)[()]

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
print("結果（不使用 Calibrated Scales）")
print("="*80)
print()

print(f"PyTorch: {pt_correct}%")
print(f"TVM: {tvm_correct}%")
print(f"兩者都正確: {both_correct}%")
print()

if tvm_correct >= 60:
    print(f"✅✅✅ 成功！問題找到了！")
    print()
    print("Calibrated scales 是錯誤的，不應該使用！")
    print("應該直接使用 checkpoint 中的 scaling factors。")
elif tvm_correct > 0:
    print(f"⚠️  有改善，但還不夠")
else:
    print(f"❌ 仍然失敗，問題更深層")

print("="*80)
