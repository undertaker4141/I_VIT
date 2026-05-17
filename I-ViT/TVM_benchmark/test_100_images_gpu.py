"""
測試 GPU 訓練後的模型 - 100 張圖片
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
print("測試 GPU 訓練模型 - 100 張圖片")
print("="*80)
print()

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
imagenet_path = '/mnt/c/桌面/冠泓/大學/專題/I-ViT/I_VIT/ImageNet'
dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform=transform)

# Use 100 images
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

checkpoint = torch.load('output_gpu/checkpoint.pth', map_location='cpu', weights_only=False)
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
os.chdir('TVM_benchmark')
sys.path.pop(0)
for k in [k for k in sys.modules if k.startswith('models')]:
    del sys.modules[k]

import tvm
from tvm import relay
import models.build_model as build_model
import convert_model

convert_model.load_qconfig(model_dict, 12, calib_scales_file='calibrated_scales_gpu.npy')

func, _ = build_model.get_workload(
    name='deit_tiny_patch16_224',
    batch_size=1, image_shape=(3, 224, 224),
    dtype='int8', data_layout='NCHW', kernel_layout='OIHW',
)

pretrained_params = np.load('params.npy', allow_pickle=True)[()]

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
pt_top5_correct = 0
tvm_top5_correct = 0

pt_time = 0
tvm_time = 0

for idx, (img, target) in enumerate(test_loader):
    target = target.item()
    
    # PyTorch inference
    t0 = time.time()
    with torch.no_grad():
        pt_logits = pt_model(img)[0].numpy()
    pt_time += time.time() - t0
    
    pt_probs = F.softmax(torch.tensor(pt_logits), dim=-1).numpy()
    pt_pred = np.argmax(pt_probs)
    pt_top5 = np.argsort(pt_probs)[::-1][:5]
    
    # TVM inference
    x_int8 = float_to_int8(img, input_scale)
    runtime.set_input('data', x_int8)
    
    t0 = time.time()
    runtime.run()
    tvm_time += time.time() - t0
    
    tvm_probs = runtime.get_output(0).numpy()[0]
    tvm_pred = np.argmax(tvm_probs)
    tvm_top5 = np.argsort(tvm_probs)[::-1][:5]
    
    # Check accuracy
    if pt_pred == target:
        pt_correct += 1
    if tvm_pred == target:
        tvm_correct += 1
    if pt_pred == target and tvm_pred == target:
        both_correct += 1
    if target in pt_top5:
        pt_top5_correct += 1
    if target in tvm_top5:
        tvm_top5_correct += 1
    
    if (idx + 1) % 10 == 0:
        print(f"  處理 {idx+1}/100 張圖片...")

print()
print("="*80)
print("結果")
print("="*80)
print()

print(f"PyTorch 結果:")
print(f"  Top-1 準確率: {pt_correct}/100 = {pt_correct}%")
print(f"  Top-5 準確率: {pt_top5_correct}/100 = {pt_top5_correct}%")
print(f"  平均推論時間: {pt_time/100*1000:.1f}ms")
print()

print(f"TVM 結果:")
print(f"  Top-1 準確率: {tvm_correct}/100 = {tvm_correct}%")
print(f"  Top-5 準確率: {tvm_top5_correct}/100 = {tvm_top5_correct}%")
print(f"  平均推論時間: {tvm_time/100*1000:.1f}ms")
print()

print(f"一致性:")
print(f"  兩者都正確: {both_correct}/100 = {both_correct}%")
print()

# Save results
with open('../output_gpu/test_results.txt', 'w') as f:
    f.write(f"Test Results (100 images)\n")
    f.write(f"=========================\n\n")
    f.write(f"PyTorch:\n")
    f.write(f"  Top-1: {pt_correct}%\n")
    f.write(f"  Top-5: {pt_top5_correct}%\n\n")
    f.write(f"TVM:\n")
    f.write(f"  Top-1: {tvm_correct}%\n")
    f.write(f"  Top-5: {tvm_top5_correct}%\n\n")
    f.write(f"Both correct: {both_correct}%\n")

print("="*80)
print("分析")
print("="*80)
print()

if tvm_correct >= 60:
    print(f"✅✅✅ 成功！TVM 準確率 {tvm_correct}% >= 60%")
    print()
    print("這證明了：")
    print("  1. Integer-only 方向完全正確")
    print("  2. GPU 訓練非常有效")
    print("  3. 可以繼續部署到 RTL")
    print()
    print("下一步：")
    print("  1. 提取 golden patterns")
    print("  2. 驗證 C-Model")
    print("  3. 部署到 RTL 模擬")
elif tvm_correct > 0:
    print(f"✅ TVM 準確率 {tvm_correct}% > 0%")
    print()
    print("方向正確，但需要更多訓練：")
    print("  - 增加到 3-5 epochs")
    print("  - 或使用完整 train 數據集")
else:
    print(f"❌ TVM 準確率 = 0%")
    print()
    print("需要更多訓練：")
    print("  - 確認訓練完成")
    print("  - 增加 epochs")
    print("  - 檢查 calibrated scales")

print()
print(f"PyTorch vs TVM 差距: {pt_correct - tvm_correct}%")
print()
print("結果已保存到: output_gpu/test_results.txt")
print("="*80)
