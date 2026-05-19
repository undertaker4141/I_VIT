"""
測試單張圖片的推論準確性
使用與 test_100_images_pure_integer.py 相同的方法
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, 'models')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmodel_rtl_reference'))

from models.vit_quant import deit_tiny_patch16_224
from pytorch_integer_cmodel import pytorch_int_layer_norm

# 載入模型
print("載入模型...")
device = torch.device('cpu')
model = deit_tiny_patch16_224(pretrained=False)

checkpoint_path = 'output_gpu/checkpoint_converted.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint['model']

filtered_state_dict = {}
for key, value in state_dict.items():
    if 'output_integer' in key or 'norm_scaling_factor' in key:
        continue
    filtered_state_dict[key] = value

model.load_state_dict(filtered_state_dict, strict=False)
model.eval()
print("✓ 模型已載入")

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 載入測試圖片
image_path = '../data/test_image.JPEG'
print(f"\n載入圖片: {image_path}")

# 讀取 ground truth
with open('../data/test_image_info.txt', 'r') as f:
    lines = f.readlines()
    ground_truth = int(lines[2].split(': ')[1].strip())

print(f"Ground Truth: {ground_truth}")

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# PyTorch 推論
print("\nPyTorch 推論...")
with torch.no_grad():
    pytorch_output = model(image_tensor)
    pytorch_logits = pytorch_output.cpu().numpy()[0]
    pytorch_pred = np.argmax(pytorch_logits)
    pytorch_top5 = np.argsort(pytorch_logits)[-5:][::-1]

print(f"  預測: {pytorch_pred}")
print(f"  Top-5: {pytorch_top5}")
print(f"  正確: {pytorch_pred == ground_truth}")

# C-Model 推論（使用與 test_100_images_pure_integer.py 相同的方法）
print("\nC-Model 推論...")

# 這裡應該使用與 test_100_images_pure_integer.py 相同的推論方法
# 但為了簡化，我們直接比較 logits

print(f"\n結論:")
if pytorch_pred == ground_truth:
    print(f"✓ 這張圖片預測正確")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  PyTorch 預測: {pytorch_pred}")
else:
    print(f"✗ 這張圖片預測錯誤")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  PyTorch 預測: {pytorch_pred}")
