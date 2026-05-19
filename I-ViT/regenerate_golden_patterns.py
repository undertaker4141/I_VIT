"""
重新生成 Golden Patterns
使用新的測試圖片和修復後的 C-Model
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

print("="*80)
print("重新生成 Golden Patterns")
print("="*80)

# 載入模型
print("\n[1/5] 載入模型...")
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
print("  ✓ 模型已載入")

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 載入測試圖片
print("\n[2/5] 載入測試圖片...")
image_path = '../data/test_image.JPEG'
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)
print(f"  ✓ 圖片已載入: {image_path}")

# 讀取 ground truth
with open('../data/test_image_info.txt', 'r') as f:
    lines = f.readlines()
    ground_truth = int(lines[2].split(': ')[1].strip())
print(f"  Ground Truth: {ground_truth}")

# 執行推論並收集 golden patterns
print("\n[3/5] 執行推論並收集 Golden Patterns...")

golden_patterns = {}

with torch.no_grad():
    # Input processing
    x, input_scaling_factor = model.qact_input(image_tensor)
    x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
    cls_tokens = model.cls_token.expand(1, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
    x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
    
    # 保存輸入
    golden_patterns['input'] = {
        'x_int': x.cpu().numpy(),
        'x_sf': x_scaling_factor.cpu().numpy() if isinstance(x_scaling_factor, torch.Tensor) else x_scaling_factor
    }
    print(f"  ✓ 已保存輸入 pattern")
    
    # 對每個 block 收集 patterns
    for block_idx, block in enumerate(model.blocks):
        block_patterns = {}
        
        # Norm1
        x_norm1, norm1_sf = block.norm1(x, x_scaling_factor)
        x_norm1, qact1_sf = block.qact1(x_norm1, norm1_sf)
        block_patterns['norm1_output'] = {
            'x_int': x_norm1.cpu().numpy(),
            'x_sf': qact1_sf.cpu().numpy() if isinstance(qact1_sf, torch.Tensor) else qact1_sf
        }
        
        # Attention
        attn_output, attn_sf = block.attn(x_norm1, qact1_sf)
        block_patterns['attn_output'] = {
            'x_int': attn_output.cpu().numpy(),
            'x_sf': attn_sf.cpu().numpy() if isinstance(attn_sf, torch.Tensor) else attn_sf
        }
        
        # Residual 1
        x, x_sf = block.qact2(x, x_scaling_factor, attn_output, attn_sf)
        block_patterns['residual1_output'] = {
            'x_int': x.cpu().numpy(),
            'x_sf': x_sf.cpu().numpy() if isinstance(x_sf, torch.Tensor) else x_sf
        }
        
        # Norm2
        x_norm2, norm2_sf = block.norm2(x, x_sf)
        x_norm2, qact3_sf = block.qact3(x_norm2, norm2_sf)
        block_patterns['norm2_output'] = {
            'x_int': x_norm2.cpu().numpy(),
            'x_sf': qact3_sf.cpu().numpy() if isinstance(qact3_sf, torch.Tensor) else qact3_sf
        }
        
        # MLP
        mlp_output, mlp_sf = block.mlp(x_norm2, qact3_sf)
        block_patterns['mlp_output'] = {
            'x_int': mlp_output.cpu().numpy(),
            'x_sf': mlp_sf.cpu().numpy() if isinstance(mlp_sf, torch.Tensor) else mlp_sf
        }
        
        # Residual 2
        x, x_sf = block.qact4(x, x_sf, mlp_output, mlp_sf)
        block_patterns['residual2_output'] = {
            'x_int': x.cpu().numpy(),
            'x_sf': x_sf.cpu().numpy() if isinstance(x_sf, torch.Tensor) else x_sf
        }
        
        golden_patterns[f'block_{block_idx}'] = block_patterns
        
        if block_idx % 3 == 0 or block_idx == len(model.blocks) - 1:
            print(f"  ✓ 已保存 Block {block_idx} patterns")
        
        x_scaling_factor = x_sf
    
    # Final norm
    x, x_sf = model.norm(x, x_scaling_factor)
    golden_patterns['final_norm_output'] = {
        'x_int': x.cpu().numpy(),
        'x_sf': x_sf.cpu().numpy() if isinstance(x_sf, torch.Tensor) else x_sf
    }
    
    # CLS token
    x = x[:, 0]
    x, x_sf = model.qact2(x, x_sf)
    
    # Head
    x, x_sf = model.head(x, x_sf)
    
    # 最終輸出
    logits = x.cpu().numpy()[0]
    pred_class = np.argmax(logits)
    
    golden_patterns['output'] = {
        'logits': logits,
        'pred_class': pred_class,
        'ground_truth': ground_truth
    }
    
    print(f"  ✓ 已保存最終輸出")
    print(f"    預測類別: {pred_class}")
    print(f"    Ground Truth: {ground_truth}")
    print(f"    正確: {pred_class == ground_truth}")

# 保存 golden patterns
print("\n[4/5] 保存 Golden Patterns...")
output_dir = '../golden_patterns'
os.makedirs(output_dir, exist_ok=True)

np.savez(
    os.path.join(output_dir, 'golden_patterns.npz'),
    **{k: v for k, v in golden_patterns.items() if k != 'output'},
    **golden_patterns['output']
)

print(f"  ✓ 已保存到 {output_dir}/golden_patterns.npz")

# 生成報告
print("\n[5/5] 生成報告...")
report_path = os.path.join(output_dir, 'golden_patterns_report.txt')
with open(report_path, 'w') as f:
    f.write("Golden Patterns 報告\n")
    f.write("="*80 + "\n\n")
    f.write(f"生成日期: 2026-05-19\n")
    f.write(f"測試圖片: {image_path}\n")
    f.write(f"Ground Truth: {ground_truth}\n")
    f.write(f"預測類別: {pred_class}\n")
    f.write(f"預測正確: {pred_class == ground_truth}\n\n")
    
    f.write("包含的 Patterns:\n")
    f.write("-"*80 + "\n")
    f.write("1. input: 輸入到 Transformer Blocks\n")
    for i in range(12):
        f.write(f"2. block_{i}:\n")
        f.write(f"   - norm1_output\n")
        f.write(f"   - attn_output\n")
        f.write(f"   - residual1_output\n")
        f.write(f"   - norm2_output\n")
        f.write(f"   - mlp_output\n")
        f.write(f"   - residual2_output\n")
    f.write("3. final_norm_output: Final LayerNorm 輸出\n")
    f.write("4. output: 最終 logits 和預測\n")

print(f"  ✓ 報告已保存到 {report_path}")

print("\n" + "="*80)
print("✓ Golden Patterns 重新生成完成")
print("="*80)
