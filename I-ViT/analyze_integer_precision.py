"""
分析 PyTorch 模型中各層使用的整數精度
==================================================================
目標：確定每一層的整數輸入/輸出範圍，以便在 C-Model 中使用正確的精度
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.vit_quant import deit_tiny_patch16_224


def analyze_precision(model, image_tensor):
    """分析每一層的整數精度"""
    print("="*80)
    print("分析整數精度")
    print("="*80)
    
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # ============================================================
        # Stage 1: Input Processing
        # ============================================================
        print(f"\n[Stage 1] Input Processing")
        
        # Input quantization
        x, input_scaling_factor = model.qact_input(image_tensor)
        x_int = (x / input_scaling_factor).cpu().numpy()
        print(f"  qact_input:")
        print(f"    整數範圍: [{x_int.min():.0f}, {x_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(x_int.min()) <= 128 and x_int.max() <= 127 else f"    建議精度: int16")
        
        # Patch embedding
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        x_int = (x / patch_scaling_factor).cpu().numpy()
        print(f"  patch_embed:")
        print(f"    整數範圍: [{x_int.min():.0f}, {x_int.max():.0f}]")
        print(f"    建議精度: int16" if abs(x_int.min()) <= 32768 and x_int.max() <= 32767 else f"    建議精度: int32")
        
        # Add CLS token
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        x_int = (x / x_scaling_factor).cpu().numpy()
        print(f"  qact1 (residual add):")
        print(f"    整數範圍: [{x_int.min():.0f}, {x_int.max():.0f}]")
        print(f"    建議精度: int16" if abs(x_int.min()) <= 32768 and x_int.max() <= 32767 else f"    建議精度: int32")
        
        # ============================================================
        # Stage 2: First Transformer Block (詳細分析)
        # ============================================================
        print(f"\n[Stage 2] Block 0 (詳細分析)")
        
        block = model.blocks[0]
        
        # Norm1
        x_norm1, norm1_scaling_factor = block.norm1(x, x_scaling_factor)
        x_norm1_int = (x_norm1 / norm1_scaling_factor).cpu().numpy()
        print(f"  norm1:")
        print(f"    整數範圍: [{x_norm1_int.min():.0f}, {x_norm1_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # qact1
        x_qact1, qact1_scaling_factor = block.qact1(x_norm1, norm1_scaling_factor)
        x_qact1_int = (x_qact1 / qact1_scaling_factor).cpu().numpy()
        print(f"  qact1:")
        print(f"    整數範圍: [{x_qact1_int.min():.0f}, {x_qact1_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(x_qact1_int.min()) <= 128 and x_qact1_int.max() <= 127 else f"    建議精度: int16")
        
        # QKV
        x_qkv, qkv_scaling_factor = block.attn.qkv(x_qact1, qact1_scaling_factor)
        x_qkv_int = (x_qkv / qkv_scaling_factor).cpu().numpy()
        print(f"  attn.qkv:")
        print(f"    整數範圍: [{x_qkv_int.min():.0f}, {x_qkv_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # qact1 (after QKV)
        x_qkv_qact, qkv_qact_scaling_factor = block.attn.qact1(x_qkv, qkv_scaling_factor)
        x_qkv_qact_int = (x_qkv_qact / qkv_qact_scaling_factor).cpu().numpy()
        print(f"  attn.qact1:")
        print(f"    整數範圍: [{x_qkv_qact_int.min():.0f}, {x_qkv_qact_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(x_qkv_qact_int.min()) <= 128 and x_qkv_qact_int.max() <= 127 else f"    建議精度: int16")
        
        # Reshape to Q, K, V
        N = x.shape[1]
        C = x.shape[2]
        qkv = x_qkv_qact.reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Q @ K^T
        attn_scores, attn_scaling_factor = block.attn.matmul_1(q, qkv_qact_scaling_factor,
                                                               k.transpose(-2, -1), qkv_qact_scaling_factor)
        attn_scores = attn_scores * block.attn.scale
        attn_scaling_factor = attn_scaling_factor * block.attn.scale
        attn_scores, attn_scaling_factor = block.attn.qact_attn1(attn_scores, attn_scaling_factor)
        attn_scores_int = (attn_scores / attn_scaling_factor).cpu().numpy()
        print(f"  attn.matmul_1 (Q@K^T):")
        print(f"    整數範圍: [{attn_scores_int.min():.0f}, {attn_scores_int.max():.0f}]")
        print(f"    建議精度: int16" if abs(attn_scores_int.min()) <= 32768 and attn_scores_int.max() <= 32767 else f"    建議精度: int32")
        
        # Softmax
        attn_softmax, softmax_scaling_factor = block.attn.int_softmax(attn_scores, attn_scaling_factor)
        attn_softmax_int = (attn_softmax / softmax_scaling_factor).cpu().numpy()
        print(f"  attn.softmax:")
        print(f"    整數範圍: [{attn_softmax_int.min():.0f}, {attn_softmax_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(attn_softmax_int.min()) <= 128 and attn_softmax_int.max() <= 127 else f"    建議精度: int16")
        
        # Attn @ V
        x_attn, attn_out_scaling_factor = block.attn.matmul_2(attn_softmax, softmax_scaling_factor,
                                                              v, qkv_qact_scaling_factor)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn_int = (x_attn / attn_out_scaling_factor).cpu().numpy()
        print(f"  attn.matmul_2 (Attn@V):")
        print(f"    整數範圍: [{x_attn_int.min():.0f}, {x_attn_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # Attention output projection
        x_attn, attn_out_scaling_factor = block.attn.qact2(x_attn, attn_out_scaling_factor)
        x_attn, attn_out_scaling_factor = block.attn.proj(x_attn, attn_out_scaling_factor)
        x_attn, attn_out_scaling_factor = block.attn.qact3(x_attn, attn_out_scaling_factor)
        x_attn_int = (x_attn / attn_out_scaling_factor).cpu().numpy()
        print(f"  attn.proj:")
        print(f"    整數範圍: [{x_attn_int.min():.0f}, {x_attn_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(x_attn_int.min()) <= 128 and x_attn_int.max() <= 127 else f"    建議精度: int16")
        
        # Residual 1
        x_res1, res1_scaling_factor = block.qact2(x_attn, attn_out_scaling_factor, x, x_scaling_factor)
        x_res1_int = (x_res1 / res1_scaling_factor).cpu().numpy()
        print(f"  qact2 (residual 1):")
        print(f"    整數範圍: [{x_res1_int.min():.0f}, {x_res1_int.max():.0f}]")
        print(f"    建議精度: int16 ⚠ 殘差連接" if abs(x_res1_int.min()) <= 32768 and x_res1_int.max() <= 32767 else f"    建議精度: int32 ⚠ 殘差連接")
        
        # Norm2
        x_norm2, norm2_scaling_factor = block.norm2(x_res1, res1_scaling_factor)
        x_norm2_int = (x_norm2 / norm2_scaling_factor).cpu().numpy()
        print(f"  norm2:")
        print(f"    整數範圍: [{x_norm2_int.min():.0f}, {x_norm2_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # qact3
        x_qact3, qact3_scaling_factor = block.qact3(x_norm2, norm2_scaling_factor)
        x_qact3_int = (x_qact3 / qact3_scaling_factor).cpu().numpy()
        print(f"  qact3:")
        print(f"    整數範圍: [{x_qact3_int.min():.0f}, {x_qact3_int.max():.0f}]")
        print(f"    建議精度: int8" if abs(x_qact3_int.min()) <= 128 and x_qact3_int.max() <= 127 else f"    建議精度: int16")
        
        # MLP fc1
        x_mlp, mlp_scaling_factor = block.mlp.fc1(x_qact3, qact3_scaling_factor)
        x_mlp_int = (x_mlp / mlp_scaling_factor).cpu().numpy()
        print(f"  mlp.fc1:")
        print(f"    整數範圍: [{x_mlp_int.min():.0f}, {x_mlp_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # GELU
        x_gelu, gelu_scaling_factor = block.mlp.act(x_mlp, mlp_scaling_factor)
        x_gelu_int = (x_gelu / gelu_scaling_factor).cpu().numpy()
        print(f"  mlp.gelu:")
        print(f"    整數範圍: [{x_gelu_int.min():.0f}, {x_gelu_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # MLP fc2
        x_mlp2, mlp2_scaling_factor = block.mlp.fc2(x_gelu, gelu_scaling_factor)
        x_mlp2_int = (x_mlp2 / mlp2_scaling_factor).cpu().numpy()
        print(f"  mlp.fc2:")
        print(f"    整數範圍: [{x_mlp2_int.min():.0f}, {x_mlp2_int.max():.0f}]")
        print(f"    建議精度: int32")
        
        # Residual 2
        x_res2, res2_scaling_factor = block.qact4(x_mlp2, mlp2_scaling_factor, x_res1, res1_scaling_factor)
        x_res2_int = (x_res2 / res2_scaling_factor).cpu().numpy()
        print(f"  qact4 (residual 2):")
        print(f"    整數範圍: [{x_res2_int.min():.0f}, {x_res2_int.max():.0f}]")
        print(f"    建議精度: int16 ⚠ 殘差連接" if abs(x_res2_int.min()) <= 32768 and x_res2_int.max() <= 32767 else f"    建議精度: int32 ⚠ 殘差連接")
        
        print(f"\n" + "="*80)
        print(f"總結")
        print(f"="*80)
        print(f"✓ 大部分層使用 int8 或 int16")
        print(f"✓ LayerNorm 輸出使用 int32")
        print(f"✓ 線性層輸出使用 int32")
        print(f"⚠ 殘差連接可能需要 int16 或 int32")


def main():
    """主函數"""
    print("="*80)
    print("分析 PyTorch 模型的整數精度")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    
    # 載入模型
    print("\n[1/3] 載入模型...")
    device = torch.device('cpu')
    model = deit_tiny_patch16_224(pretrained=False)
    
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
    
    # 載入圖片
    print("\n[2/3] 載入圖片...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print("✓ 圖片已載入")
    
    # 執行一次前向傳播
    print("\n[3/3] 初始化模型參數...")
    with torch.no_grad():
        _ = model(image_tensor)
    print("✓ 參數已初始化")
    
    # 分析精度
    analyze_precision(model, image_tensor)


if __name__ == '__main__':
    main()
