"""
完整測試 Attention 的每一步
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmodel_rtl_reference'))

from models.vit_quant import deit_tiny_patch16_224
from pure_integer_end_to_end import pure_integer_attention, extract_weights_and_scales
from pure_numpy_cmodel import quantize_to_int


def extract_scaling_factor(sf):
    """提取 scaling factor"""
    if isinstance(sf, torch.Tensor):
        if sf.numel() == 1:
            return sf.item()
        else:
            return sf.cpu().numpy()
    else:
        return sf


def compare_tensors(name, pytorch_tensor, pytorch_sf, cmodel_tensor, cmodel_sf):
    """比較兩個 tensor"""
    # 反量化
    if isinstance(pytorch_tensor, torch.Tensor):
        pytorch_float = pytorch_tensor.cpu().numpy() * pytorch_sf
    else:
        pytorch_float = pytorch_tensor * pytorch_sf
    
    cmodel_float = cmodel_tensor * cmodel_sf
    
    # 計算相關係數
    corr = np.corrcoef(pytorch_float.flatten(), cmodel_float.flatten())[0, 1]
    
    # 計算差異
    diff = np.abs(pytorch_float - cmodel_float)
    
    print(f"\n{name}:")
    print(f"  PyTorch: range=[{pytorch_float.min():.4f}, {pytorch_float.max():.4f}]")
    print(f"  C-Model: range=[{cmodel_float.min():.4f}, {cmodel_float.max():.4f}]")
    print(f"  相關係數: {corr:.6f}")
    print(f"  最大差異: {diff.max():.4f}")
    print(f"  平均差異: {diff.mean():.4f}")
    
    if corr > 0.99:
        print(f"  ✓ 通過")
        return True
    else:
        print(f"  ✗ 失敗")
        return False


def main():
    print("="*80)
    print("完整測試 Attention")
    print("="*80)
    
    # 載入模型
    device = torch.device('cpu')
    model = deit_tiny_patch16_224(pretrained=False)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'output_gpu', 'checkpoint_converted.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'output_integer' in key or 'norm_scaling_factor' in key:
            continue
        filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    # 載入圖片
    test_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_image.JPEG')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # 提取權重
    weights = extract_weights_and_scales(model, image_tensor)
    block_weights = weights['block_0']
    
    # 執行到 QuantAct1
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        block = model.blocks[0]
        
        # Norm1
        x_norm1, norm1_sf = block.norm1(x, x_scaling_factor)
        
        # QuantAct1
        x_qact1, qact1_sf = block.qact1(x_norm1, norm1_sf)
        
        # Attention
        x_attn, attn_sf = block.attn(x_qact1, qact1_sf)
        
        print(f"\nPyTorch Attention 輸出:")
        print(f"  range=[{x_attn.min():.4f}, {x_attn.max():.4f}], sf={extract_scaling_factor(attn_sf):.6f}")
    
    # C-Model Attention
    print(f"\nC-Model Attention:")
    
    # 準備輸入
    x_qact1_np = x_qact1.cpu().numpy()
    x_qact1_sf_np = extract_scaling_factor(qact1_sf)
    x_qact1_int8 = quantize_to_int(x_qact1_np, x_qact1_sf_np, bits=8)
    
    print(f"  輸入: shape={x_qact1_int8.shape}, range=[{x_qact1_int8.min()}, {x_qact1_int8.max()}], sf={x_qact1_sf_np:.6f}")
    
    # 執行 C-Model Attention
    attn_output_int16, attn_output_sf = pure_integer_attention(
        x_qact1_int8, x_qact1_sf_np, block_weights
    )
    
    print(f"  輸出: shape={attn_output_int16.shape}, range=[{attn_output_int16.min()}, {attn_output_int16.max()}], sf={attn_output_sf:.6f}")
    
    # 比較
    print(f"\n" + "="*80)
    print("比較結果")
    print("="*80)
    
    success = compare_tensors(
        "Attention 輸出",
        x_attn, extract_scaling_factor(attn_sf),
        attn_output_int16, attn_output_sf
    )
    
    if success:
        print("\n" + "="*80)
        print("✓ Attention 測試通過！")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("✗ Attention 測試失敗")
        print("="*80)
        
        # 額外診斷
        pytorch_int = (x_attn.cpu().numpy() / extract_scaling_factor(attn_sf)).astype(np.int16)
        print(f"\n診斷資訊:")
        print(f"  PyTorch int16: range=[{pytorch_int.min()}, {pytorch_int.max()}]")
        print(f"  C-Model int16: range=[{attn_output_int16.min()}, {attn_output_int16.max()}]")
        
        int_corr = np.corrcoef(pytorch_int.flatten(), attn_output_int16.flatten())[0, 1]
        print(f"  整數值相關係數: {int_corr:.6f}")
        
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
