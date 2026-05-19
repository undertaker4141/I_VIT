"""
測試 Block 0 的 Residual 連接
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
from pure_numpy_cmodel import quant_act_residual, quantize_to_int


def extract_scaling_factor(sf):
    """提取 scaling factor"""
    if isinstance(sf, torch.Tensor):
        if sf.numel() == 1:
            return sf.item()
        else:
            return sf.cpu().numpy()
    else:
        return sf


def compare_values(name, pytorch_val, cmodel_val, pytorch_sf, cmodel_sf):
    """比較浮點值"""
    pytorch_np = pytorch_val.cpu().numpy() if isinstance(pytorch_val, torch.Tensor) else pytorch_val
    
    corr = np.corrcoef(pytorch_np.flatten(), cmodel_val.flatten())[0, 1]
    diff = np.abs(pytorch_np - cmodel_val)
    
    print(f"\n{name}:")
    print(f"  PyTorch: range=[{pytorch_np.min():.6f}, {pytorch_np.max():.6f}], sf={pytorch_sf:.9f}")
    print(f"  C-Model: range=[{cmodel_val.min():.6f}, {cmodel_val.max():.6f}], sf={cmodel_sf:.9f}")
    print(f"  相關係數: {corr:.6f}")
    print(f"  最大差異: {diff.max():.6f}")
    print(f"  平均差異: {diff.mean():.6f}")
    
    if corr > 0.99:
        print(f"  ✓ 通過")
        return True
    else:
        print(f"  ✗ 失敗")
        return False


def main():
    print("="*80)
    print("測試 Block 0 的 Residual 連接")
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
    
    # 執行到 Block 0
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        print(f"\n[1] Block 0 輸入:")
        print(f"  range=[{x.min():.6f}, {x.max():.6f}], sf={extract_scaling_factor(x_scaling_factor):.9f}")
        
        # 保存輸入
        x_input = x.clone()
        x_input_sf = x_scaling_factor
        
        block = model.blocks[0]
        
        # Norm1 + QuantAct1
        x_norm1, norm1_sf = block.norm1(x, x_scaling_factor)
        x_qact1, qact1_sf = block.qact1(x_norm1, norm1_sf)
        
        # Attention
        x_attn, x_attn_sf = block.attn(x_qact1, qact1_sf)
        
        print(f"\n[2] Attention 輸出:")
        print(f"  range=[{x_attn.min():.6f}, {x_attn.max():.6f}], sf={extract_scaling_factor(x_attn_sf):.9f}")
        
        # Residual 1
        x_res1, x_res1_sf = block.qact2(x_input, x_input_sf, x_attn, x_attn_sf)
        
        print(f"\n[3] Residual 1 輸出:")
        print(f"  range=[{x_res1.min():.6f}, {x_res1.max():.6f}], sf={extract_scaling_factor(x_res1_sf):.9f}")
        
        # 保存用於 Residual 2
        x_input2 = x_res1.clone()
        x_input2_sf = x_res1_sf
        
        # Norm2 + QuantAct3
        x_norm2, norm2_sf = block.norm2(x_res1, x_res1_sf)
        x_qact3, qact3_sf = block.qact3(x_norm2, norm2_sf)
        
        # MLP
        x_mlp, x_mlp_sf = block.mlp(x_qact3, qact3_sf)
        
        print(f"\n[4] MLP 輸出:")
        print(f"  range=[{x_mlp.min():.6f}, {x_mlp.max():.6f}], sf={extract_scaling_factor(x_mlp_sf):.9f}")
        
        # Residual 2
        x_output, x_output_sf = block.qact4(x_input2, x_input2_sf, x_mlp, x_mlp_sf)
        
        print(f"\n[5] Residual 2 輸出 (Block 0 輸出):")
        print(f"  range=[{x_output.min():.6f}, {x_output.max():.6f}], sf={extract_scaling_factor(x_output_sf):.9f}")
        
        # 測試 C-Model Residual 1
        print(f"\n" + "="*80)
        print("[C-Model] Residual 1")
        print("="*80)
        
        # 轉換為整數
        x_input_np = x_input.cpu().numpy()
        x_input_sf_np = extract_scaling_factor(x_input_sf)
        x_input_int16 = quantize_to_int(x_input_np, x_input_sf_np, bits=16)
        
        x_attn_np = x_attn.cpu().numpy()
        x_attn_sf_np = extract_scaling_factor(x_attn_sf)
        x_attn_int16 = quantize_to_int(x_attn_np, x_attn_sf_np, bits=16)
        
        qact2_sf = extract_scaling_factor(block.qact2.act_scaling_factor)
        
        x_res1_int16 = quant_act_residual(
            x_attn_int16, x_attn_sf_np,
            x_input_int16, x_input_sf_np,
            qact2_sf, output_bits=16
        )
        
        x_res1_float_cmodel = x_res1_int16.astype(np.float32) * qact2_sf
        
        success1 = compare_values("Residual 1", x_res1, x_res1_float_cmodel,
                                  extract_scaling_factor(x_res1_sf), qact2_sf)
        
        if not success1:
            print("\n✗ Residual 1 失敗！")
            return False
        
        # 測試 C-Model Residual 2
        print(f"\n" + "="*80)
        print("[C-Model] Residual 2")
        print("="*80)
        
        x_input2_np = x_input2.cpu().numpy()
        x_input2_sf_np = extract_scaling_factor(x_input2_sf)
        x_input2_int16 = quantize_to_int(x_input2_np, x_input2_sf_np, bits=16)
        
        x_mlp_np = x_mlp.cpu().numpy()
        x_mlp_sf_np = extract_scaling_factor(x_mlp_sf)
        x_mlp_int16 = quantize_to_int(x_mlp_np, x_mlp_sf_np, bits=16)
        
        qact4_sf = extract_scaling_factor(block.qact4.act_scaling_factor)
        
        x_output_int16 = quant_act_residual(
            x_mlp_int16, x_mlp_sf_np,
            x_input2_int16, x_input2_sf_np,
            qact4_sf, output_bits=16
        )
        
        x_output_float_cmodel = x_output_int16.astype(np.float32) * qact4_sf
        
        success2 = compare_values("Residual 2 (Block 0 輸出)", x_output, x_output_float_cmodel,
                                  extract_scaling_factor(x_output_sf), qact4_sf)
        
        if not success2:
            print("\n✗ Residual 2 失敗！")
            return False
        
        print("\n" + "="*80)
        print("✓ 所有 Residual 測試通過！")
        print("="*80)
        
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
