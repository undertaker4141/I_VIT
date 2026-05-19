"""
詳細測試 MLP
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
from pure_numpy_cmodel import (
    int_dense,
    int_gelu,
    requantize,
    quantize_to_int,
)


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
    print("詳細測試 MLP")
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
    
    # 執行到 Norm2 + QuantAct3
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        # 保存輸入
        x_input = x.clone()
        x_input_sf = x_scaling_factor
        
        block = model.blocks[0]
        
        # Norm1
        x_norm1, norm1_sf = block.norm1(x, x_scaling_factor)
        
        # QuantAct1
        x_qact1, qact1_sf = block.qact1(x_norm1, norm1_sf)
        
        # Attention
        x_attn, x_attn_sf = block.attn(x_qact1, qact1_sf)
        
        # Residual 1
        x_res1, x_res1_sf = block.qact2(x_input, x_input_sf, x_attn, x_attn_sf)
        
        # Norm2
        x_norm2, norm2_sf = block.norm2(x_res1, x_res1_sf)
        
        # QuantAct3
        x_qact3, qact3_sf = block.qact3(x_norm2, norm2_sf)
        
        print(f"\n[PyTorch] MLP 輸入:")
        print(f"  range=[{x_qact3.min():.6f}, {x_qact3.max():.6f}], sf={extract_scaling_factor(qact3_sf):.9f}")
        
        # FC1
        x_fc1, fc1_sf = block.mlp.fc1(x_qact3, qact3_sf)
        x_fc1_quant, fc1_quant_sf = block.mlp.qact1(x_fc1, fc1_sf)
        
        print(f"\n[PyTorch] FC1 輸出:")
        print(f"  range=[{x_fc1_quant.min():.6f}, {x_fc1_quant.max():.6f}], sf={extract_scaling_factor(fc1_quant_sf):.9f}")
        
        # GELU
        x_gelu, gelu_sf = block.mlp.act(x_fc1_quant, fc1_quant_sf)
        x_gelu_quant, gelu_quant_sf = block.mlp.qact_gelu(x_gelu, gelu_sf)
        
        print(f"\n[PyTorch] GELU 輸出:")
        print(f"  range=[{x_gelu_quant.min():.6f}, {x_gelu_quant.max():.6f}], sf={extract_scaling_factor(gelu_quant_sf):.9f}")
        
        # FC2
        x_fc2, fc2_sf = block.mlp.fc2(x_gelu_quant, gelu_quant_sf)
        x_fc2_quant, fc2_quant_sf = block.mlp.qact2(x_fc2, fc2_sf)
        
        print(f"\n[PyTorch] FC2 輸出 (MLP 輸出):")
        print(f"  range=[{x_fc2_quant.min():.6f}, {x_fc2_quant.max():.6f}], sf={extract_scaling_factor(fc2_quant_sf):.9f}")
        
        # 提取整數值
        x_qact3_np = x_qact3.cpu().numpy()
        x_qact3_sf_np = extract_scaling_factor(qact3_sf)
        x_qact3_int8 = quantize_to_int(x_qact3_np, x_qact3_sf_np, bits=8)
        
        print(f"\n" + "="*80)
        print("[C-Model] MLP")
        print("="*80)
        
        # 提取權重
        fc1_weight_int = block.mlp.fc1.weight_integer.cpu().numpy().astype(np.int8)
        fc1_bias_int = block.mlp.fc1.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc1.bias is not None else None
        fc1_fc_sf = extract_scaling_factor(block.mlp.fc1.fc_scaling_factor)
        
        fc2_weight_int = block.mlp.fc2.weight_integer.cpu().numpy().astype(np.int8)
        fc2_bias_int = block.mlp.fc2.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc2.bias is not None else None
        fc2_fc_sf = extract_scaling_factor(block.mlp.fc2.fc_scaling_factor)
        
        # 提取 QuantAct SF
        mlp_qact1_sf = extract_scaling_factor(block.mlp.qact1.act_scaling_factor)
        mlp_qact_gelu_sf = extract_scaling_factor(block.mlp.qact_gelu.act_scaling_factor)
        mlp_qact2_sf = extract_scaling_factor(block.mlp.qact2.act_scaling_factor)
        
        # FC1
        fc1_int32 = int_dense(x_qact3_int8, fc1_weight_int, fc1_bias_int)
        fc1_sf_cmodel = x_qact3_sf_np * fc1_fc_sf
        fc1_int8 = requantize(fc1_int32, fc1_sf_cmodel, mlp_qact1_sf, output_bits=8)
        
        # 比較 FC1
        fc1_float_cmodel = fc1_int8.astype(np.float32) * mlp_qact1_sf
        success1 = compare_values("FC1 輸出", x_fc1_quant, fc1_float_cmodel, 
                                  extract_scaling_factor(fc1_quant_sf), mlp_qact1_sf)
        
        if not success1:
            print("\n✗ FC1 失敗！")
            return False
        
        # GELU
        gelu_int32 = int_gelu(fc1_int8.astype(np.int32), mlp_qact1_sf, output_bit=8, n=23)
        
        # PyTorch IntGELU 的輸出 SF = input_sf * (1 / 2^(output_bit-1))
        sigmoid_sf = 1.0 / (2 ** (8 - 1))
        gelu_output_sf = mlp_qact1_sf * sigmoid_sf
        
        gelu_int8 = requantize(gelu_int32, gelu_output_sf, mlp_qact_gelu_sf, output_bits=8)
        
        # 比較 GELU
        gelu_float_cmodel = gelu_int8.astype(np.float32) * mlp_qact_gelu_sf
        success2 = compare_values("GELU 輸出", x_gelu_quant, gelu_float_cmodel,
                                  extract_scaling_factor(gelu_quant_sf), mlp_qact_gelu_sf)
        
        if not success2:
            print("\n✗ GELU 失敗！")
            return False
        
        # FC2
        fc2_int32 = int_dense(gelu_int8, fc2_weight_int, fc2_bias_int)
        fc2_sf_cmodel = mlp_qact_gelu_sf * fc2_fc_sf
        fc2_int16 = requantize(fc2_int32, fc2_sf_cmodel, mlp_qact2_sf, output_bits=16)
        
        # 比較 FC2
        fc2_float_cmodel = fc2_int16.astype(np.float32) * mlp_qact2_sf
        success3 = compare_values("FC2 輸出 (MLP 輸出)", x_fc2_quant, fc2_float_cmodel,
                                  extract_scaling_factor(fc2_quant_sf), mlp_qact2_sf)
        
        if not success3:
            print("\n✗ FC2 失敗！")
            return False
        
        print("\n" + "="*80)
        print("✓ MLP 所有步驟測試通過！")
        print("="*80)
        
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
