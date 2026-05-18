"""
直接比較 C-Model 和 PyTorch LayerNorm 的最終輸出
==================================================================
目標：確保 C-Model 的 LayerNorm 輸出（包含 scaling）與 PyTorch 完全一致
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
from pytorch_integer_cmodel import pytorch_int_layer_norm


def main():
    """主函數"""
    print("="*80)
    print("比較 C-Model 和 PyTorch LayerNorm 的最終輸出")
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
    
    # 追蹤到 norm1
    print("\n[3/3] 追蹤到 blocks.0.norm1...")
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # Forward pass
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        print(f"\nnorm1 輸入:")
        print(f"  shape: {x.shape}")
        print(f"  range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  x_scaling_factor shape: {x_scaling_factor.shape}")
        print(f"  x_scaling_factor: {x_scaling_factor.item() if x_scaling_factor.numel() == 1 else 'per-channel'}")
        
        # PyTorch LayerNorm
        norm1 = model.blocks[0].norm1
        x_norm_pytorch, norm_scaling_factor = norm1(x, x_scaling_factor)
        
        print(f"\nPyTorch LayerNorm 輸出:")
        print(f"  shape: {x_norm_pytorch.shape}")
        print(f"  range: [{x_norm_pytorch.min().item():.4f}, {x_norm_pytorch.max().item():.4f}]")
        print(f"  norm_scaling_factor shape: {norm_scaling_factor.shape}")
        print(f"  norm_scaling_factor range: [{norm_scaling_factor.min().item():.10f}, {norm_scaling_factor.max().item():.10f}]")
        
        # C-Model LayerNorm
        print(f"\nC-Model LayerNorm:")
        x_int = (x / x_scaling_factor).cpu().numpy().astype(np.float64)
        bias_int = norm1.bias_integer.cpu().numpy().astype(np.float64)
        weight = norm1.weight.data.cpu().numpy()
        bias = norm1.bias.data.cpu().numpy()
        dim_sqrt = np.sqrt(x_int.shape[-1])
        
        x_norm_int = pytorch_int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
        
        # 轉回浮點數 - 使用正確的 per-channel scaling
        base_scaling_factor = dim_sqrt / 2 ** 30
        cmodel_scaling_factor = base_scaling_factor * weight
        x_norm_cmodel = x_norm_int * cmodel_scaling_factor
        
        print(f"  output_integer range: [{x_norm_int.min():.0f}, {x_norm_int.max():.0f}]")
        print(f"  cmodel_scaling_factor range: [{cmodel_scaling_factor.min():.10f}, {cmodel_scaling_factor.max():.10f}]")
        print(f"  final output range: [{x_norm_cmodel.min():.4f}, {x_norm_cmodel.max():.4f}]")
        
        # 比較
        print(f"\n" + "="*80)
        print(f"比較結果")
        print(f"="*80)
        
        pytorch_output = x_norm_pytorch.cpu().numpy()
        diff = np.abs(pytorch_output - x_norm_cmodel)
        
        print(f"\nPyTorch 輸出:")
        print(f"  range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
        print(f"  mean: {pytorch_output.mean():.4f}, std: {pytorch_output.std():.4f}")
        
        print(f"\nC-Model 輸出:")
        print(f"  range: [{x_norm_cmodel.min():.4f}, {x_norm_cmodel.max():.4f}]")
        print(f"  mean: {x_norm_cmodel.mean():.4f}, std: {x_norm_cmodel.std():.4f}")
        
        print(f"\n差異統計:")
        print(f"  最大差異: {diff.max():.6f}")
        print(f"  平均差異: {diff.mean():.6f}")
        print(f"  中位數差異: {np.median(diff):.6f}")
        print(f"  相對差異: {(diff / (np.abs(pytorch_output) + 1e-8)).mean():.6f}")
        
        # 計算相關係數
        corr = np.corrcoef(pytorch_output.flatten(), x_norm_cmodel.flatten())[0, 1]
        print(f"\n相關係數: {corr:.10f}")
        
        # 判斷
        print(f"\n" + "="*80)
        if corr > 0.9999 and diff.mean() < 0.001:
            print("✓ 驗證成功！C-Model LayerNorm 輸出與 PyTorch 幾乎完全一致")
            print("  可以安全地在端到端推論中使用")
            return True
        elif corr > 0.999:
            print("⚠ 部分成功：相關係數很高但存在小誤差")
            print(f"  相關係數 {corr:.6f} 很好")
            print(f"  但平均差異 {diff.mean():.6f} 可能影響最終預測")
            return False
        else:
            print("✗ 驗證失敗：輸出差異太大")
            print(f"  相關係數 {corr:.6f} 太低")
            return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
