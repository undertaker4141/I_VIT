"""
驗證 C-Model 能否重現 PyTorch 的整數輸出
==================================================================
這是關鍵測試：如果 C-Model 能重現 PyTorch 的整數輸出，
那麼 RTL 實作就有了可靠的參考。
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
    print("驗證 C-Model 能否重現 PyTorch 的整數輸出")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    
    # 載入模型
    print("\n[1/4] 載入模型...")
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
    print("\n[2/4] 載入圖片...")
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
    print("\n[3/4] 追蹤到 blocks.0.norm1...")
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # 1. Input quantization
        x, input_scaling_factor = model.qact_input(image_tensor)
        print(f"qact_input 輸出: range=[{x.min().item():.4f}, {x.max().item():.4f}], scale={input_scaling_factor.item():.6f}")
        
        # 2. Patch embedding
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        print(f"patch_embed 輸出: shape={x.shape}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # 3. Add CLS token (CRITICAL: this was missing!)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print(f"+ cls_token 輸出: shape={x.shape}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # 4. Quantize position embedding
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        print(f"qact_pos 輸出: range=[{x_pos.min().item():.4f}, {x_pos.max().item():.4f}], scale={act_scaling_factor_pos.item():.6f}")
        
        # 5. Identity addition with position embedding (qact1 handles this)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        print(f"qact1 輸出 (x + pos_embed): range=[{x.min().item():.4f}, {x.max().item():.4f}], scale={x_scaling_factor.item():.6f}")
        
        print(f"\nnorm1 輸入:")
        print(f"  shape: {x.shape}")
        print(f"  range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  scaling_factor: {x_scaling_factor.item():.6f}")
        
        # 6. 計算整數輸入
        x_int = x / x_scaling_factor
        print(f"\nnorm1 整數輸入:")
        print(f"  range: [{x_int.min().item():.0f}, {x_int.max().item():.0f}]")
        
        # 7. 執行 norm1
        norm1 = model.blocks[0].norm1
        x_norm, norm_scaling_factor = norm1(x, x_scaling_factor)
        
        print(f"\nPyTorch norm1 輸出:")
        print(f"  output_integer range: [{norm1.output_integer.min().item():.0f}, {norm1.output_integer.max().item():.0f}]")
        print(f"  bias_integer range: [{norm1.bias_integer.min().item():.0f}, {norm1.bias_integer.max().item():.0f}]")
    
    # 使用 C-Model 重現
    print("\n[4/4] 使用 C-Model 重現...")
    
    # 準備輸入 - 使用 float64 保持精度
    x_int_np = x_int.cpu().numpy().astype(np.float64)
    bias_int_np = norm1.bias_integer.cpu().numpy().astype(np.float64)
    weight_np = norm1.weight.data.cpu().numpy()
    bias_np = norm1.bias.data.cpu().numpy()
    dim_sqrt = np.sqrt(x_int_np.shape[-1])
    
    print(f"C-Model 輸入:")
    print(f"  x_int range: [{x_int_np.min():.0f}, {x_int_np.max():.0f}]")
    print(f"  bias_int range: [{bias_int_np.min():.0f}, {bias_int_np.max():.0f}]")
    print(f"  dim_sqrt: {dim_sqrt:.6f}")
    
    # 執行 C-Model (使用 PyTorch 匹配算法)
    cmodel_output = pytorch_int_layer_norm(x_int_np, bias_int_np, weight_np, bias_np, dim_sqrt)
    
    print(f"\nC-Model 輸出:")
    print(f"  range: [{cmodel_output.min():.0f}, {cmodel_output.max():.0f}]")
    
    # 比較
    print("\n" + "="*80)
    print("比較結果")
    print("="*80)
    
    pytorch_output = norm1.output_integer.cpu().numpy()
    
    print(f"\nPyTorch output_integer:")
    print(f"  range: [{pytorch_output.min():.0f}, {pytorch_output.max():.0f}]")
    print(f"  mean: {pytorch_output.mean():.0f}, std: {pytorch_output.std():.0f}")
    
    print(f"\nC-Model 輸出:")
    print(f"  range: [{cmodel_output.min():.0f}, {cmodel_output.max():.0f}]")
    print(f"  mean: {cmodel_output.mean():.0f}, std: {cmodel_output.std():.0f}")
    
    # 計算差異
    diff = np.abs(pytorch_output - cmodel_output)
    print(f"\n差異統計:")
    print(f"  最大差異: {diff.max():.0f}")
    print(f"  平均差異: {diff.mean():.0f}")
    print(f"  中位數差異: {np.median(diff):.0f}")
    print(f"  差異 > 0 的比例: {(diff > 0).sum() / diff.size * 100:.2f}%")
    print(f"  差異 > 1000 的比例: {(diff > 1000).sum() / diff.size * 100:.2f}%")
    
    # 計算相關係數
    corr = np.corrcoef(pytorch_output.flatten(), cmodel_output.flatten())[0, 1]
    print(f"\n相關係數: {corr:.6f}")
    
    # 判斷結果
    print("\n" + "="*80)
    if corr > 0.99 and diff.mean() < 1000:
        print("✓ 驗證成功！C-Model 可以重現 PyTorch 的整數輸出")
        print("  這意味著 RTL 實作可以使用 C-Model 作為參考")
        success = True
    elif corr > 0.9:
        print("⚠ 部分成功：C-Model 與 PyTorch 中度相關")
        print(f"  相關係數 {corr:.4f} 還不錯，但平均差異 {diff.mean():.0f} 較大")
        print("  可能需要調整量化方式或算法細節")
        success = False
    else:
        print("✗ 驗證失敗：C-Model 無法重現 PyTorch 的整數輸出")
        print(f"  相關係數 {corr:.4f} 太低")
        print("  需要深入分析 PyTorch 的整數運算邏輯")
        success = False
    print("="*80)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
