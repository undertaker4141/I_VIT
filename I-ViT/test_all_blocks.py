"""
測試所有 12 個 Transformer Blocks
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
from pure_integer_end_to_end import extract_weights_and_scales, pure_integer_transformer_block
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


def test_block(model, block_idx, x_input, x_input_sf, weights):
    """測試單個 block"""
    block = model.blocks[block_idx]
    
    # PyTorch 推論
    with torch.no_grad():
        x_output_pytorch, x_output_sf_pytorch = block(x_input, x_input_sf)
    
    # C-Model 推論
    x_input_np = x_input.cpu().numpy()
    x_input_sf_np = extract_scaling_factor(x_input_sf)
    x_input_int16 = quantize_to_int(x_input_np, x_input_sf_np, bits=16)
    
    dim_sqrt = np.sqrt(x_input_int16.shape[-1])
    block_weights = weights[f'block_{block_idx}']
    
    x_output_int16, x_output_sf_cmodel = pure_integer_transformer_block(
        x_input_int16, x_input_sf_np, block_weights, dim_sqrt, block_idx
    )
    
    x_output_cmodel = x_output_int16.astype(np.float32) * x_output_sf_cmodel
    
    # 比較
    pytorch_np = x_output_pytorch.cpu().numpy()
    corr = np.corrcoef(pytorch_np.flatten(), x_output_cmodel.flatten())[0, 1]
    diff = np.abs(pytorch_np - x_output_cmodel)
    
    return {
        'block_idx': block_idx,
        'correlation': corr,
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'pytorch_range': (pytorch_np.min(), pytorch_np.max()),
        'cmodel_range': (x_output_cmodel.min(), x_output_cmodel.max()),
        'pytorch_sf': extract_scaling_factor(x_output_sf_pytorch),
        'cmodel_sf': x_output_sf_cmodel,
        'x_output_pytorch': x_output_pytorch,
        'x_output_sf_pytorch': x_output_sf_pytorch,
    }


def main():
    print("="*80)
    print("測試所有 12 個 Transformer Blocks")
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
    print("\n提取權重和 scaling factors...")
    weights = extract_weights_and_scales(model, image_tensor)
    print("✓ 權重已提取")
    
    # 準備輸入
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
    
    print(f"\n初始輸入: range=[{x.min():.6f}, {x.max():.6f}], sf={extract_scaling_factor(x_scaling_factor):.9f}")
    
    # 測試所有 blocks
    print("\n" + "="*80)
    print("測試結果")
    print("="*80)
    
    results = []
    x_current = x
    x_current_sf = x_scaling_factor
    
    for block_idx in range(12):
        result = test_block(model, block_idx, x_current, x_current_sf, weights)
        results.append(result)
        
        # 更新輸入為當前 block 的輸出（使用 PyTorch 的輸出）
        x_current = result['x_output_pytorch']
        x_current_sf = result['x_output_sf_pytorch']
        
        # 打印結果
        status = "✓" if result['correlation'] > 0.98 else "✗"
        print(f"\nBlock {block_idx}: {status}")
        print(f"  相關係數: {result['correlation']:.6f}")
        print(f"  最大差異: {result['max_diff']:.6f}")
        print(f"  平均差異: {result['mean_diff']:.6f}")
        print(f"  PyTorch range: [{result['pytorch_range'][0]:.4f}, {result['pytorch_range'][1]:.4f}]")
        print(f"  C-Model range: [{result['cmodel_range'][0]:.4f}, {result['cmodel_range'][1]:.4f}]")
    
    # 統計分析
    print("\n" + "="*80)
    print("統計分析")
    print("="*80)
    
    correlations = [r['correlation'] for r in results]
    max_diffs = [r['max_diff'] for r in results]
    mean_diffs = [r['mean_diff'] for r in results]
    
    print(f"\n相關係數:")
    print(f"  最小值: {min(correlations):.6f} (Block {correlations.index(min(correlations))})")
    print(f"  最大值: {max(correlations):.6f} (Block {correlations.index(max(correlations))})")
    print(f"  平均值: {np.mean(correlations):.6f}")
    print(f"  標準差: {np.std(correlations):.6f}")
    
    print(f"\n最大差異:")
    print(f"  最小值: {min(max_diffs):.6f} (Block {max_diffs.index(min(max_diffs))})")
    print(f"  最大值: {max(max_diffs):.6f} (Block {max_diffs.index(max(max_diffs))})")
    print(f"  平均值: {np.mean(max_diffs):.6f}")
    
    print(f"\n平均差異:")
    print(f"  最小值: {min(mean_diffs):.6f} (Block {mean_diffs.index(min(mean_diffs))})")
    print(f"  最大值: {max(mean_diffs):.6f} (Block {mean_diffs.index(max(mean_diffs))})")
    print(f"  平均值: {np.mean(mean_diffs):.6f}")
    
    # 通過率
    passed = sum(1 for r in results if r['correlation'] > 0.98)
    print(f"\n通過率 (相關係數 > 0.98): {passed}/12 ({passed/12*100:.1f}%)")
    
    # 找出問題 blocks
    problem_blocks = [r['block_idx'] for r in results if r['correlation'] < 0.98]
    if problem_blocks:
        print(f"\n需要優化的 blocks: {problem_blocks}")
    else:
        print(f"\n✓ 所有 blocks 都通過測試！")
    
    return results


if __name__ == '__main__':
    results = main()
