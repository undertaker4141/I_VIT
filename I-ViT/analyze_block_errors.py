"""
分析 Block 誤差模式
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


def analyze_block_components(model, block_idx, x_input, x_input_sf, weights):
    """分析單個 block 的各個組件"""
    block = model.blocks[block_idx]
    
    with torch.no_grad():
        # 保存輸入
        x_input_save = x_input.clone()
        x_input_sf_save = x_input_sf
        
        # Norm1 + QuantAct1
        x_norm1, norm1_sf = block.norm1(x_input, x_input_sf)
        x_qact1, qact1_sf = block.qact1(x_norm1, norm1_sf)
        
        # Attention
        x_attn, x_attn_sf = block.attn(x_qact1, qact1_sf)
        
        # Residual 1
        x_res1, x_res1_sf = block.qact2(x_input_save, x_input_sf_save, x_attn, x_attn_sf)
        
        # 保存用於 Residual 2
        x_input2_save = x_res1.clone()
        x_input2_sf_save = x_res1_sf
        
        # Norm2 + QuantAct3
        x_norm2, norm2_sf = block.norm2(x_res1, x_res1_sf)
        x_qact3, qact3_sf = block.qact3(x_norm2, norm2_sf)
        
        # MLP
        x_mlp, x_mlp_sf = block.mlp(x_qact3, qact3_sf)
        
        # Residual 2
        x_output, x_output_sf = block.qact4(x_input2_save, x_input2_sf_save, x_mlp, x_mlp_sf)
    
    # 提取 scaling factors
    qact1_sf_val = extract_scaling_factor(qact1_sf)
    qact2_sf_val = extract_scaling_factor(x_res1_sf)
    qact3_sf_val = extract_scaling_factor(qact3_sf)
    qact4_sf_val = extract_scaling_factor(x_output_sf)
    
    attn_qact1_sf = extract_scaling_factor(block.attn.qact1.act_scaling_factor)
    attn_qact2_sf = extract_scaling_factor(block.attn.qact2.act_scaling_factor)
    attn_qact3_sf = extract_scaling_factor(block.attn.qact3.act_scaling_factor)
    
    mlp_qact1_sf = extract_scaling_factor(block.mlp.qact1.act_scaling_factor)
    mlp_qact_gelu_sf = extract_scaling_factor(block.mlp.qact_gelu.act_scaling_factor)
    mlp_qact2_sf = extract_scaling_factor(block.mlp.qact2.act_scaling_factor)
    
    return {
        'block_idx': block_idx,
        'qact1_sf': qact1_sf_val,
        'qact2_sf': qact2_sf_val,
        'qact3_sf': qact3_sf_val,
        'qact4_sf': qact4_sf_val,
        'attn_qact1_sf': attn_qact1_sf,
        'attn_qact2_sf': attn_qact2_sf,
        'attn_qact3_sf': attn_qact3_sf,
        'mlp_qact1_sf': mlp_qact1_sf,
        'mlp_qact_gelu_sf': mlp_qact_gelu_sf,
        'mlp_qact2_sf': mlp_qact2_sf,
        'attn_output_range': (x_attn.min().item(), x_attn.max().item()),
        'mlp_output_range': (x_mlp.min().item(), x_mlp.max().item()),
        'block_output_range': (x_output.min().item(), x_output.max().item()),
    }


def main():
    print("="*80)
    print("分析 Block 誤差模式")
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
    
    # 準備輸入
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
    
    # 分析所有 blocks
    print("\n分析各 Block 的 Scaling Factors:")
    print("="*80)
    
    analyses = []
    x_current = x
    x_current_sf = x_scaling_factor
    
    # 從測試結果中已知的相關係數
    correlations = [0.987, 0.944, 0.875, 0.917, 0.841, 0.972, 0.995, 0.983, 0.974, 0.994, 0.983, 0.978]
    
    for block_idx in range(12):
        analysis = analyze_block_components(model, block_idx, x_current, x_current_sf, weights)
        analysis['correlation'] = correlations[block_idx]
        analyses.append(analysis)
        
        # 更新輸入
        with torch.no_grad():
            x_current, x_current_sf = model.blocks[block_idx](x_current, x_current_sf)
    
    # 打印分析結果
    print(f"\n{'Block':<6} {'Corr':<7} {'QAct1 SF':<12} {'QAct2 SF':<12} {'QAct3 SF':<12} {'QAct4 SF':<12}")
    print("-" * 80)
    
    for a in analyses:
        status = "✓" if a['correlation'] > 0.98 else "✗"
        print(f"{a['block_idx']:<6} {a['correlation']:.4f} {status} "
              f"{a['qact1_sf']:.6e}  {a['qact2_sf']:.6e}  "
              f"{a['qact3_sf']:.6e}  {a['qact4_sf']:.6e}")
    
    # 分析 SF 的變化
    print("\n" + "="*80)
    print("Scaling Factor 統計:")
    print("="*80)
    
    qact1_sfs = [a['qact1_sf'] for a in analyses]
    qact2_sfs = [a['qact2_sf'] for a in analyses]
    qact3_sfs = [a['qact3_sf'] for a in analyses]
    qact4_sfs = [a['qact4_sf'] for a in analyses]
    
    print(f"\nQAct1 SF (Norm1 後):")
    print(f"  範圍: [{min(qact1_sfs):.6e}, {max(qact1_sfs):.6e}]")
    print(f"  平均: {np.mean(qact1_sfs):.6e}")
    print(f"  變異係數: {np.std(qact1_sfs)/np.mean(qact1_sfs):.4f}")
    
    print(f"\nQAct2 SF (Residual 1 後):")
    print(f"  範圍: [{min(qact2_sfs):.6e}, {max(qact2_sfs):.6e}]")
    print(f"  平均: {np.mean(qact2_sfs):.6e}")
    print(f"  變異係數: {np.std(qact2_sfs)/np.mean(qact2_sfs):.4f}")
    
    print(f"\nQAct3 SF (Norm2 後):")
    print(f"  範圍: [{min(qact3_sfs):.6e}, {max(qact3_sfs):.6e}]")
    print(f"  平均: {np.mean(qact3_sfs):.6e}")
    print(f"  變異係數: {np.std(qact3_sfs)/np.mean(qact3_sfs):.4f}")
    
    print(f"\nQAct4 SF (Residual 2 後 / Block 輸出):")
    print(f"  範圍: [{min(qact4_sfs):.6e}, {max(qact4_sfs):.6e}]")
    print(f"  平均: {np.mean(qact4_sfs):.6e}")
    print(f"  變異係數: {np.std(qact4_sfs)/np.mean(qact4_sfs):.4f}")
    
    # 尋找相關性
    print("\n" + "="*80)
    print("相關性分析:")
    print("="*80)
    
    # 檢查 SF 變化與相關係數的關係
    qact4_sf_changes = [abs(qact4_sfs[i] - qact4_sfs[i-1])/qact4_sfs[i-1] if i > 0 else 0 for i in range(12)]
    
    print(f"\n{'Block':<6} {'Corr':<7} {'QAct4 SF':<12} {'SF 變化%':<12} {'輸出範圍':<30}")
    print("-" * 80)
    
    for i, a in enumerate(analyses):
        status = "✓" if a['correlation'] > 0.98 else "✗"
        output_range = f"[{a['block_output_range'][0]:.2f}, {a['block_output_range'][1]:.2f}]"
        print(f"{a['block_idx']:<6} {a['correlation']:.4f} {status} "
              f"{a['qact4_sf']:.6e}  {qact4_sf_changes[i]*100:>10.2f}%  {output_range:<30}")
    
    # 找出問題 blocks 的共同特徵
    print("\n" + "="*80)
    print("問題 Blocks 分析:")
    print("="*80)
    
    problem_blocks = [a for a in analyses if a['correlation'] < 0.98]
    good_blocks = [a for a in analyses if a['correlation'] >= 0.98]
    
    if problem_blocks:
        print(f"\n問題 Blocks: {[a['block_idx'] for a in problem_blocks]}")
        print(f"好的 Blocks: {[a['block_idx'] for a in good_blocks]}")
        
        # 比較平均 SF
        problem_avg_qact4 = np.mean([a['qact4_sf'] for a in problem_blocks])
        good_avg_qact4 = np.mean([a['qact4_sf'] for a in good_blocks])
        
        print(f"\n平均 QAct4 SF:")
        print(f"  問題 Blocks: {problem_avg_qact4:.6e}")
        print(f"  好的 Blocks: {good_avg_qact4:.6e}")
        print(f"  差異: {abs(problem_avg_qact4 - good_avg_qact4)/good_avg_qact4*100:.2f}%")


if __name__ == '__main__':
    main()
