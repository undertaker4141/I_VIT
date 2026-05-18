#!/usr/bin/env python3
"""
檢查 params.npy 是從哪個 checkpoint 生成的
"""
import numpy as np
import torch
import os

print("="*80)
print("檢查 params.npy 來源")
print("="*80)
print()

# 1. 加載 params.npy
if not os.path.exists('params.npy'):
    print("✗ 找不到 params.npy")
    print("請執行: python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12")
    exit(1)

params = np.load('params.npy', allow_pickle=True)[()]
print(f"✓ 加載 params.npy")
print(f"  參數數量: {len(params)}")
print()

# 2. 加載兩個 checkpoint
checkpoints = {
    'GPU 訓練': '../output_gpu/checkpoint.pth',
    '原始 QAT': '../checkpoints/qat_calibrated.pth'
}

checkpoint_weights = {}

for name, path in checkpoints.items():
    if os.path.exists(path):
        print(f"✓ 找到 {name} checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model_dict = ckpt['model'] if 'model' in ckpt else ckpt
        checkpoint_weights[name] = model_dict
    else:
        print(f"✗ 找不到 {name} checkpoint: {path}")

print()

if len(checkpoint_weights) == 0:
    print("✗ 沒有找到任何 checkpoint")
    exit(1)

# 3. 比較 params.npy 與 checkpoint 的權重
print("="*80)
print("比較權重")
print("="*80)
print()

# 檢查幾個關鍵權重
test_keys = [
    ('embed_conv_weight', 'patch_embed.proj.weight_integer'),
    ('block_0_attn_qkv_weight', 'blocks.0.attn.qkv.weight_integer'),
    ('head_weight', 'head.weight_integer')
]

for params_key, ckpt_key in test_keys:
    if params_key not in params:
        print(f"✗ params.npy 中缺少 {params_key}")
        continue
    
    params_weight = params[params_key]
    print(f"檢查 {params_key}:")
    print(f"  params.npy: shape={params_weight.shape}, dtype={params_weight.dtype}")
    print(f"  範圍: [{params_weight.min()}, {params_weight.max()}]")
    print(f"  mean={params_weight.mean():.4f}, std={params_weight.std():.4f}")
    print()
    
    for name, model_dict in checkpoint_weights.items():
        if ckpt_key in model_dict:
            ckpt_weight = model_dict[ckpt_key].cpu().numpy().astype('int8')
            
            # 比較
            if params_weight.shape == ckpt_weight.shape:
                diff = np.abs(params_weight - ckpt_weight).sum()
                total = params_weight.size
                match_rate = (params_weight == ckpt_weight).sum() / total * 100
                
                print(f"  vs {name}:")
                print(f"    範圍: [{ckpt_weight.min()}, {ckpt_weight.max()}]")
                print(f"    mean={ckpt_weight.mean():.4f}, std={ckpt_weight.std():.4f}")
                print(f"    匹配率: {match_rate:.2f}%")
                print(f"    差異總和: {diff}")
                
                if match_rate > 99:
                    print(f"    ✓ 幾乎完全匹配！")
                elif match_rate > 90:
                    print(f"    ⚠️  大部分匹配")
                else:
                    print(f"    ✗ 差異很大")
            else:
                print(f"  vs {name}: shape 不匹配")
        print()

print()
print("="*80)
print("結論")
print("="*80)
print()

# 找出最匹配的 checkpoint
best_match = None
best_match_rate = 0

for name, model_dict in checkpoint_weights.items():
    total_match = 0
    total_elements = 0
    
    for params_key, ckpt_key in test_keys:
        if params_key in params and ckpt_key in model_dict:
            params_weight = params[params_key]
            ckpt_weight = model_dict[ckpt_key].cpu().numpy().astype('int8')
            
            if params_weight.shape == ckpt_weight.shape:
                total_match += (params_weight == ckpt_weight).sum()
                total_elements += params_weight.size
    
    if total_elements > 0:
        match_rate = total_match / total_elements * 100
        print(f"{name}: 總體匹配率 {match_rate:.2f}%")
        
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_match = name

print()
if best_match:
    if best_match_rate > 99:
        print(f"✓ params.npy 是從 {best_match} checkpoint 生成的")
    else:
        print(f"⚠️  params.npy 可能是從 {best_match} checkpoint 生成的（匹配率 {best_match_rate:.2f}%）")
    
    if best_match == '原始 QAT' and 'GPU 訓練' in checkpoint_weights:
        print()
        print("❌ 問題找到了！")
        print("   params.npy 是用原始 QAT checkpoint 生成的，")
        print("   但應該用 GPU 訓練後的 checkpoint！")
        print()
        print("   解決方案：重新生成 params.npy")
        print("   python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12")

print("="*80)
