#!/usr/bin/env python3
"""
診斷 TVM 模型問題
"""
import numpy as np
import os

print("="*80)
print("TVM 模型診斷")
print("="*80)
print()

# 1. 檢查 params.npy
print("1. 檢查 params.npy")
if os.path.exists('params.npy'):
    params = np.load('params.npy', allow_pickle=True)[()]
    print(f"   ✓ 找到 params.npy")
    print(f"   參數數量: {len(params.keys())}")
    
    # 檢查幾個關鍵權重
    key_weights = ['embed_conv_weight', 'block_0_attn_qkv_weight', 'head_weight']
    for key in key_weights:
        if key in params:
            w = params[key]
            print(f"   {key}:")
            print(f"     shape={w.shape}, dtype={w.dtype}")
            print(f"     range=[{w.min()}, {w.max()}]")
            print(f"     mean={w.mean():.4f}, std={w.std():.4f}")
        else:
            print(f"   ✗ 缺少 {key}")
else:
    print("   ✗ 找不到 params.npy")
    print("   請先執行: python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12")

print()

# 2. 檢查 calibrated_scales_gpu.npy
print("2. 檢查 calibrated scales")

# 檢查兩個可能的文件
scale_files = ['calibrated_scales_gpu.npy', 'calibrated_scales.npy']
scales = None
scale_file_found = None

for scale_file in scale_files:
    if os.path.exists(scale_file):
        scale_file_found = scale_file
        scales = np.load(scale_file, allow_pickle=True)[()]
        print(f"   ✓ 找到 {scale_file}")
        break

if scales is not None:
    print(f"   Scaling factors 數量: {len(scales.keys())}")
    
    # 顯示前 10 個
    print("   前 10 個 scaling factors:")
    for i, (k, v) in enumerate(list(scales.items())[:10]):
        print(f"     {k}: {v}")
    
    # 檢查關鍵的 scales
    key_scales = ['qact_input.act_scaling_factor', 'qact1.act_scaling_factor', 'qact2.act_scaling_factor']
    print("   關鍵 scaling factors:")
    for key in key_scales:
        if key in scales:
            print(f"     {key}: {scales[key]}")
        else:
            print(f"     ✗ 缺少 {key}")
else:
    print("   ✗ 找不到 calibrated_scales_gpu.npy 或 calibrated_scales.npy")
    print("   請先執行: python generate_calibrated_scales.py ...")

print()

# 3. 檢查 checkpoint
print("3. 檢查訓練 checkpoint")
checkpoint_paths = ['../output_gpu/checkpoint.pth', '../../output_gpu/checkpoint.pth']
checkpoint_path = None

for path in checkpoint_paths:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path:
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
    else:
        model_dict = checkpoint
    
    print(f"   ✓ 找到 checkpoint")
    print(f"   Keys 數量: {len(model_dict.keys())}")
    
    # 檢查是否有 weight_integer
    weight_integer_keys = [k for k in model_dict.keys() if 'weight_integer' in k]
    print(f"   weight_integer 數量: {len(weight_integer_keys)}")
    
    if len(weight_integer_keys) > 0:
        # 檢查第一個 weight_integer
        key = weight_integer_keys[0]
        w = model_dict[key]
        print(f"   範例 {key}:")
        print(f"     shape={w.shape}, dtype={w.dtype}")
        print(f"     range=[{w.min().item()}, {w.max().item()}]")
    else:
        print("   ✗ 沒有找到 weight_integer！")
        print("   這表示 checkpoint 不是量化後的模型")
    
    # 檢查訓練信息
    if 'train_acc' in checkpoint:
        print(f"   訓練準確率: {checkpoint['train_acc']:.2f}%")
    if 'val_acc' in checkpoint:
        print(f"   驗證準確率: {checkpoint['val_acc']:.2f}%")
else:
    print("   ✗ 找不到 checkpoint")

print()
print("="*80)
print("診斷完成")
print("="*80)
