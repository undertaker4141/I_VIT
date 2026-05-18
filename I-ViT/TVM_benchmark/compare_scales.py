#!/usr/bin/env python3
"""
比較兩個 calibrated scales 文件
"""
import numpy as np
import os

print("="*80)
print("比較 Calibrated Scales")
print("="*80)
print()

# 檢查兩個文件
files = {
    'calibrated_scales.npy': None,
    'calibrated_scales_gpu.npy': None
}

for filename in files.keys():
    if os.path.exists(filename):
        files[filename] = np.load(filename, allow_pickle=True)[()]
        print(f"✓ 找到 {filename}")
        print(f"  Scaling factors 數量: {len(files[filename])}")
    else:
        print(f"✗ 找不到 {filename}")

print()

# 如果兩個文件都存在，比較它們
if files['calibrated_scales.npy'] is not None and files['calibrated_scales_gpu.npy'] is not None:
    scales1 = files['calibrated_scales.npy']
    scales2 = files['calibrated_scales_gpu.npy']
    
    print("="*80)
    print("比較結果")
    print("="*80)
    print()
    
    # 檢查 keys 是否相同
    keys1 = set(scales1.keys())
    keys2 = set(scales2.keys())
    
    if keys1 == keys2:
        print("✓ 兩個文件的 keys 完全相同")
    else:
        print("✗ 兩個文件的 keys 不同")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  只在 calibrated_scales.npy 中: {only_in_1}")
        if only_in_2:
            print(f"  只在 calibrated_scales_gpu.npy 中: {only_in_2}")
    
    print()
    
    # 比較關鍵的 scaling factors
    key_scales = [
        'qact_input.act_scaling_factor',
        'qact1.act_scaling_factor', 
        'qact2.act_scaling_factor',
        'blocks.0.qact1.act_scaling_factor',
        'blocks.11.qact4.act_scaling_factor'
    ]
    
    print("關鍵 Scaling Factors 比較:")
    print()
    print(f"{'Key':<50} {'calibrated_scales.npy':<25} {'calibrated_scales_gpu.npy':<25} {'差異%':<10}")
    print("-"*110)
    
    max_diff = 0
    max_diff_key = None
    
    for key in key_scales:
        if key in scales1 and key in scales2:
            val1 = float(scales1[key])
            val2 = float(scales2[key])
            diff_pct = abs(val1 - val2) / val1 * 100 if val1 != 0 else 0
            
            if diff_pct > max_diff:
                max_diff = diff_pct
                max_diff_key = key
            
            status = "✓" if diff_pct < 5 else "✗"
            print(f"{status} {key:<48} {val1:<25.10f} {val2:<25.10f} {diff_pct:<10.2f}")
    
    print()
    
    # 統計所有差異
    all_diffs = []
    for key in keys1 & keys2:
        val1 = float(scales1[key])
        val2 = float(scales2[key])
        diff_pct = abs(val1 - val2) / val1 * 100 if val1 != 0 else 0
        all_diffs.append(diff_pct)
    
    print("="*80)
    print("統計")
    print("="*80)
    print(f"平均差異: {np.mean(all_diffs):.2f}%")
    print(f"最大差異: {max_diff:.2f}% (key: {max_diff_key})")
    print(f"差異 > 5% 的數量: {sum(1 for d in all_diffs if d > 5)}/{len(all_diffs)}")
    print()
    
    if max_diff > 10:
        print("⚠️  警告: 兩個文件差異很大 (>10%)！")
        print("   這可能表示它們是用不同的 checkpoint 生成的。")
        print()
        print("   calibrated_scales.npy 可能是用原始 QAT checkpoint 生成")
        print("   calibrated_scales_gpu.npy 應該是用 output_gpu/checkpoint.pth 生成")
    elif max_diff > 5:
        print("⚠️  注意: 兩個文件有一些差異 (>5%)")
    else:
        print("✓ 兩個文件基本相同 (差異 <5%)")

elif files['calibrated_scales_gpu.npy'] is not None:
    print("只有 calibrated_scales_gpu.npy 存在")
    print("這是正確的！應該使用 GPU 訓練後的 checkpoint 生成的 scales")
elif files['calibrated_scales.npy'] is not None:
    print("只有 calibrated_scales.npy 存在")
    print()
    print("⚠️  警告: 應該生成 calibrated_scales_gpu.npy")
    print("   使用以下命令:")
    print("   python generate_calibrated_scales.py \\")
    print("       --model-path '../output_gpu/checkpoint.pth' \\")
    print("       --output calibrated_scales_gpu.npy \\")
    print("       --image '../../data/test_image.JPEG'")
else:
    print("兩個文件都不存在！")
    print("請先生成 calibrated scales")

print()
print("="*80)
