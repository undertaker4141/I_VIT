"""
Checkpoint 轉換腳本
==================================================================
將舊版本的 checkpoint 轉換為與當前代碼兼容的格式

主要修改：
1. 將 scalar buffer (torch.Size([])) 轉換為 1D tensor (torch.Size([1]))
2. 保持其他參數不變
"""

import os
import torch
import argparse


def convert_checkpoint(input_path, output_path):
    """
    轉換 checkpoint 格式
    
    Args:
        input_path: 輸入 checkpoint 路徑
        output_path: 輸出 checkpoint 路徑
    """
    print("=" * 80)
    print("Checkpoint 轉換工具")
    print("=" * 80)
    
    # 載入舊 checkpoint
    print(f"\n載入 checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # 獲取 state_dict
    if 'model' in checkpoint:
        old_state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        old_state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        old_state_dict = checkpoint['state_dict']
    else:
        print(f"✗ 無法識別 checkpoint 格式")
        return False
    
    print(f"State dict 包含 {len(old_state_dict)} 個參數")
    
    # 轉換 state_dict
    print("\n開始轉換...")
    new_state_dict = {}
    converted_count = 0
    skipped_count = 0
    
    for key, value in old_state_dict.items():
        # 檢查是否需要轉換
        needs_conversion = False
        
        # 只轉換 buffer（scaling_factor 和 integer）
        if ('scaling_factor' in key or 'integer' in key):
            if isinstance(value, torch.Tensor):
                # 檢查是否是 scalar
                if value.dim() == 0:
                    needs_conversion = True
                # 檢查是否是 per-channel 但與當前模型不匹配
                elif value.dim() == 1 and value.size(0) > 1:
                    # LayerNorm 的 norm_scaling_factor 是 per-channel [192]
                    # 但當前模型期望 [1]，這種情況保持不變
                    # 因為這是正確的 per-channel quantization
                    pass
        
        if needs_conversion:
            # Scalar -> 1D tensor
            new_state_dict[key] = value.unsqueeze(0)
            converted_count += 1
            if converted_count <= 5:  # 只顯示前 5 個
                print(f"  ✓ {key}: {value.shape} -> {new_state_dict[key].shape}")
        else:
            new_state_dict[key] = value
            skipped_count += 1
    
    print(f"\n轉換完成:")
    print(f"  轉換: {converted_count} 個參數")
    print(f"  保持: {skipped_count} 個參數")
    
    # 保存新 checkpoint
    print(f"\n保存新 checkpoint: {output_path}")
    
    # 保持原有的其他信息
    new_checkpoint = {}
    for key in checkpoint.keys():
        if key in ['model', 'model_state_dict', 'state_dict']:
            # 使用統一的 key 'model'
            new_checkpoint['model'] = new_state_dict
        else:
            new_checkpoint[key] = checkpoint[key]
    
    torch.save(new_checkpoint, output_path)
    print(f"✓ 保存成功")
    
    # 驗證
    print(f"\n驗證新 checkpoint...")
    verify_checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
    
    if 'model' in verify_checkpoint:
        verify_state_dict = verify_checkpoint['model']
    else:
        print(f"✗ 驗證失敗：無法找到 model key")
        return False
    
    print(f"✓ 驗證成功：包含 {len(verify_state_dict)} 個參數")
    
    # 顯示一些轉換後的參數
    print(f"\n轉換後的參數示例:")
    count = 0
    for key, value in verify_state_dict.items():
        if 'scaling_factor' in key and count < 5:
            print(f"  {key}: {value.shape}")
            count += 1
    
    return True


def main():
    parser = argparse.ArgumentParser(description='轉換 checkpoint 格式')
    parser.add_argument('--input', type=str, default='output_gpu/checkpoint.pth',
                        help='輸入 checkpoint 路徑')
    parser.add_argument('--output', type=str, default='output_gpu/checkpoint_converted.pth',
                        help='輸出 checkpoint 路徑')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not os.path.exists(args.input):
        print(f"✗ 找不到輸入文件: {args.input}")
        return
    
    # 轉換
    success = convert_checkpoint(args.input, args.output)
    
    if success:
        print("\n" + "=" * 80)
        print("轉換完成！")
        print("=" * 80)
        print(f"\n新 checkpoint: {args.output}")
        print(f"\n下一步:")
        print(f"  1. 使用新 checkpoint 測試模型:")
        print(f"     python test_cmodel.py")
        print(f"  2. 或者直接使用 PyTorch 模型:")
        print(f"     python extract_golden_patterns.py")
    else:
        print("\n✗ 轉換失敗")


if __name__ == '__main__':
    main()
