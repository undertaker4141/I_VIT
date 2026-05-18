"""
生成 RTL 測試向量
==================================================================
從 Golden Patterns 生成 RTL 模擬器可讀的測試向量

支援的格式：
1. .hex - 十六進制文件（每行一個 byte）
2. .mem - 記憶體初始化文件（地址 + 數據）
3. .txt - 文本格式（十進制）
"""

import os
import sys
import numpy as np
import argparse


def to_hex_file(data, filename, description=""):
    """
    轉換為 .hex 格式
    
    格式：每行一個 byte 的十六進制值
    例如：
    FF
    00
    7F
    """
    print(f"  生成 {filename}...")
    
    # 展平數據並轉換為 uint8
    flat_data = data.flatten()
    
    with open(filename, 'w') as f:
        # 寫入描述
        if description:
            f.write(f"// {description}\n")
            f.write(f"// Size: {len(flat_data)} bytes\n")
            f.write(f"// Shape: {data.shape}\n")
            f.write(f"// Dtype: {data.dtype}\n")
            f.write("//\n")
        
        # 寫入數據
        for value in flat_data:
            # 處理不同的數據類型
            if data.dtype in [np.int8, np.int16, np.int32, np.int64]:
                # 有號整數：轉換為無號表示
                if data.dtype == np.int8:
                    byte_val = value & 0xFF
                elif data.dtype == np.int16:
                    byte_val = value & 0xFFFF
                elif data.dtype == np.int32:
                    byte_val = value & 0xFFFFFFFF
                else:
                    byte_val = value & 0xFFFFFFFFFFFFFFFF
            else:
                byte_val = int(value)
            
            # 根據數據類型寫入不同長度
            if data.dtype in [np.int8, np.uint8]:
                f.write(f"{byte_val:02X}\n")
            elif data.dtype in [np.int16, np.uint16]:
                f.write(f"{byte_val:04X}\n")
            elif data.dtype in [np.int32, np.uint32]:
                f.write(f"{byte_val:08X}\n")
            else:
                f.write(f"{byte_val:016X}\n")


def to_mem_file(data, filename, description=""):
    """
    轉換為 .mem 格式（記憶體初始化文件）
    
    格式：@地址 數據
    例如：
    @0000 FF
    @0001 00
    @0002 7F
    """
    print(f"  生成 {filename}...")
    
    flat_data = data.flatten()
    
    with open(filename, 'w') as f:
        # 寫入描述
        if description:
            f.write(f"// {description}\n")
            f.write(f"// Size: {len(flat_data)} bytes\n")
            f.write(f"// Shape: {data.shape}\n")
            f.write(f"// Dtype: {data.dtype}\n")
            f.write("//\n")
        
        # 寫入數據
        for i, value in enumerate(flat_data):
            if data.dtype in [np.int8, np.int16, np.int32, np.int64]:
                if data.dtype == np.int8:
                    byte_val = value & 0xFF
                elif data.dtype == np.int16:
                    byte_val = value & 0xFFFF
                elif data.dtype == np.int32:
                    byte_val = value & 0xFFFFFFFF
                else:
                    byte_val = value & 0xFFFFFFFFFFFFFFFF
            else:
                byte_val = int(value)
            
            if data.dtype in [np.int8, np.uint8]:
                f.write(f"@{i:04X} {byte_val:02X}\n")
            elif data.dtype in [np.int16, np.uint16]:
                f.write(f"@{i:04X} {byte_val:04X}\n")
            elif data.dtype in [np.int32, np.uint32]:
                f.write(f"@{i:04X} {byte_val:08X}\n")
            else:
                f.write(f"@{i:04X} {byte_val:016X}\n")


def to_txt_file(data, filename, description=""):
    """
    轉換為 .txt 格式（十進制文本）
    
    格式：每行一個數值的十進制表示
    """
    print(f"  生成 {filename}...")
    
    flat_data = data.flatten()
    
    with open(filename, 'w') as f:
        # 寫入描述
        if description:
            f.write(f"# {description}\n")
            f.write(f"# Size: {len(flat_data)} values\n")
            f.write(f"# Shape: {data.shape}\n")
            f.write(f"# Dtype: {data.dtype}\n")
            f.write("#\n")
        
        # 寫入數據
        for value in flat_data:
            f.write(f"{value}\n")


def generate_rtl_vectors(golden_patterns_path, output_dir, format='hex'):
    """
    生成 RTL 測試向量
    
    Args:
        golden_patterns_path: Golden patterns .npz 文件路徑
        output_dir: 輸出目錄
        format: 輸出格式 ('hex', 'mem', 'txt', 'all')
    """
    print("=" * 80)
    print("生成 RTL 測試向量")
    print("=" * 80)
    
    # 載入 golden patterns
    print(f"\n載入 Golden Patterns: {golden_patterns_path}")
    patterns = np.load(golden_patterns_path, allow_pickle=True)
    print(f"✓ 載入了 {len(patterns.files)} 個數據項")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成關鍵層的測試向量
    print(f"\n生成測試向量 (格式: {format})...")
    
    # 1. 輸入圖片
    if 'input_image' in patterns:
        input_image = patterns['input_image']
        desc = "Input image (normalized and quantized)"
        
        if format in ['hex', 'all']:
            to_hex_file(input_image, os.path.join(output_dir, 'input_image.hex'), desc)
        if format in ['mem', 'all']:
            to_mem_file(input_image, os.path.join(output_dir, 'input_image.mem'), desc)
        if format in ['txt', 'all']:
            to_txt_file(input_image, os.path.join(output_dir, 'input_image.txt'), desc)
    
    # 2. Patch embedding 輸出
    if '1_patch_embed_output' in patterns:
        patch_embed = patterns['1_patch_embed_output']
        desc = "Patch embedding output"
        
        if format in ['hex', 'all']:
            to_hex_file(patch_embed, os.path.join(output_dir, 'patch_embed_output.hex'), desc)
        if format in ['mem', 'all']:
            to_mem_file(patch_embed, os.path.join(output_dir, 'patch_embed_output.mem'), desc)
        if format in ['txt', 'all']:
            to_txt_file(patch_embed, os.path.join(output_dir, 'patch_embed_output.txt'), desc)
    
    # 3. 第一個 Transformer block 的輸出（用於單元測試）
    if '3_block0_norm1_output' in patterns:
        block0_norm1 = patterns['3_block0_norm1_output']
        desc = "Block 0 LayerNorm1 output"
        
        if format in ['hex', 'all']:
            to_hex_file(block0_norm1, os.path.join(output_dir, 'block0_norm1_output.hex'), desc)
        if format in ['mem', 'all']:
            to_mem_file(block0_norm1, os.path.join(output_dir, 'block0_norm1_output.mem'), desc)
        if format in ['txt', 'all']:
            to_txt_file(block0_norm1, os.path.join(output_dir, 'block0_norm1_output.txt'), desc)
    
    # 4. 最終輸出
    if 'final_output' in patterns:
        final_output = patterns['final_output']
        desc = "Final output (logits)"
        
        if format in ['hex', 'all']:
            to_hex_file(final_output, os.path.join(output_dir, 'final_output.hex'), desc)
        if format in ['mem', 'all']:
            to_mem_file(final_output, os.path.join(output_dir, 'final_output.mem'), desc)
        if format in ['txt', 'all']:
            to_txt_file(final_output, os.path.join(output_dir, 'final_output.txt'), desc)
    
    # 5. 預測結果
    if 'prediction' in patterns:
        prediction = patterns['prediction'].item()
        pred_file = os.path.join(output_dir, 'prediction.txt')
        
        with open(pred_file, 'w') as f:
            f.write(f"Predicted class: {prediction['class']}\n")
            f.write(f"Probability: {prediction['probability']:.6f}\n")
            f.write(f"\nTop-5 predictions:\n")
            for i, (cls, prob) in enumerate(zip(prediction['top5_classes'], prediction['top5_probs'])):
                f.write(f"  {i+1}. Class {cls}: {prob:.6f}\n")
        
        print(f"  生成 prediction.txt...")
    
    # 生成摘要文件
    summary_file = os.path.join(output_dir, 'README.txt')
    with open(summary_file, 'w') as f:
        f.write("RTL 測試向量\n")
        f.write("=" * 80 + "\n\n")
        f.write("此目錄包含從 PyTorch 量化模型提取的 Golden Patterns，\n")
        f.write("轉換為 RTL 模擬器可讀的格式。\n\n")
        f.write("文件說明：\n")
        f.write("-" * 80 + "\n")
        f.write("input_image.*        - 輸入圖片（歸一化並量化後）\n")
        f.write("patch_embed_output.* - Patch Embedding 層輸出\n")
        f.write("block0_norm1_output.*- Block 0 LayerNorm1 輸出（單元測試用）\n")
        f.write("final_output.*       - 最終輸出 logits\n")
        f.write("prediction.txt       - 預測結果\n\n")
        f.write("格式說明：\n")
        f.write("-" * 80 + "\n")
        f.write(".hex - 十六進制格式（每行一個值）\n")
        f.write(".mem - 記憶體初始化格式（@地址 數據）\n")
        f.write(".txt - 十進制文本格式\n\n")
        f.write("使用方法：\n")
        f.write("-" * 80 + "\n")
        f.write("在 SystemVerilog testbench 中：\n")
        f.write("  $readmemh(\"input_image.hex\", input_mem);\n")
        f.write("  $readmemh(\"final_output.hex\", golden_mem);\n\n")
        f.write("在 Verilog testbench 中：\n")
        f.write("  $readmemh(\"input_image.hex\", input_mem);\n")
        f.write("  $readmemh(\"final_output.hex\", golden_mem);\n")
    
    print(f"  生成 README.txt...")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n輸出目錄: {output_dir}")
    print(f"  生成的文件可用於 RTL 模擬驗證")


def main():
    parser = argparse.ArgumentParser(description='生成 RTL 測試向量')
    parser.add_argument('--input', type=str, 
                        default='../golden_patterns/golden_patterns.npz',
                        help='Golden patterns .npz 文件路徑')
    parser.add_argument('--output', type=str, 
                        default='../rtl_vectors',
                        help='輸出目錄')
    parser.add_argument('--format', type=str, 
                        choices=['hex', 'mem', 'txt', 'all'],
                        default='all',
                        help='輸出格式')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not os.path.exists(args.input):
        print(f"✗ 找不到 Golden Patterns: {args.input}")
        print(f"  請先執行: python extract_golden_patterns.py")
        return
    
    # 生成測試向量
    generate_rtl_vectors(args.input, args.output, args.format)


if __name__ == '__main__':
    main()
