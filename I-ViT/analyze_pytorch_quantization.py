"""
深入分析 PyTorch 量化模型的整數運算邏輯
==================================================================
目標：完全理解 PyTorch 量化模型如何進行整數運算，
     以便在 C-Model 中完全複製這個邏輯。
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from models.vit_quant import deit_tiny_patch16_224


def analyze_quantized_module(module, name):
    """分析量化模組的參數和 buffer"""
    print(f"\n{'='*80}")
    print(f"模組: {name}")
    print(f"類型: {type(module).__name__}")
    print(f"{'='*80}")
    
    # 檢查所有參數
    print("\n參數 (Parameters):")
    for param_name, param in module.named_parameters(recurse=False):
        print(f"  {param_name}:")
        print(f"    shape: {param.shape}")
        print(f"    dtype: {param.dtype}")
        print(f"    range: [{param.min().item():.6f}, {param.max().item():.6f}]")
    
    # 檢查所有 buffer
    print("\nBuffer:")
    for buffer_name, buffer in module.named_buffers(recurse=False):
        print(f"  {buffer_name}:")
        print(f"    shape: {buffer.shape}")
        print(f"    dtype: {buffer.dtype}")
        if buffer.numel() > 0:
            print(f"    range: [{buffer.min().item():.6f}, {buffer.max().item():.6f}]")
    
    # 檢查特殊屬性
    print("\n特殊屬性:")
    special_attrs = ['weight_integer', 'bias_integer', 'fc_scaling_factor', 
                     'conv_scaling_factor', 'norm_scaling_factor',
                     'output_integer', 'input_zero_point', 'output_zero_point']
    
    for attr in special_attrs:
        if hasattr(module, attr):
            val = getattr(module, attr)
            if isinstance(val, torch.Tensor):
                print(f"  {attr}:")
                print(f"    shape: {val.shape}")
                print(f"    dtype: {val.dtype}")
                if val.numel() > 0 and val.numel() < 10:
                    print(f"    values: {val.tolist()}")
                elif val.numel() > 0:
                    print(f"    range: [{val.min().item():.6f}, {val.max().item():.6f}]")
            else:
                print(f"  {attr}: {val}")


def trace_forward_pass(model, image_tensor):
    """追蹤前向傳播過程"""
    print("\n" + "="*80)
    print("追蹤前向傳播過程")
    print("="*80)
    
    # 註冊 hooks 來追蹤每一層的輸入和輸出
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            # 保存輸入
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0]
                if isinstance(inp, torch.Tensor):
                    activations[f"{name}_input"] = {
                        'shape': inp.shape,
                        'dtype': inp.dtype,
                        'range': [inp.min().item(), inp.max().item()],
                        'mean': inp.mean().item(),
                        'std': inp.std().item()
                    }
            
            # 保存輸出
            if isinstance(output, torch.Tensor):
                activations[f"{name}_output"] = {
                    'shape': output.shape,
                    'dtype': output.dtype,
                    'range': [output.min().item(), output.max().item()],
                    'mean': output.mean().item(),
                    'std': output.std().item()
                }
            
            # 保存 output_integer (如果有)
            if hasattr(module, 'output_integer'):
                int_out = module.output_integer
                activations[f"{name}_output_integer"] = {
                    'shape': int_out.shape,
                    'dtype': int_out.dtype,
                    'range': [int_out.min().item(), int_out.max().item()],
                    'mean': int_out.mean().item(),
                    'std': int_out.std().item()
                }
        return hook
    
    # 註冊關鍵層的 hooks
    model.qact_input.register_forward_hook(make_hook("input_quant"))
    model.patch_embed.proj.register_forward_hook(make_hook("patch_embed"))
    model.qact1.register_forward_hook(make_hook("pos_embed"))
    
    # 第一個 block
    model.blocks[0].norm1.register_forward_hook(make_hook("block0_norm1"))
    model.blocks[0].attn.qkv.register_forward_hook(make_hook("block0_qkv"))
    model.blocks[0].attn.proj.register_forward_hook(make_hook("block0_proj"))
    model.blocks[0].norm2.register_forward_hook(make_hook("block0_norm2"))
    model.blocks[0].mlp.fc1.register_forward_hook(make_hook("block0_fc1"))
    model.blocks[0].mlp.fc2.register_forward_hook(make_hook("block0_fc2"))
    
    # 執行前向傳播
    with torch.no_grad():
        output = model(image_tensor)
    
    # 顯示結果
    print("\n激活值統計:")
    for name, stats in activations.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return activations


def compare_float_and_integer(model, image_tensor):
    """比較浮點輸出和整數輸出"""
    print("\n" + "="*80)
    print("比較浮點輸出和整數輸出")
    print("="*80)
    
    # 獲取第一個 LayerNorm
    norm1 = model.blocks[0].norm1
    
    print(f"\n分析 blocks.0.norm1:")
    print(f"  類型: {type(norm1).__name__}")
    
    # 檢查 forward 方法
    import inspect
    print(f"\nforward 方法簽名:")
    print(inspect.signature(norm1.forward))
    
    # 檢查源代碼
    print(f"\nforward 方法源代碼:")
    try:
        source = inspect.getsource(norm1.forward)
        print(source[:500])  # 只顯示前 500 個字符
    except:
        print("  無法獲取源代碼")
    
    # 執行前向傳播並檢查
    with torch.no_grad():
        # 先執行到 norm1 之前
        x = model.qact_input(image_tensor)
        x = model.patch_embed(x)
        x = x + model.pos_embed
        x = model.qact1(x)
        
        print(f"\nnorm1 輸入:")
        print(f"  shape: {x.shape}")
        print(f"  dtype: {x.dtype}")
        print(f"  range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        
        # 執行 norm1
        x_norm = norm1(x)
        
        print(f"\nnorm1 輸出 (float):")
        print(f"  shape: {x_norm.shape}")
        print(f"  dtype: {x_norm.dtype}")
        print(f"  range: [{x_norm.min().item():.6f}, {x_norm.max().item():.6f}]")
        
        # 檢查 output_integer
        if hasattr(norm1, 'output_integer'):
            int_out = norm1.output_integer
            print(f"\nnorm1 輸出 (integer):")
            print(f"  shape: {int_out.shape}")
            print(f"  dtype: {int_out.dtype}")
            print(f"  range: [{int_out.min().item():.6f}, {int_out.max().item():.6f}]")
            
            # 檢查 scaling factor
            if hasattr(norm1, 'norm_scaling_factor'):
                scale = norm1.norm_scaling_factor
                print(f"\nnorm_scaling_factor:")
                print(f"  shape: {scale.shape}")
                print(f"  dtype: {scale.dtype}")
                print(f"  values: {scale[:5].tolist() if scale.numel() > 5 else scale.tolist()}")
                
                # 驗證關係: float_output = integer_output * scaling_factor
                reconstructed = int_out.float() * scale.unsqueeze(0).unsqueeze(0)
                diff = torch.abs(x_norm - reconstructed)
                print(f"\n驗證 float = integer * scale:")
                print(f"  最大差異: {diff.max().item():.6f}")
                print(f"  平均差異: {diff.mean().item():.6f}")


def main():
    """主函數"""
    print("="*80)
    print("深入分析 PyTorch 量化模型")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    
    # 載入模型
    print("\n載入模型...")
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
    print("\n載入圖片...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print("✓ 圖片已載入")
    
    # 1. 分析關鍵模組
    print("\n" + "="*80)
    print("Part 1: 分析關鍵模組")
    print("="*80)
    
    analyze_quantized_module(model.qact_input, "qact_input")
    analyze_quantized_module(model.patch_embed.proj, "patch_embed.proj")
    analyze_quantized_module(model.blocks[0].norm1, "blocks.0.norm1")
    analyze_quantized_module(model.blocks[0].attn.qkv, "blocks.0.attn.qkv")
    
    # 2. 追蹤前向傳播
    print("\n" + "="*80)
    print("Part 2: 追蹤前向傳播")
    print("="*80)
    
    activations = trace_forward_pass(model, image_tensor)
    
    # 3. 比較浮點和整數輸出
    print("\n" + "="*80)
    print("Part 3: 比較浮點和整數輸出")
    print("="*80)
    
    compare_float_and_integer(model, image_tensor)
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == '__main__':
    main()
