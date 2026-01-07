#!/usr/bin/env python3
"""
I-ViT Pattern Extraction Script
================================

從 QAT 訓練後的模型抽取硬體加速器所需的 patterns:
- INT8 權重 + INT32 Bias
- Scaling Factors (M/S 格式)
- 每層的 Golden Outputs (INT + Float)

Usage:
    python extract_patterns.py \
        --checkpoint checkpoints/qat_calibrated.pth \
        --test-image ../test_data/test_image.JPEG \
        --output ../patterns
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import decimal

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import *
from models.quantization_utils.quant_utils import batch_frexp


def parse_args():
    parser = argparse.ArgumentParser(description="I-ViT Pattern Extraction")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to QAT checkpoint')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--output', type=str, default='../patterns',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='deit_tiny',
                        choices=['deit_tiny', 'deit_small', 'deit_base'])
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def str2model(name):
    """Get model constructor by name"""
    models = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
    }
    return models[name]


def load_and_preprocess_image(image_path, device):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor, image


def scale_to_ms(scale_tensor, max_bit=31):
    """
    將 floating-point scale 轉換為 M (Mantissa) 和 S (Shift) 格式
    
    公式: scale ≈ M * 2^(-S)
    
    與 I-ViT 的 batch_frexp 相容
    """
    scale = scale_tensor.detach().cpu().numpy().flatten()
    
    if len(scale) == 0:
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int32)
    
    # 使用 numpy frexp
    mantissa, exponent = np.frexp(scale)
    
    # 將 mantissa [0.5, 1.0) 乘以 2^31 變成整數
    M = np.round(mantissa * (2 ** max_bit)).astype(np.int64)
    
    # S = max_bit - exponent (這樣 M * 2^(-S) ≈ scale)
    S = (max_bit - exponent).astype(np.int32)
    
    return M, S


def save_npy_and_hex(data, base_path, dtype_hint=None):
    """
    儲存 NumPy 陣列為 .npy 和 .txt (hex) 格式
    
    Args:
        data: numpy array
        base_path: 檔案路徑 (不含副檔名)
        dtype_hint: 'int8', 'int32', 'int64', 'float32'
    """
    # Save npy
    np.save(f"{base_path}.npy", data)
    
    # Determine format for hex
    if dtype_hint is None:
        if data.dtype in [np.float32, np.float64]:
            return  # 浮點數不輸出 hex
        elif data.dtype == np.int8:
            dtype_hint = 'int8'
        elif data.dtype in [np.int32, np.int64]:
            dtype_hint = 'int32'
        else:
            dtype_hint = 'int8'
    
    # Save hex txt
    with open(f"{base_path}.txt", 'w') as f:
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Dtype: {data.dtype}\n")
        
        flat = data.flatten()
        
        if dtype_hint == 'int8':
            f.write("# Format: space-separated 2-digit hex\n")
            # 每行 16 個值
            for i in range(0, len(flat), 16):
                row = flat[i:i+16]
                hex_values = [f"{int(v) & 0xFF:02X}" for v in row]
                f.write(" ".join(hex_values) + "\n")
        
        elif dtype_hint in ['int32', 'int64']:
            f.write("# Format: one value per line, 8-digit hex, big-endian\n")
            f.write("# Compatible with Verilog $readmemh\n")
            for v in flat:
                # 32-bit signed to unsigned
                v_int = int(v)
                if v_int < 0:
                    v_int = v_int & 0xFFFFFFFF
                f.write(f"{v_int:08X}\n")


class PatternExtractor:
    """Pattern 抽取器：使用 hook 捕捉每層的中間結果"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.hooks = []
        self.intermediate_outputs = {}
        self.intermediate_outputs_int = {}
        
        # 註冊 hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """為每個需要捕捉的層註冊 forward hook"""
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # (output, scaling_factor) 格式
                    out = output[0]
                    scale = output[1] if len(output) > 1 else None
                else:
                    out = output
                    scale = None
                
                # 儲存 float 版本
                self.intermediate_outputs[name] = out.detach().cpu()
                
                # 嘗試計算 int 版本
                if scale is not None:
                    try:
                        int_out = (out / scale).round()
                        self.intermediate_outputs_int[name] = int_out.detach().cpu()
                    except:
                        pass
                
                # 儲存 scale
                if scale is not None:
                    self.intermediate_outputs[f"{name}_scale"] = scale.detach().cpu()
            
            return hook
        
        # 註冊到各層
        for name, module in self.model.named_modules():
            # QuantAct 層 (activation quantization)
            if 'qact' in name.lower():
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # IntSoftmax
            elif isinstance(module, IntSoftmax):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # IntGELU
            elif isinstance(module, IntGELU):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # IntLayerNorm
            elif isinstance(module, IntLayerNorm):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # QuantLinear
            elif isinstance(module, QuantLinear):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # QuantConv2d
            elif isinstance(module, QuantConv2d):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
            
            # QuantMatMul
            elif isinstance(module, QuantMatMul):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def extract(self, input_tensor):
        """執行 forward pass 並抽取所有中間結果"""
        self.intermediate_outputs = {}
        self.intermediate_outputs_int = {}
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return output


def extract_weights(model, output_dir):
    """抽取所有權重和 bias"""
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    weight_info = {}
    
    for name, module in model.named_modules():
        # QuantLinear
        if isinstance(module, QuantLinear):
            prefix = name.replace('.', '_')
            
            # Weight (INT8)
            if hasattr(module, 'weight_integer'):
                w = module.weight_integer.detach().cpu().numpy().astype(np.int8)
                save_npy_and_hex(w, weights_dir / f"{prefix}_weight", 'int8')
                weight_info[f"{prefix}_weight"] = {
                    'shape': list(w.shape), 'dtype': 'int8',
                    'min': int(w.min()), 'max': int(w.max())
                }
            
            # Bias (INT32)
            if hasattr(module, 'bias_integer') and module.bias_integer is not None:
                b = module.bias_integer.detach().cpu().numpy().astype(np.int32)
                save_npy_and_hex(b, weights_dir / f"{prefix}_bias", 'int32')
                weight_info[f"{prefix}_bias"] = {
                    'shape': list(b.shape), 'dtype': 'int32',
                    'min': int(b.min()), 'max': int(b.max())
                }
        
        # QuantConv2d
        elif isinstance(module, QuantConv2d):
            prefix = name.replace('.', '_')
            
            # Weight (INT8)
            if hasattr(module, 'weight_integer'):
                w = module.weight_integer.detach().cpu().numpy().astype(np.int8)
                save_npy_and_hex(w, weights_dir / f"{prefix}_weight", 'int8')
                weight_info[f"{prefix}_weight"] = {
                    'shape': list(w.shape), 'dtype': 'int8',
                    'min': int(w.min()), 'max': int(w.max())
                }
            
            # Bias (INT32)
            if hasattr(module, 'bias_integer') and module.bias_integer is not None:
                b = module.bias_integer.detach().cpu().numpy().astype(np.int32)
                save_npy_and_hex(b, weights_dir / f"{prefix}_bias", 'int32')
                weight_info[f"{prefix}_bias"] = {
                    'shape': list(b.shape), 'dtype': 'int32',
                    'min': int(b.min()), 'max': int(b.max())
                }
        
        # IntLayerNorm
        elif isinstance(module, IntLayerNorm):
            prefix = name.replace('.', '_')
            
            # Weight (FP32 - 作為 scale)
            w = module.weight.detach().cpu().numpy().astype(np.float32)
            np.save(weights_dir / f"{prefix}_weight.npy", w)
            weight_info[f"{prefix}_weight"] = {
                'shape': list(w.shape), 'dtype': 'float32',
                'min': float(w.min()), 'max': float(w.max())
            }
            
            # Bias Integer (INT)
            if hasattr(module, 'bias_integer'):
                b = module.bias_integer.detach().cpu().numpy().astype(np.int32)
                save_npy_and_hex(b, weights_dir / f"{prefix}_bias_integer", 'int32')
                weight_info[f"{prefix}_bias_integer"] = {
                    'shape': list(b.shape), 'dtype': 'int32',
                    'min': int(b.min()), 'max': int(b.max())
                }
    
    return weight_info


def extract_embeddings(model, output_dir, device):
    """抽取 cls_token 和 pos_embed"""
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    
    embed_info = {}
    
    # cls_token
    if hasattr(model, 'cls_token'):
        cls_token = model.cls_token.detach().cpu().numpy().astype(np.float32)
        np.save(embed_dir / "cls_token_float.npy", cls_token)
        embed_info['cls_token'] = {
            'shape': list(cls_token.shape), 
            'dtype': 'float32',
            'note': 'Quantized during forward pass by qact1'
        }
        
        # 模擬量化 (使用 qact_pos 的 scale)
        if hasattr(model, 'qact1') and hasattr(model.qact1, 'act_scaling_factor'):
            scale = model.qact1.act_scaling_factor.detach().cpu().numpy()
            scale_val = float(scale.flatten()[0]) if scale.size > 0 else 0
            if scale_val != 0:
                cls_int = np.round(cls_token / scale_val).astype(np.int8)
                save_npy_and_hex(cls_int, embed_dir / "cls_token_int8", 'int8')
                
                M, S = scale_to_ms(model.qact1.act_scaling_factor)
                np.save(embed_dir / "cls_token_scale_M.npy", M)
                np.save(embed_dir / "cls_token_scale_S.npy", S)
    
    # pos_embed
    if hasattr(model, 'pos_embed'):
        pos_embed = model.pos_embed.detach().cpu().numpy().astype(np.float32)
        np.save(embed_dir / "pos_embed_float.npy", pos_embed)
        embed_info['pos_embed'] = {
            'shape': list(pos_embed.shape), 
            'dtype': 'float32',
            'note': 'Quantized during forward pass by qact_pos'
        }
        
        # 模擬量化
        if hasattr(model, 'qact_pos') and hasattr(model.qact_pos, 'act_scaling_factor'):
            scale = model.qact_pos.act_scaling_factor.detach().cpu().numpy()
            scale_val = float(scale.flatten()[0]) if scale.size > 0 else 0
            if scale_val != 0:
                pos_int = np.round(pos_embed / scale_val).astype(np.int8)
                save_npy_and_hex(pos_int, embed_dir / "pos_embed_int8", 'int8')
                
                M, S = scale_to_ms(model.qact_pos.act_scaling_factor)
                np.save(embed_dir / "pos_embed_scale_M.npy", M)
                np.save(embed_dir / "pos_embed_scale_S.npy", S)
    
    return embed_info


def extract_scales(model, output_dir):
    """抽取所有 scaling factors 並轉換為 M/S 格式"""
    scales_dir = output_dir / "scales"
    scales_dir.mkdir(parents=True, exist_ok=True)
    
    all_scales = {}
    scale_info = {}
    
    for name, module in model.named_modules():
        prefix = name.replace('.', '_')
        
        # act_scaling_factor
        if hasattr(module, 'act_scaling_factor'):
            sf = module.act_scaling_factor
            if sf is not None and sf.numel() > 0:
                scale_val = sf.detach().cpu().numpy().flatten()
                if len(scale_val) > 0 and scale_val[0] != 0:
                    all_scales[f"{prefix}_act"] = float(scale_val[0])
                    
                    M, S = scale_to_ms(sf)
                    np.save(scales_dir / f"{prefix}_act_M.npy", M)
                    np.save(scales_dir / f"{prefix}_act_S.npy", S)
                    
                    # Direction: I-ViT 都是右移
                    np.save(scales_dir / f"{prefix}_act_direction.npy", 
                           np.array([-1], dtype=np.int8))
                    
                    scale_info[f"{prefix}_act"] = {
                        'float_value': float(scale_val[0]),
                        'M': int(M[0]),
                        'S': int(S[0]),
                        'direction': 'right_shift'
                    }
        
        # fc_scaling_factor
        if hasattr(module, 'fc_scaling_factor'):
            sf = module.fc_scaling_factor
            if sf is not None and sf.numel() > 0:
                scale_val = sf.detach().cpu().numpy().flatten()
                all_scales[f"{prefix}_fc"] = [float(v) for v in scale_val[:5]]  # 只存前 5 個示例
                
                M, S = scale_to_ms(sf)
                np.save(scales_dir / f"{prefix}_fc_M.npy", M)
                np.save(scales_dir / f"{prefix}_fc_S.npy", S)
                np.save(scales_dir / f"{prefix}_fc_direction.npy", 
                       np.array([-1], dtype=np.int8))
        
        # conv_scaling_factor
        if hasattr(module, 'conv_scaling_factor'):
            sf = module.conv_scaling_factor
            if sf is not None and sf.numel() > 0:
                scale_val = sf.detach().cpu().numpy().flatten()
                all_scales[f"{prefix}_conv"] = [float(v) for v in scale_val[:5]]
                
                M, S = scale_to_ms(sf)
                np.save(scales_dir / f"{prefix}_conv_M.npy", M)
                np.save(scales_dir / f"{prefix}_conv_S.npy", S)
                np.save(scales_dir / f"{prefix}_conv_direction.npy", 
                       np.array([-1], dtype=np.int8))
        
        # norm_scaling_factor
        if hasattr(module, 'norm_scaling_factor'):
            sf = module.norm_scaling_factor
            if sf is not None and sf.numel() > 0:
                scale_val = sf.detach().cpu().numpy().flatten()
                all_scales[f"{prefix}_norm"] = [float(v) for v in scale_val[:5]]
                
                M, S = scale_to_ms(sf)
                np.save(scales_dir / f"{prefix}_norm_M.npy", M)
                np.save(scales_dir / f"{prefix}_norm_S.npy", S)
    
    # Save all scales as JSON
    with open(scales_dir / "all_scales.json", 'w') as f:
        json.dump(all_scales, f, indent=2)
    
    return scale_info


def extract_golden_outputs(extractor, output_dir):
    """從 extractor 的中間結果抽取 golden outputs"""
    golden_dir = output_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    golden_info = {}
    
    # Float outputs
    for name, tensor in extractor.intermediate_outputs.items():
        if '_scale' in name:
            continue  # Skip scales
        
        prefix = name.replace('.', '_')
        data = tensor.numpy().astype(np.float32)
        np.save(golden_dir / f"{prefix}_float.npy", data)
        
        golden_info[prefix] = {
            'shape': list(data.shape),
            'dtype_float': 'float32',
            'min_float': float(data.min()),
            'max_float': float(data.max()),
        }
    
    # Int outputs
    for name, tensor in extractor.intermediate_outputs_int.items():
        prefix = name.replace('.', '_')
        data = tensor.numpy()
        
        # 判斷適當的 int dtype
        if data.max() <= 127 and data.min() >= -128:
            data = data.astype(np.int8)
            dtype = 'int8'
        else:
            data = data.astype(np.int32)
            dtype = 'int32'
        
        save_npy_and_hex(data, golden_dir / f"{prefix}_int", dtype)
        
        if prefix in golden_info:
            golden_info[prefix]['dtype_int'] = dtype
            golden_info[prefix]['min_int'] = int(data.min())
            golden_info[prefix]['max_int'] = int(data.max())
    
    # Save info
    with open(golden_dir / "layer_info.json", 'w') as f:
        json.dump(golden_info, f, indent=2)
    
    return golden_info


def main():
    args = parse_args()
    
    print("=" * 60)
    print("I-ViT Pattern Extraction")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test image: {args.test_image}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/6] Loading model...")
    model = str2model(args.model)(pretrained=True, num_classes=1000)
    
    # Load checkpoint if exists
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out scaling factors with shape mismatch (they will be recalculated)
        model_state = model.state_dict()
        filtered_state = {}
        skipped = 0
        for k, v in state_dict.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped += 1
            else:
                filtered_state[k] = v
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded checkpoint (skipped {skipped} mismatched scaling factors)")
    else:
        print("Warning: Checkpoint not found, using pretrained weights")
    
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    print("\n[2/6] Loading test image...")
    input_tensor, original_image = load_and_preprocess_image(args.test_image, device)
    
    # Save input
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy original image
    original_image.save(input_dir / "test_image.png")
    
    # Save float input
    input_float = input_tensor.cpu().numpy().astype(np.float32)
    np.save(input_dir / "input_float.npy", input_float)
    
    # Run one forward to populate quantization parameters
    print("\n[3/6] Running forward pass to collect quantization params...")
    extractor = PatternExtractor(model, device)
    
    # First forward to initialize all quantization params
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Save quantized input
    if hasattr(model, 'qact_input') and hasattr(model.qact_input, 'act_scaling_factor'):
        scale = model.qact_input.act_scaling_factor.cpu().numpy()
        scale_val = float(scale.flatten()[0]) if scale.size > 0 else 0
        if scale_val != 0:
            input_int = np.round(input_float / scale_val).astype(np.int8)
            save_npy_and_hex(input_int, input_dir / "input_int8", 'int8')
            
            M, S = scale_to_ms(model.qact_input.act_scaling_factor)
            with open(input_dir / "input_scale.json", 'w') as f:
                json.dump({
                    'float_scale': scale_val,
                    'M': int(M.flatten()[0]),
                    'S': int(S.flatten()[0])
                }, f, indent=2)
    
    # Extract patterns
    print("\n[4/6] Extracting weights...")
    weight_info = extract_weights(model, output_dir)
    print(f"  Extracted {len(weight_info)} weight tensors")
    
    print("\n[5/6] Extracting embeddings...")
    embed_info = extract_embeddings(model, output_dir, device)
    print(f"  Extracted {len(embed_info)} embedding tensors")
    
    print("\n[6/6] Extracting scales and golden outputs...")
    scale_info = extract_scales(model, output_dir)
    print(f"  Extracted {len(scale_info)} scaling factors")
    
    # Run forward with hooks to get golden outputs
    output = extractor.extract(input_tensor)
    golden_info = extract_golden_outputs(extractor, output_dir)
    print(f"  Extracted {len(golden_info)} golden outputs")
    
    # Save final output
    final_output = output.cpu().numpy()
    np.save(output_dir / "golden" / "final_output.npy", final_output)
    
    # Get prediction
    pred_class = final_output.argmax()
    print(f"\n  Predicted class: {pred_class}")
    
    # Clean up
    extractor.remove_hooks()
    
    # Save config
    config = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'test_image': args.test_image,
        'extraction_time': datetime.now().isoformat(),
        'input_shape': list(input_tensor.shape),
        'num_weights': len(weight_info),
        'num_scales': len(scale_info),
        'num_golden': len(golden_info),
        'predicted_class': int(pred_class),
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Pattern Extraction Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - input/       : Input image and quantized tensor")
    print(f"  - embeddings/  : cls_token, pos_embed")
    print(f"  - weights/     : INT8 weights, INT32 biases")
    print(f"  - scales/      : M/S format scaling factors")
    print(f"  - golden/      : Layer outputs (INT + Float)")
    print("=" * 60)


if __name__ == "__main__":
    main()
