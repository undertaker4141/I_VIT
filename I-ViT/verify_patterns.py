#!/usr/bin/env python3
"""
I-ViT Pattern Verification Script
==================================

驗證抽取的 patterns 是否正確

Usage:
    python verify_patterns.py --patterns ../patterns
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify extracted patterns")
    parser.add_argument('--patterns', type=str, default='../patterns',
                        help='Path to patterns directory')
    return parser.parse_args()


def check_file_exists(path, required=True):
    """檢查檔案是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else ("✗" if required else "○")
    return exists, status


def verify_weight(npy_path):
    """驗證權重檔案"""
    if not os.path.exists(npy_path):
        return None, "File not found"
    
    data = np.load(npy_path)
    info = {
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'min': float(data.min()),
        'max': float(data.max()),
    }
    
    # 驗證 INT8 範圍
    if data.dtype == np.int8:
        if data.min() < -128 or data.max() > 127:
            return info, "ERROR: Values out of INT8 range!"
        return info, "OK"
    
    # 驗證 INT32 範圍
    if data.dtype == np.int32:
        return info, "OK"
    
    # Float
    if data.dtype in [np.float32, np.float64]:
        return info, "OK (float)"
    
    return info, f"WARNING: Unexpected dtype {data.dtype}"


def verify_scale_ms(m_path, s_path, original_scale=None):
    """驗證 M/S 格式的 scale 是否可正確重建"""
    if not os.path.exists(m_path) or not os.path.exists(s_path):
        return None, "M or S file not found"
    
    M = np.load(m_path)
    S = np.load(s_path)
    
    # 重建 scale
    reconstructed = M.astype(np.float64) * (2.0 ** (-S.astype(np.float64)))
    
    info = {
        'M_dtype': str(M.dtype),
        'S_dtype': str(S.dtype),
        'M_range': [int(M.min()), int(M.max())],
        'S_range': [int(S.min()), int(S.max())],
        'reconstructed_range': [float(reconstructed.min()), float(reconstructed.max())],
    }
    
    if original_scale is not None:
        error = abs(reconstructed[0] - original_scale) / max(abs(original_scale), 1e-10)
        info['relative_error'] = float(error)
        if error > 0.01:  # 1% error threshold
            return info, f"WARNING: Reconstruction error {error:.4%}"
    
    return info, "OK"


def verify_hex_file(txt_path, npy_path):
    """驗證 Hex 檔案與 NPY 檔案是否一致"""
    if not os.path.exists(txt_path) or not os.path.exists(npy_path):
        return None, "File not found"
    
    npy_data = np.load(npy_path).flatten()
    
    # 讀取 hex 檔案
    hex_values = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 嘗試解析
            parts = line.split()
            for part in parts:
                try:
                    val = int(part, 16)
                    hex_values.append(val)
                except ValueError:
                    pass
    
    # 比較
    info = {
        'npy_count': len(npy_data),
        'hex_count': len(hex_values),
    }
    
    if len(hex_values) == 0:
        return info, "WARNING: No hex values parsed"
    
    # 檢查數量是否匹配
    # 注意: INT32 的 hex 每個值 8 個字元，INT8 每個值 2 個字元
    if abs(len(hex_values) - len(npy_data)) > 1:
        return info, f"WARNING: Count mismatch"
    
    return info, "OK"


def main():
    args = parse_args()
    patterns_dir = Path(args.patterns)
    
    print("=" * 60)
    print("I-ViT Pattern Verification")
    print("=" * 60)
    print(f"Patterns directory: {patterns_dir}")
    print()
    
    errors = []
    warnings = []
    
    # 1. Check directory structure
    print("[1] Directory Structure")
    print("-" * 40)
    
    dirs = ['input', 'embeddings', 'weights', 'scales', 'golden']
    for d in dirs:
        exists, status = check_file_exists(patterns_dir / d)
        print(f"  {status} {d}/")
        if not exists:
            errors.append(f"Missing directory: {d}")
    
    # Check config
    config_path = patterns_dir / "config.json"
    exists, status = check_file_exists(config_path)
    print(f"  {status} config.json")
    
    if exists:
        with open(config_path) as f:
            config = json.load(f)
        print(f"      Model: {config.get('model', 'unknown')}")
        print(f"      Predicted class: {config.get('predicted_class', 'unknown')}")
    
    # 2. Verify input
    print("\n[2] Input Verification")
    print("-" * 40)
    
    input_files = [
        ('input/input_float.npy', True),
        ('input/input_int8.npy', True),
        ('input/input_int8.txt', True),
        ('input/input_scale.json', True),
    ]
    
    for f, required in input_files:
        path = patterns_dir / f
        exists, status = check_file_exists(path, required)
        print(f"  {status} {f}")
        
        if exists and f.endswith('.npy'):
            info, msg = verify_weight(path)
            if info:
                print(f"      Shape: {info['shape']}, dtype: {info['dtype']}")
                print(f"      Range: [{info['min']:.4f}, {info['max']:.4f}]")
            if 'ERROR' in msg:
                errors.append(f"{f}: {msg}")
            elif 'WARNING' in msg:
                warnings.append(f"{f}: {msg}")
    
    # 3. Verify weights
    print("\n[3] Weights Verification")
    print("-" * 40)
    
    weights_dir = patterns_dir / "weights"
    if weights_dir.exists():
        weight_files = list(weights_dir.glob("*.npy"))
        print(f"  Found {len(weight_files)} weight files")
        
        # Check some key weights
        key_weights = [
            'patch_embed_proj_weight.npy',
            'blocks_0_attn_qkv_weight.npy',
            'blocks_0_mlp_fc1_weight.npy',
        ]
        
        for w in key_weights:
            path = weights_dir / w
            if path.exists():
                info, msg = verify_weight(path)
                status = "✓" if "OK" in msg else "⚠"
                print(f"  {status} {w}")
                if info:
                    print(f"      {info['shape']} {info['dtype']} [{info['min']}, {info['max']}]")
                if 'ERROR' in msg:
                    errors.append(f"weights/{w}: {msg}")
        
        # Check bias files are INT32
        bias_files = list(weights_dir.glob("*_bias.npy"))
        int32_count = 0
        for bf in bias_files:
            data = np.load(bf)
            if data.dtype == np.int32:
                int32_count += 1
        
        print(f"  Bias files: {len(bias_files)} total, {int32_count} are INT32")
        if int32_count < len(bias_files) and len(bias_files) > 0:
            warnings.append(f"Some bias files are not INT32")
    
    # 4. Verify scales
    print("\n[4] Scales Verification")
    print("-" * 40)
    
    scales_dir = patterns_dir / "scales"
    if scales_dir.exists():
        m_files = list(scales_dir.glob("*_M.npy"))
        s_files = list(scales_dir.glob("*_S.npy"))
        print(f"  Found {len(m_files)} M files, {len(s_files)} S files")
        
        # Check all_scales.json
        all_scales_path = scales_dir / "all_scales.json"
        if all_scales_path.exists():
            with open(all_scales_path) as f:
                all_scales = json.load(f)
            print(f"  all_scales.json contains {len(all_scales)} entries")
            
            # Verify M/S reconstruction for some scales
            for name, scale_val in list(all_scales.items())[:3]:
                m_path = scales_dir / f"{name}_M.npy"
                s_path = scales_dir / f"{name}_S.npy"
                
                if m_path.exists() and s_path.exists():
                    orig = scale_val if isinstance(scale_val, float) else scale_val[0]
                    info, msg = verify_scale_ms(m_path, s_path, orig)
                    status = "✓" if "OK" in msg else "⚠"
                    print(f"  {status} {name}")
                    if info and 'relative_error' in info:
                        print(f"      Reconstruction error: {info['relative_error']:.6%}")
    
    # 5. Verify golden outputs
    print("\n[5] Golden Outputs Verification")
    print("-" * 40)
    
    golden_dir = patterns_dir / "golden"
    if golden_dir.exists():
        float_files = list(golden_dir.glob("*_float.npy"))
        int_files = list(golden_dir.glob("*_int.npy"))
        print(f"  Float outputs: {len(float_files)}")
        print(f"  Int outputs: {len(int_files)}")
        
        # Check layer_info.json
        layer_info_path = golden_dir / "layer_info.json"
        if layer_info_path.exists():
            with open(layer_info_path) as f:
                layer_info = json.load(f)
            print(f"  layer_info.json contains {len(layer_info)} entries")
        
        # Check final output
        final_output_path = golden_dir / "final_output.npy"
        if final_output_path.exists():
            final = np.load(final_output_path)
            pred = final.argmax()
            print(f"  ✓ final_output.npy: shape {final.shape}, pred class {pred}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    
    if not errors and not warnings:
        print("\n✅ All checks passed!")
    elif not errors:
        print(f"\n✅ Passed with {len(warnings)} warning(s)")
    else:
        print(f"\n❌ Failed with {len(errors)} error(s)")
    
    print()
    return len(errors)


if __name__ == "__main__":
    exit(main())
