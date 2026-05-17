#!/usr/bin/env python3
"""
TVM Pattern Extraction Script
==============================

從 TVM 全整數推理中抽取所有參數和中間層 golden outputs。
使用 TVM v0.14 relay API，在 relay graph 中插入 debug outputs 來
取得每個關鍵層的中間整數值。

Usage:
    cd I-ViT/TVM_benchmark
    python extract_tvm_patterns.py \
        --checkpoint ../checkpoints/qat_calibrated.pth \
        --test-image ../../data/test_image.JPEG \
        --output ../../patterns_tvm
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image

# Add TVM_benchmark to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tvm
from tvm import relay
import convert_model
from models.layers import QuantizeContext
import models.build_model as build_model


def parse_args():
    parser = argparse.ArgumentParser(description="TVM Pattern Extraction")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to QAT checkpoint')
    parser.add_argument('--test-image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--output', type=str, default='../../patterns_tvm',
                        help='Output directory')
    parser.add_argument('--model-name', type=str, default='deit_tiny_patch16_224',
                        choices=['deit_tiny_patch16_224', 'deit_small_patch16_224',
                                 'deit_base_patch16_224'])
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--target', type=str, default='llvm',
                        help='TVM target (llvm for CPU)')
    return parser.parse_args()


def save_npy_and_hex(data, base_path, dtype_hint=None):
    """儲存 NumPy 陣列為 .npy 和 .txt (hex) 格式"""
    np.save(f"{base_path}.npy", data)
    if dtype_hint is None:
        if data.dtype in [np.float32, np.float64]:
            return
        elif data.dtype == np.int8:
            dtype_hint = 'int8'
        elif data.dtype in [np.int32, np.int64]:
            dtype_hint = 'int32'
        else:
            return
    with open(f"{base_path}.txt", 'w') as f:
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Dtype: {data.dtype}\n")
        flat = data.flatten()
        if dtype_hint == 'int8':
            for i in range(0, len(flat), 16):
                row = flat[i:i+16]
                f.write(" ".join(f"{int(v) & 0xFF:02X}" for v in row) + "\n")
        elif dtype_hint in ['int32', 'int64']:
            for v in flat:
                v_int = int(v) & 0xFFFFFFFF if int(v) < 0 else int(v)
                f.write(f"{v_int:08X}\n")


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy(), image


def scale_to_ms(scale_val, max_bit=31):
    """將 float scale 轉換為 M/S 格式"""
    scale_val = np.atleast_1d(np.array(scale_val, dtype=np.float64))
    mantissa, exponent = np.frexp(scale_val)
    M = np.round(mantissa * (2 ** max_bit)).astype(np.int64)
    S = (max_bit - exponent).astype(np.int32)
    return M, S


# =============================================================================
# 多輸出版本的模型組建
# 修改 quantized_vit.py 讓它回傳每個關鍵節點的中間 output
# =============================================================================

from models import layers

def Q_Block_with_outputs(data, name, dim, num_heads, mlp_ratio, qk_scale, batch_size,
                         rounding='TRUNCATE'):
    """和 quantized_vit.py 的 Q_Block 相同，但同時收集中間節點的 relay expr"""
    outputs = {}

    # LayerNorm 1
    qconfig0 = layers.get_qconfig(name + '_qconfig_norm1')
    norm1_bias = relay.var(name + '_norm1_bias', shape=[dim], dtype='int32')
    norm1 = layers.quantized_layernorm(data, norm1_bias)
    outputs[f'{name}_norm1'] = norm1

    # Requantize after norm1 -> qkv input
    qconfig1 = layers.get_qconfig(name + '_qconfig_qkv')
    req1 = layers.requantize(norm1, input_scale=qconfig0.output_scale,
                             output_scale=qconfig1.input_scale,
                             out_dtype=qconfig1.input_dtype)
    outputs[f'{name}_req_norm1_to_qkv'] = req1

    # QKV Dense
    req1 = relay.reshape(req1, [-3, 0])
    qkv = layers.quantized_dense(data=req1, name=name + '_attn_qkv',
                                  input_scale=qconfig1.input_scale,
                                  kernel_scale=qconfig1.kernel_scale,
                                  units=dim*3, kernel_shape=(dim*3, dim),
                                  kernel_dtype='int8', add_bias=True)
    qkv = relay.reshape(qkv, [-4, batch_size, -1, -2])
    outputs[f'{name}_qkv'] = qkv

    # Requantize for matmul_1
    qconfig2 = layers.get_qconfig(name + '_qconfig_matmul_1')
    req2 = layers.requantize(qkv, input_scale=qconfig1.output_scale,
                             output_scale=qconfig2.input_scale,
                             out_dtype=qconfig2.input_dtype)
    outputs[f'{name}_req_qkv_to_matmul1'] = req2

    # Q, K, V split
    qkv_reshape = relay.reshape(req2, [0, 0, 3, num_heads, -1])
    qkv_t = relay.transpose(qkv_reshape, [2, 0, 3, 1, 4])
    qkv_split = relay.split(qkv_t, 3, axis=0)
    q = relay.reshape(relay.squeeze(qkv_split[0], axis=[0]), [-3, -2])
    k = relay.reshape(relay.squeeze(qkv_split[1], axis=[0]), [-3, -2])
    v = relay.reshape(relay.squeeze(qkv_split[2], axis=[0]), [-3, -2])

    # Attention matmul (Q @ K^T)
    attn = layers.quantized_matmul(q, k, input_scale1=qconfig2.input_scale,
                                   input_scale2=qconfig2.input_scale)
    attn = relay.reshape(attn, [-4, -1, num_heads, -2])
    outputs[f'{name}_attn_matmul1'] = attn

    # Softmax
    qconfig3 = layers.get_qconfig(name + '_qconfig_softmax')
    req3 = layers.requantize(attn, input_scale=qconfig2.output_scale * qk_scale,
                             output_scale=qconfig3.input_scale,
                             out_dtype=qconfig3.input_dtype)
    attn_softmax = layers.quantized_softmax(req3, qconfig3.input_scale)
    outputs[f'{name}_softmax'] = attn_softmax

    # Attention matmul (softmax @ V)
    qconfig4 = layers.get_qconfig(name + '_qconfig_matmul_2')
    attn_r = relay.reshape(attn_softmax, [-3, -2])
    v_t = relay.transpose(v, [0, 2, 1])
    attn2 = layers.quantized_matmul(attn_r, v_t, input_scale1=qconfig4.input_scale,
                                    input_scale2=qconfig2.input_scale)
    attn2 = relay.reshape(attn2, [-4, -1, num_heads, -2])
    attn2 = relay.transpose(attn2, [0, 2, 1, 3])
    attn2 = relay.reshape(attn2, [0, 0, -1])
    outputs[f'{name}_attn_matmul2'] = attn2

    # Projection
    qconfig5 = layers.get_qconfig(name + '_qconfig_proj')
    req5 = layers.requantize(attn2, input_scale=qconfig4.output_scale,
                             output_scale=qconfig5.input_scale,
                             out_dtype=qconfig5.input_dtype)
    req5_r = relay.reshape(req5, [-3, 0])
    proj = layers.quantized_dense(data=req5_r, name=name + '_attn_proj',
                                   input_scale=qconfig5.input_scale,
                                   kernel_scale=qconfig5.kernel_scale,
                                   units=dim, kernel_shape=(dim, dim),
                                   kernel_dtype='int8', add_bias=True)
    proj = relay.reshape(proj, [-4, batch_size, -1, -2])
    outputs[f'{name}_proj'] = proj

    # Add1 (shortcut)
    shortcut = data
    qconfig6 = layers.get_qconfig(name + '_qconfig_add1')
    req6 = layers.requantize(proj, input_scale=qconfig5.output_scale,
                             output_scale=qconfig6.input_scale,
                             out_dtype=qconfig6.input_dtype)
    add1 = layers.add(lhs=req6, rhs=shortcut,
                      lhs_scale=qconfig6.input_scale, rhs_scale=qconfig0.input_scale,
                      output_scale=qconfig6.output_scale)
    outputs[f'{name}_add1'] = add1

    # LayerNorm 2
    qconfig7 = layers.get_qconfig(name + '_qconfig_norm2')
    norm2_bias = relay.var(name + '_norm2_bias', shape=[dim], dtype='int32')
    norm2 = layers.quantized_layernorm(add1, norm2_bias)
    outputs[f'{name}_norm2'] = norm2

    # FC1
    qconfig8 = layers.get_qconfig(name + '_qconfig_fc1')
    req8 = layers.requantize(norm2, input_scale=qconfig7.output_scale,
                             output_scale=qconfig8.input_scale,
                             out_dtype=qconfig8.input_dtype)
    req8_r = relay.reshape(req8, [-3, 0])
    fc1 = layers.quantized_dense(data=req8_r, name=name + '_mlp_fc1',
                                  input_scale=qconfig8.input_scale,
                                  kernel_scale=qconfig8.kernel_scale,
                                  units=mlp_ratio*dim,
                                  kernel_shape=(mlp_ratio*dim, dim),
                                  kernel_dtype='int8', add_bias=True)
    fc1 = relay.reshape(fc1, [-4, batch_size, -1, -2])
    outputs[f'{name}_fc1'] = fc1

    # GELU
    qconfig9 = layers.get_qconfig(name + '_qconfig_gelu')
    req9 = layers.requantize(fc1, input_scale=qconfig8.output_scale,
                             output_scale=qconfig9.input_scale,
                             out_dtype=qconfig9.input_dtype)
    act = layers.quantized_gelu(req9, qconfig9.input_scale)
    outputs[f'{name}_gelu'] = act

    # FC2
    qconfig10 = layers.get_qconfig(name + '_qconfig_fc2')
    req10 = layers.requantize(act, input_scale=qconfig9.output_scale,
                              output_scale=qconfig10.input_scale,
                              out_dtype=qconfig10.input_dtype)
    req10_r = relay.reshape(req10, [-3, 0])
    fc2 = layers.quantized_dense(data=req10_r, name=name + '_mlp_fc2',
                                  input_scale=qconfig10.input_scale,
                                  kernel_scale=qconfig10.kernel_scale,
                                  units=dim, kernel_shape=(dim, mlp_ratio*dim),
                                  kernel_dtype='int8', add_bias=True)
    fc2 = relay.reshape(fc2, [-4, batch_size, -1, -2])
    outputs[f'{name}_fc2'] = fc2

    # Add2 (shortcut)
    shortcut2 = add1
    qconfig11 = layers.get_qconfig(name + '_qconfig_add2')
    req11 = layers.requantize(fc2, input_scale=qconfig10.output_scale,
                              output_scale=qconfig11.input_scale,
                              out_dtype=qconfig11.input_dtype)
    add2 = layers.add(lhs=req11, rhs=shortcut2,
                      lhs_scale=qconfig11.input_scale, rhs_scale=qconfig7.input_scale,
                      output_scale=qconfig11.output_scale)
    outputs[f'{name}_add2'] = add2

    add2 = relay.annotation.stop_fusion(add2)

    return add2, outputs


def build_model_with_outputs(data_shape, model_name, depth=12):
    """構建模型並返回所有中間層的 relay expr"""
    if model_name == 'deit_tiny_patch16_224':
        embed_dim, num_heads = 192, 3
    elif model_name == 'deit_small_patch16_224':
        embed_dim, num_heads = 384, 6
    elif model_name == 'deit_base_patch16_224':
        embed_dim, num_heads = 768, 12
    else:
        raise ValueError(f"Unknown model: {model_name}")

    batch_size = data_shape[0]
    all_outputs = {}

    data = relay.var('data', shape=data_shape, dtype='int8')

    # Embedding convolution
    qconfig_embed_conv = layers.get_qconfig('qconfig_embed_conv')
    proj = layers.quantized_conv2d(data=data, name='embed_conv', add_bias=True,
                                    input_channels=3, output_channels=embed_dim,
                                    kernel_dtype=qconfig_embed_conv.kernel_dtype,
                                    input_scale=qconfig_embed_conv.input_scale,
                                    kernel_scale=qconfig_embed_conv.kernel_scale,
                                    kernel_size=(16, 16), strides=(16, 16), padding=(0, 0),
                                    data_layout='NCHW', kernel_layout='OIHW')
    proj = relay.reshape(proj, [0, 0, -1])
    body = relay.transpose(proj, [0, 2, 1])
    all_outputs['embed_conv'] = body

    # Requantize + add pos embed
    qconfig_add = layers.get_qconfig('qconfig_addpos')
    body = layers.requantize(body, input_scale=qconfig_embed_conv.output_scale,
                             output_scale=qconfig_add.input_scale,
                             out_dtype=qconfig_add.input_dtype)

    cls_token = relay.var('cls_token_weight', shape=(1, 1, embed_dim))
    cls_token = layers.quantize(cls_token, output_scale=qconfig_add.input_scale,
                                out_dtype=qconfig_add.input_dtype)
    cls_tokens = relay.repeat(cls_token, data_shape[0], axis=0)
    body = relay.concatenate([cls_tokens, body], axis=1)

    pos_embed = relay.var('pos_embed_weight', shape=(1, 197, embed_dim))
    qconfig_pos = layers.get_qconfig('qconfig_pos')
    pos_embed = layers.quantize(pos_embed, output_scale=qconfig_pos.output_scale,
                                out_dtype=qconfig_add.input_dtype)
    body = layers.add(lhs=body, rhs=pos_embed,
                      lhs_scale=qconfig_add.input_scale,
                      rhs_scale=qconfig_pos.output_scale,
                      output_scale=qconfig_add.output_scale)
    body = relay.annotation.stop_fusion(body)
    all_outputs['embed_add_pos'] = body

    qk_scale = (embed_dim // num_heads) ** -0.5

    # Transformer blocks
    for i in range(depth):
        body, block_outputs = Q_Block_with_outputs(
            body, name=f'block_{i}', dim=embed_dim, num_heads=num_heads,
            mlp_ratio=4, qk_scale=qk_scale, batch_size=batch_size)
        all_outputs.update(block_outputs)

    # Final norm
    qconfig_norm = layers.get_qconfig('qconfig_norm')
    norm_bias = relay.var('norm_bias', shape=[embed_dim], dtype='int32')
    norm = layers.quantized_layernorm(body, norm_bias)
    all_outputs['final_norm'] = norm

    # CLS token extraction
    body = relay.split(norm, 197, axis=1)
    body = relay.squeeze(body[0], axis=[1])

    # Head
    qconfig_head = layers.get_qconfig('qconfig_head')
    req = layers.requantize(body, input_scale=qconfig_norm.output_scale,
                            output_scale=qconfig_head.input_scale,
                            out_dtype=qconfig_head.input_dtype)
    head = layers.quantized_dense(data=req, name='head',
                                   input_scale=qconfig_head.input_scale,
                                   kernel_scale=qconfig_head.kernel_scale,
                                   units=1000, kernel_shape=(1000, embed_dim),
                                   kernel_dtype='int8', add_bias=True)
    all_outputs['head'] = head

    # Dequantize + softmax (final output)
    net = layers.dequantize(head, input_scale=qconfig_head.output_scale)
    net = relay.nn.softmax(data=net)

    return net, all_outputs


def main():
    args = parse_args()

    print("=" * 60)
    print("TVM Pattern Extraction")
    print("=" * 60)
    print(f"TVM version: {tvm.__version__}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test image: {args.test_image}")
    print(f"Output: {args.output}")
    print(f"Target: {args.target}")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # Step 1: Load checkpoint and QConfig
    # ==========================================================
    print("\n[1/6] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_state = ckpt['model'] if 'model' in ckpt else ckpt
    print(f"  {len(model_state)} keys loaded")

    print("\n[2/6] Loading QConfig...")
    convert_model.load_qconfig(model_state, args.depth)
    qconfig_dict = QuantizeContext.qconfig_dict
    print(f"  {len(qconfig_dict)} qconfigs loaded")

    # ==========================================================
    # Step 2: Save weights/biases
    # ==========================================================
    print("\n[3/6] Extracting weights/biases...")
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    params_dir = output_dir / "tvm_params"
    params_dir.mkdir(parents=True, exist_ok=True)

    convert_model.save_params(model_state, args.depth, str(params_dir))
    pretrained_params = np.load(params_dir / "params.npy", allow_pickle=True)[()]

    weight_info = {}
    for name, arr in pretrained_params.items():
        safe_name = name.replace('.', '_')
        save_npy_and_hex(arr, weights_dir / safe_name,
                        'int8' if arr.dtype == np.int8 else 'int32' if arr.dtype == np.int32 else None)
        weight_info[safe_name] = {
            'shape': list(arr.shape), 'dtype': str(arr.dtype),
            'min': float(arr.min()), 'max': float(arr.max()),
        }
    print(f"  {len(weight_info)} weight tensors saved")

    # ==========================================================
    # Step 3: Save scaling factors
    # ==========================================================
    print("\n[4/6] Extracting scaling factors...")
    scales_dir = output_dir / "scales"
    scales_dir.mkdir(parents=True, exist_ok=True)

    scale_info = {}
    for qname, qcfg in qconfig_dict.items():
        for field in ['input_scale', 'output_scale', 'kernel_scale']:
            val = getattr(qcfg, field, None)
            if val is not None:
                scale_val = np.atleast_1d(np.array(val, dtype=np.float64))
                key = f"{qname}_{field}".replace('.', '_')
                np.save(scales_dir / f"{key}_float.npy", scale_val.astype(np.float32))
                M, S = scale_to_ms(scale_val)
                np.save(scales_dir / f"{key}_M.npy", M)
                np.save(scales_dir / f"{key}_S.npy", S)
                scale_info[key] = {
                    'float_value': [float(v) for v in scale_val],
                    'M': [int(m) for m in M], 'S': [int(s) for s in S],
                }
    with open(scales_dir / "all_scales.json", 'w') as f:
        json.dump(scale_info, f, indent=2)
    print(f"  {len(scale_info)} scaling factors saved")

    # ==========================================================
    # Step 4: Preprocess input
    # ==========================================================
    print("\n[5/6] Preprocessing input...")
    input_float, original_image = load_and_preprocess_image(args.test_image)
    input_scale = qconfig_dict['qconfig_embed_conv'].input_scale
    input_int = np.clip(np.round(input_float / input_scale), -128, 127).astype('int8')

    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    original_image.save(input_dir / "test_image.png")
    np.save(input_dir / "input_float.npy", input_float.astype(np.float32))
    save_npy_and_hex(input_int, input_dir / "input_int8", 'int8')
    print(f"  Input: {input_int.shape}, scale={input_scale:.10f}")

    # ==========================================================
    # Step 5: Build and run TVM model, extract each layer separately
    # ==========================================================
    print("\n[6/6] Building TVM models and extracting golden outputs...")
    golden_dir = output_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    # Build model with all intermediate outputs tracked
    final_net, all_outputs = build_model_with_outputs(
        data_shape=list(input_int.shape),
        model_name=args.model_name,
        depth=args.depth)

    golden_info = {}
    target = args.target
    input_data = input_int.copy()

    # Run full model first for final output
    print("  Building full model...")
    full_func = relay.Function(relay.analysis.free_vars(final_net), final_net)
    full_mod = tvm.IRModule.from_expr(full_func)
    full_mod = relay.transform.InferType()(full_mod)

    # Get param shapes for initialization
    from models.utils import create_workload, QuantizeInitializer
    _, init_params = create_workload(full_func, QuantizeInitializer())

    # Override with pretrained params
    run_params = {}
    for k, v in init_params.items():
        if k in pretrained_params:
            run_params[k] = tvm.nd.array(pretrained_params[k])
        else:
            run_params[k] = v

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(full_mod, target=target, params=run_params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input('data', input_data)
    runtime.run()

    final_output = runtime.get_output(0).numpy()
    np.save(golden_dir / "final_output.npy", final_output)
    pred_class = int(final_output.argmax())
    print(f"  Final output predicted class: {pred_class}")

    # Now build separate models for each intermediate output
    print(f"  Extracting {len(all_outputs)} intermediate outputs...")
    for layer_name, layer_expr in all_outputs.items():
        try:
            # Build a mini-model that outputs this specific layer
            layer_func = relay.Function(relay.analysis.free_vars(layer_expr), layer_expr)
            layer_mod = tvm.IRModule.from_expr(layer_func)
            layer_mod = relay.transform.InferType()(layer_mod)

            # Get params needed for this sub-model
            layer_params = {}
            shape_dict = {v.name_hint: v.checked_type for v in layer_mod["main"].params}
            for k in shape_dict:
                if k == "data":
                    continue
                if k in pretrained_params:
                    layer_params[k] = tvm.nd.array(pretrained_params[k])
                elif k in init_params:
                    layer_params[k] = init_params[k]

            with tvm.transform.PassContext(opt_level=3):
                layer_lib = relay.build(layer_mod, target=target, params=layer_params)

            layer_runtime = tvm.contrib.graph_executor.GraphModule(
                layer_lib["default"](dev))
            layer_runtime.set_input('data', input_data)
            layer_runtime.run()

            data = layer_runtime.get_output(0).numpy()
            safe_name = layer_name.replace('.', '_')

            # Save output
            np.save(golden_dir / f"{safe_name}.npy", data)
            if data.dtype in [np.int8, np.int32, np.int64]:
                dtype_hint = 'int8' if data.dtype == np.int8 else 'int32'
                save_npy_and_hex(data, golden_dir / f"{safe_name}_int", dtype_hint)

            golden_info[safe_name] = {
                'shape': list(data.shape), 'dtype': str(data.dtype),
                'min': float(data.min()), 'max': float(data.max()),
            }
            print(f"    ✓ {layer_name}: {data.shape} ({data.dtype})")
        except Exception as e:
            print(f"    ✗ {layer_name}: {e}")

    # Save golden info
    with open(golden_dir / "layer_info.json", 'w') as f:
        json.dump(golden_info, f, indent=2)
    print(f"  Total: {len(golden_info)} golden outputs extracted")

    # ==========================================================
    # Save config
    # ==========================================================
    config = {
        'model': args.model_name, 'checkpoint': args.checkpoint,
        'test_image': args.test_image, 'tvm_version': tvm.__version__,
        'target': target, 'extraction_time': datetime.now().isoformat(),
        'input_shape': list(input_int.shape), 'input_scale': float(input_scale),
        'num_weights': len(weight_info), 'num_scales': len(scale_info),
        'num_golden': len(golden_info), 'predicted_class': pred_class,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("TVM Pattern Extraction Complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"  weights/     : {len(weight_info)} tensors")
    print(f"  scales/      : {len(scale_info)} factors")
    print(f"  golden/      : {len(golden_info)} intermediate outputs")
    print(f"  Predicted class: {pred_class}")
    print("=" * 60)


if __name__ == "__main__":
    main()
