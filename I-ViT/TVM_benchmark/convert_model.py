
import torch
import numpy as np
import argparse

import os

from models.layers import QConfig, QuantizeContext


def save_params(model, depth, save_path):
    ## weight and bias (conv and dense)
    params = {}
    for (key, tensor) in model.items():
        if 'weight_integer' in key:
            print(key)
            params[key] = tensor.cpu().numpy().astype('int8')
        if 'bias_integer' in key:
            print(key)
            params[key] = np.nan_to_num(tensor.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).clip(-2147483648, 2147483647).astype('int32')

    renamed_params = {}
    renamed_params['embed_conv_weight'] = params['patch_embed.proj.weight_integer']
    renamed_params['embed_conv_bias'] = params['patch_embed.proj.bias_integer'].reshape(1, -1, 1, 1)

    for i in range(depth):
        for key in ['weight_integer', 'bias_integer']:
            old_name = 'blocks.%d.attn.qkv.' % (i) + key
            new_name = 'block_%d_attn_qkv_' % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = 'blocks.%d.attn.proj.' % (i) + key
            new_name = 'block_%d_attn_proj_' % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = 'blocks.%d.mlp.fc1.' % (i) + key
            new_name = 'block_%d_mlp_fc1_' % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = 'blocks.%d.mlp.fc2.' % (i) + key
            new_name = 'block_%d_mlp_fc2_' % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

    renamed_params['head_weight'] = params['head.weight_integer']
    renamed_params['head_bias'] = params['head.bias_integer']

    ## norm
    for i in range(depth):
        for key in ['bias_integer']:
            old_name = 'blocks.%d.norm1.' % (i) + key
            new_name = 'block_%d_norm1_' % (i) + key[:-8]
            renamed_params[new_name] = np.nan_to_num(model[old_name].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype('int64')

            old_name = 'blocks.%d.norm2.' % (i) + key
            new_name = 'block_%d_norm2_' % (i) + key[:-8]
            renamed_params[new_name] = np.nan_to_num(model[old_name].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype('int64')

    renamed_params['norm_bias'] = np.nan_to_num(model['norm.bias_integer'].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype('int64')


    ## other params
    renamed_params['cls_token_weight'] = model['cls_token'].cpu().numpy()
    renamed_params['pos_embed_weight'] = model['pos_embed'].cpu().numpy()

    np.save(os.path.join(save_path, 'params.npy'), renamed_params)


def load_qconfig(model, depth, calib_scales=None, calib_scales_file=None):
    """
    Load quantization configs from model checkpoint.
    
    Args:
        model: state_dict from QAT checkpoint
        depth: number of transformer blocks
        calib_scales: optional dict of {key: float} to OVERRIDE stale checkpoint
                      scaling factors with calibration-forward-derived values.
        calib_scales_file: optional path to .npy file containing calibrated scales.
                           If provided, loads all scales from this file.
                           Default: looks for 'calibrated_scales.npy' in current dir.
    """
    params = {}
    for (key, tensor) in model.items():
        if 'scaling_factor' in key:
            #print(key)
            tensor_np = tensor.cpu().numpy().reshape((-1))
            params[key] = tensor_np
        if "act_scaling_factor" in key and np.ndim(tensor_np) == 1:
            tensor_np = tensor_np[0]
            params[key] = tensor_np

    # Auto-load calibrated scales if file exists
    if calib_scales_file is None:
        calib_scales_file = os.path.join(os.path.dirname(__file__), 'calibrated_scales.npy')
    
    if calib_scales is None and os.path.exists(calib_scales_file):
        print(f"[INFO] Loading calibrated scales from {calib_scales_file}")
        calib_scales = np.load(calib_scales_file, allow_pickle=True)[()]
        print(f"[INFO] Loaded {len(calib_scales)} calibrated scaling factors")

    # Override with calibration-derived scales if provided
    if calib_scales:
        for k, v in calib_scales.items():
            params[k] = np.array(v).reshape((-1))
            if np.ndim(params[k]) == 1 and params[k].size == 1:
                params[k] = float(params[k][0])

    QuantizeContext.qconfig_dict['qconfig_pos'] = QConfig(output_scale=params['qact_pos.act_scaling_factor'])
    QuantizeContext.qconfig_dict['qconfig_addpos'] = QConfig(input_scale=params['patch_embed.qact.act_scaling_factor'], input_dtype='int16', output_scale=params['qact1.act_scaling_factor'])
    ## Embed
    conv_input_scale = params['qact_input.act_scaling_factor']
    conv_kernel_scale = params['patch_embed.proj.conv_scaling_factor']
    conv_output_scale = conv_input_scale * conv_kernel_scale
    QuantizeContext.qconfig_dict['qconfig_embed_conv'] = \
            QConfig(input_scale=conv_input_scale, kernel_scale=conv_kernel_scale, output_scale=conv_output_scale)

    for i in range(depth):
        # The block input data scale (from global qact1 or prev block's qact4)
        shortcut_scale = params['qact1.act_scaling_factor'] if i == 0 else params['blocks.%d.qact4.act_scaling_factor' % (i-1)]
        
        # norm1 input scale is the block input scale.
        # This will natively be used as add1's rhs_scale via qconfig0.input_scale
        output_scale = params['blocks.%d.norm1.norm_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_norm1' % (i)] = QConfig(input_scale=shortcut_scale, output_scale=output_scale)

        # qkv input scale is blocks.i.qact1 (the QuantAct layer BETWEEN norm1 and attn)
        qkv_input_scale = params['blocks.%d.qact1.act_scaling_factor' % (i)]
        kernel_scale = params['blocks.%d.attn.qkv.fc_scaling_factor' % (i)]
        output_scale = qkv_input_scale * kernel_scale
        QuantizeContext.qconfig_dict['block_%d_qconfig_qkv' % (i)] = QConfig(input_scale=qkv_input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.attn.qact1.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.attn.matmul_1.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_matmul_1' % (i)] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.attn.qact_attn1.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.attn.int_softmax.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_softmax' % (i)] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.attn.int_softmax.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.attn.matmul_2.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_matmul_2' % (i)] = QConfig(input_scale=input_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.attn.qact2.act_scaling_factor' % (i)]
        kernel_scale = params['blocks.%d.attn.proj.fc_scaling_factor' % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict['block_%d_qconfig_proj' % (i)] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.attn.qact3.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.qact2.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_add1' % (i)] = QConfig(input_scale=input_scale, input_dtype='int16', output_scale=output_scale) 

        input_scale = params['blocks.%d.qact2.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.norm2.norm_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_norm2' % (i)] = QConfig(input_scale=input_scale, output_scale=output_scale)

        # qconfig_shortcut2: add2 rhs scale = add1 output = block qact2 output
        QuantizeContext.qconfig_dict['block_%d_qconfig_shortcut2' % (i)] = QConfig(input_scale=input_scale)

        input_scale = params['blocks.%d.qact3.act_scaling_factor' % (i)]
        kernel_scale = params['blocks.%d.mlp.fc1.fc_scaling_factor' % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict['block_%d_qconfig_fc1' % (i)] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.mlp.qact_gelu.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.mlp.act.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_gelu' % (i)] = QConfig(input_scale=input_scale, output_scale=output_scale, input_dtype='int8')

        input_scale = params['blocks.%d.mlp.qact1.act_scaling_factor' % (i)]
        kernel_scale = params['blocks.%d.mlp.fc2.fc_scaling_factor' % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict['block_%d_qconfig_fc2' % (i)] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)

        input_scale = params['blocks.%d.mlp.qact2.act_scaling_factor' % (i)]
        output_scale = params['blocks.%d.qact4.act_scaling_factor' % (i)]
        QuantizeContext.qconfig_dict['block_%d_qconfig_add2' % (i)] = QConfig(input_scale=input_scale, input_dtype='int16', output_scale=output_scale)

    # BUG FIX: input to final norm is output of the last block's residual add (qact4),
    # NOT blocks.11.mlp.qact2 (which was the stale for-loop variable).
    # blocks.11.mlp.qact2 feeds into qact4 (add2 residual), whose output goes to norm.
    final_block_output_scale = params['blocks.%d.qact4.act_scaling_factor' % (depth - 1)]
    output_scale = params['norm.norm_scaling_factor']
    QuantizeContext.qconfig_dict['qconfig_norm'] = QConfig(input_scale=final_block_output_scale, output_scale=output_scale)


    # NOTE: 'qact2.act_scaling_factor' in the checkpoint is the scale BEFORE norm,
    # which gets updated during forward. The actual head input scale after norm + qact2
    # must be obtained by running a calibration forward.
    # However, 'qact2.act_scaling_factor' is still the best static approximation.
    input_scale = params['qact2.act_scaling_factor']
    kernel_scale = params['head.fc_scaling_factor']
    output_scale = input_scale * kernel_scale
    QuantizeContext.qconfig_dict['qconfig_head'] = QConfig(input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I-ViT convert model',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', default='',
                        help='saved checkpoint path in QAT (checkpoint.pth.tar)')
    parser.add_argument('--params-path', default='',
                        help='Saved parameters directory')
    parser.add_argument('--depth', default=12,
                        help='Depth of ViT')

    args = parser.parse_args()
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model = checkpoint['state_dict']
    else:
        model = checkpoint
    # print(model.keys())
    
    save_params(model, int(args.depth), args.params_path)
