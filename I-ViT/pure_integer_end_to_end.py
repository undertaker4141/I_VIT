"""
完全純整數端到端推論 - 不依賴 PyTorch
==================================================================
目標：使用純 NumPy 實現完整的 ViT 推論，所有運算都是純整數

關鍵要求：
1. 運算圖必須是純整數的
2. 不依賴 PyTorch 進行推論（只用於載入模型和提取權重）
3. 所有中間值都是整數（int8, int16, int32, int64）
4. Scaling factors 只用於計算，不參與整數運算

運算流程：
Input (PyTorch) -> 整數 -> Transformer Blocks (純整數) -> 整數 -> Head (PyTorch) -> Output
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmodel_rtl_reference'))

from models.vit_quant import deit_tiny_patch16_224
from pure_numpy_cmodel import (
    int_layer_norm,
    int_dense,
    int_gelu,
    int_softmax,
    quantize_to_int,
    dequantize_to_float,
    requantize,
    quant_act_residual
)


def extract_scaling_factor(sf):
    """提取 scaling factor（處理 tensor 和 scalar）"""
    if isinstance(sf, torch.Tensor):
        if sf.numel() == 1:
            return sf.item()
        else:
            return sf.cpu().numpy()
    else:
        return sf


def extract_weights_and_scales(model, image_tensor):
    """從 PyTorch 模型提取整數權重和 scaling factors（執行一次推論以獲取 runtime scaling factors）"""
    print("\n[1/4] 提取整數權重和 scaling factors...")
    
    # 先執行一次推論，讓模型計算 runtime scaling factors
    print("  執行推論以獲取 runtime scaling factors...")
    with torch.no_grad():
        _ = model(image_tensor)
    
    weights = {}
    
    # 對每個 block 提取
    for i, block in enumerate(model.blocks):
        block_weights = {}
        
        # Norm1
        block_weights['norm1_bias_int'] = block.norm1.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm1_weight'] = block.norm1.weight.data.cpu().numpy()
        block_weights['norm1_bias'] = block.norm1.bias.data.cpu().numpy()
        
        # QuantAct1 (Norm1 後) - 使用 runtime scaling factor
        block_weights['qact1_sf'] = extract_scaling_factor(block.qact1.act_scaling_factor)
        
        # Attention - QKV
        block_weights['qkv_weight_int'] = block.attn.qkv.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['qkv_bias_int'] = block.attn.qkv.bias_integer.cpu().numpy().astype(np.int32) if block.attn.qkv.bias is not None else None
        block_weights['qkv_fc_scaling_factor'] = extract_scaling_factor(block.attn.qkv.fc_scaling_factor)
        
        # Attention 內部的 QuantAct - 使用 runtime scaling factor
        block_weights['attn_qact1_sf'] = extract_scaling_factor(block.attn.qact1.act_scaling_factor)
        block_weights['attn_qact_attn1_sf'] = extract_scaling_factor(block.attn.qact_attn1.act_scaling_factor)
        block_weights['attn_matmul_1_sf'] = extract_scaling_factor(block.attn.matmul_1.act_scaling_factor)
        block_weights['attn_qact2_sf'] = extract_scaling_factor(block.attn.qact2.act_scaling_factor)
        block_weights['attn_qact3_sf'] = extract_scaling_factor(block.attn.qact3.act_scaling_factor)
        
        # Attention - Proj
        block_weights['proj_weight_int'] = block.attn.proj.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['proj_bias_int'] = block.attn.proj.bias_integer.cpu().numpy().astype(np.int32) if block.attn.proj.bias is not None else None
        block_weights['proj_fc_scaling_factor'] = extract_scaling_factor(block.attn.proj.fc_scaling_factor)
        
        # Attention - 參數
        block_weights['num_heads'] = block.attn.num_heads
        block_weights['scale'] = block.attn.scale
        
        # QuantAct2 (Residual 1 後) - 使用 runtime scaling factor
        block_weights['qact2_sf'] = extract_scaling_factor(block.qact2.act_scaling_factor)
        
        # Norm2
        block_weights['norm2_bias_int'] = block.norm2.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm2_weight'] = block.norm2.weight.data.cpu().numpy()
        block_weights['norm2_bias'] = block.norm2.bias.data.cpu().numpy()
        
        # QuantAct3 (Norm2 後) - 使用 runtime scaling factor
        block_weights['qact3_sf'] = extract_scaling_factor(block.qact3.act_scaling_factor)
        
        # MLP - FC1
        block_weights['fc1_weight_int'] = block.mlp.fc1.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['fc1_bias_int'] = block.mlp.fc1.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc1.bias is not None else None
        block_weights['fc1_fc_scaling_factor'] = extract_scaling_factor(block.mlp.fc1.fc_scaling_factor)
        
        # MLP 內部的 QuantAct - 使用 runtime scaling factor
        block_weights['mlp_qact1_sf'] = extract_scaling_factor(block.mlp.qact1.act_scaling_factor)
        block_weights['mlp_qact_gelu_sf'] = extract_scaling_factor(block.mlp.qact_gelu.act_scaling_factor)
        block_weights['mlp_qact2_sf'] = extract_scaling_factor(block.mlp.qact2.act_scaling_factor)
        
        # MLP - FC2
        block_weights['fc2_weight_int'] = block.mlp.fc2.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['fc2_bias_int'] = block.mlp.fc2.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc2.bias is not None else None
        block_weights['fc2_fc_scaling_factor'] = extract_scaling_factor(block.mlp.fc2.fc_scaling_factor)
        
        # QuantAct4 (Residual 2 後) - 使用 runtime scaling factor
        block_weights['qact4_sf'] = extract_scaling_factor(block.qact4.act_scaling_factor)
        
        weights[f'block_{i}'] = block_weights
    
    # Final norm
    weights['final_norm'] = {
        'bias_int': model.norm.bias_integer.cpu().numpy().astype(np.float64),
        'weight': model.norm.weight.data.cpu().numpy(),
        'bias': model.norm.bias.data.cpu().numpy()
    }
    
    print(f"  ✓ 已提取 {len(model.blocks)} 個 blocks 的權重和 scaling factors")
    print(f"  ✓ 已提取所有 QuantAct runtime scaling factors")
    
    return weights


def pure_integer_attention(x_int8, x_sf, block_weights):
    """
    純整數 Attention（使用固定的 QuantAct scaling factors）
    
    參數:
        x_int8: 整數輸入 [B, N, C] (int8)
        x_sf: 輸入 scaling factor (scalar)
        block_weights: Block 權重字典
    
    返回:
        output_int16: 整數輸出 [B, N, C] (int16)
        output_sf: 輸出 scaling factor (scalar)
    """
    B, N, C = x_int8.shape
    num_heads = block_weights['num_heads']
    head_dim = C // num_heads
    scale = block_weights['scale']
    
    # 使用固定的 QuantAct scaling factors
    attn_qact1_sf = block_weights['attn_qact1_sf']
    attn_qact_attn1_sf = block_weights['attn_qact_attn1_sf']
    attn_matmul_1_sf = block_weights['attn_matmul_1_sf']
    attn_qact2_sf = block_weights['attn_qact2_sf']
    attn_qact3_sf = block_weights['attn_qact3_sf']
    
    # Step 1: QKV projection (int8 @ int8 -> int32)
    qkv_int32 = int_dense(x_int8, block_weights['qkv_weight_int'], block_weights['qkv_bias_int'])
    
    # Step 2: 計算 QKV scaling factor 並 requantize (保持 per-channel)
    qkv_fc_sf = block_weights['qkv_fc_scaling_factor']
    qkv_sf = x_sf * qkv_fc_sf  # per-channel
    
    # 使用 attn_qact1_sf 作為目標 scaling factor
    qkv_int8 = requantize(qkv_int32, qkv_sf, attn_qact1_sf, output_bits=8)
    
    # Step 3: Reshape to [B, N, 3, num_heads, head_dim]
    qkv_int8 = qkv_int8.reshape(B, N, 3, num_heads, head_dim)
    
    # Step 4: Permute to [3, B, num_heads, N, head_dim]
    qkv_int8 = qkv_int8.transpose(2, 0, 3, 1, 4)
    
    # Step 5: Split Q, K, V
    q_int8, k_int8, v_int8 = qkv_int8[0], qkv_int8[1], qkv_int8[2]
    
    # Step 6: Q @ K^T (int8 @ int8 -> int32)
    attn_int32 = np.matmul(q_int8.astype(np.int32), k_int8.transpose(0, 1, 3, 2).astype(np.int32))
    
    # Step 7: 計算 attn scaling factor 並乘以 scale
    # PyTorch: attn = attn * self.scale, act_scaling_factor = act_scaling_factor * self.scale
    attn_sf = attn_qact1_sf * attn_qact1_sf * scale
    
    # 使用 attn_qact_attn1_sf 作為目標 scaling factor
    attn_int8 = requantize(attn_int32, attn_sf, attn_qact_attn1_sf, output_bits=8)
    
    # Step 8: Softmax (使用 output_bit=16，與 PyTorch 一致)
    attn_softmax_int32 = int_softmax(attn_int8.astype(np.int32), attn_qact_attn1_sf, output_bit=16, n=15)
    
    # PyTorch IntSoftmax 返回固定的 scaling factor: 1 / 2^(output_bit-1)
    softmax_output_sf = 1.0 / (2 ** (16 - 1))  # = 1/32768 = 0.000031
    
    # 注意：PyTorch 的 QuantAct2 直接作用在 Softmax 輸出上，不需要中間的 requantize
    # 我們直接使用 Softmax 的 int32 輸出（範圍 [0, 32767]）
    # 並將其視為具有 softmax_output_sf 的整數
    
    # Step 9: Attn @ V (int32 @ int8 -> int32)
    # 注意：Softmax 輸出是 int32，範圍 [0, 32767]
    attn_v_int32 = np.matmul(attn_softmax_int32, v_int8.astype(np.int32))
    
    # Step 10: Transpose to [B, N, num_heads, head_dim]
    attn_v_int32 = attn_v_int32.transpose(0, 2, 1, 3)
    
    # Step 11: Reshape to [B, N, C]
    attn_v_int32 = attn_v_int32.reshape(B, N, C)
    
    # Step 12: 計算 attn_v scaling factor 並 requantize
    # attn_v 的 SF = softmax_output_sf * v_sf
    attn_v_sf = softmax_output_sf * attn_qact1_sf
    
    # 使用 attn_qact2_sf 作為目標 scaling factor
    attn_v_int8 = requantize(attn_v_int32, attn_v_sf, attn_qact2_sf, output_bits=8)
    
    # Step 13: Projection (int8 @ int8 -> int32)
    output_int32 = int_dense(attn_v_int8, block_weights['proj_weight_int'], block_weights['proj_bias_int'])
    
    # Step 14: 計算 output scaling factor 並 requantize (保持 per-channel)
    proj_fc_sf = block_weights['proj_fc_scaling_factor']
    output_sf = attn_qact2_sf * proj_fc_sf  # per-channel
    
    # 使用 attn_qact3_sf 作為目標 scaling factor
    output_int16 = requantize(output_int32, output_sf, attn_qact3_sf, output_bits=16)
    
    return output_int16, attn_qact3_sf


def pure_integer_mlp(x_int8, x_sf, block_weights):
    """
    純整數 MLP（使用固定的 QuantAct scaling factors）
    
    參數:
        x_int8: 整數輸入 [B, N, C] (int8)
        x_sf: 輸入 scaling factor (scalar)
        block_weights: Block 權重字典
    
    返回:
        output_int16: 整數輸出 [B, N, C] (int16)
        output_sf: 輸出 scaling factor (scalar)
    """
    # 使用固定的 QuantAct scaling factors
    mlp_qact1_sf = block_weights['mlp_qact1_sf']
    mlp_qact_gelu_sf = block_weights['mlp_qact_gelu_sf']
    mlp_qact2_sf = block_weights['mlp_qact2_sf']
    
    # Step 1: FC1 (int8 @ int8 -> int32)
    fc1_int32 = int_dense(x_int8, block_weights['fc1_weight_int'], block_weights['fc1_bias_int'])
    
    # Step 2: 計算 FC1 scaling factor 並 requantize (保持 per-channel)
    fc1_fc_sf = block_weights['fc1_fc_scaling_factor']
    fc1_sf = x_sf * fc1_fc_sf  # per-channel
    
    # 使用 mlp_qact1_sf 作為目標 scaling factor
    fc1_int8 = requantize(fc1_int32, fc1_sf, mlp_qact1_sf, output_bits=8)
    
    # Step 3: GELU
    gelu_int32 = int_gelu(fc1_int8.astype(np.int32), mlp_qact1_sf, output_bit=8, n=23)
    
    # PyTorch IntGELU 的輸出 SF 計算：
    # output_sf = input_sf * sigmoid_sf
    # 其中 sigmoid_sf = 1 / 2^(output_bit-1) = 1/128
    sigmoid_sf = 1.0 / (2 ** (8 - 1))
    gelu_output_sf = mlp_qact1_sf * sigmoid_sf
    
    # 使用 mlp_qact_gelu_sf 作為目標 scaling factor
    gelu_int8 = requantize(gelu_int32, gelu_output_sf, mlp_qact_gelu_sf, output_bits=8)
    
    # Step 4: FC2 (int8 @ int8 -> int32)
    fc2_int32 = int_dense(gelu_int8, block_weights['fc2_weight_int'], block_weights['fc2_bias_int'])
    
    # Step 5: 計算 FC2 scaling factor 並 requantize (保持 per-channel)
    fc2_fc_sf = block_weights['fc2_fc_scaling_factor']
    fc2_sf = mlp_qact_gelu_sf * fc2_fc_sf  # per-channel
    
    # 使用 mlp_qact2_sf 作為目標 scaling factor
    output_int16 = requantize(fc2_int32, fc2_sf, mlp_qact2_sf, output_bits=16)
    
    return output_int16, mlp_qact2_sf


def pure_integer_transformer_block(x_int16, x_sf, block_weights, dim_sqrt, block_idx):
    """
    純整數 Transformer Block（使用固定的 QuantAct scaling factors）
    
    參數:
        x_int16: 整數輸入 [B, N, C] (int16)
        x_sf: 輸入 scaling factor (scalar)
        block_weights: Block 權重字典
        dim_sqrt: sqrt(C)
        block_idx: Block 索引
    
    返回:
        output_int16: 整數輸出 [B, N, C] (int16)
        output_sf: 輸出 scaling factor (scalar)
    """
    # 保存輸入用於殘差連接 1
    x_input_for_residual1 = x_int16.copy()
    x_sf_for_residual1 = x_sf
    
    # ============================================================
    # Norm1 (純整數)
    # ============================================================
    x_norm1_int32 = int_layer_norm(
        x_int16.astype(np.float64),
        block_weights['norm1_bias_int'],
        block_weights['norm1_weight'],
        block_weights['norm1_bias'],
        dim_sqrt
    )
    
    # 計算 norm1 scaling factor (per-channel)
    base_scaling_factor = dim_sqrt / (2 ** 30)
    norm1_sf = base_scaling_factor * block_weights['norm1_weight']  # per-channel
    
    # 使用 qact1_sf 作為目標 scaling factor
    x_norm1_int8 = requantize(x_norm1_int32, norm1_sf, block_weights['qact1_sf'], output_bits=8)
    
    # ============================================================
    # Attention (純整數)
    # ============================================================
    attn_output_int16, attn_output_sf = pure_integer_attention(
        x_norm1_int8, block_weights['qact1_sf'], block_weights
    )
    
    # ============================================================
    # Residual 1 (純整數)
    # ============================================================
    # 使用 qact2_sf 作為目標 scaling factor
    x_int16 = quant_act_residual(
        attn_output_int16, attn_output_sf,
        x_input_for_residual1, x_sf_for_residual1,
        block_weights['qact2_sf'], output_bits=16
    )
    x_sf = block_weights['qact2_sf']
    
    # 保存輸入用於殘差連接 2
    x_input_for_residual2 = x_int16.copy()
    x_sf_for_residual2 = x_sf
    
    # ============================================================
    # Norm2 (純整數)
    # ============================================================
    x_norm2_int32 = int_layer_norm(
        x_int16.astype(np.float64),
        block_weights['norm2_bias_int'],
        block_weights['norm2_weight'],
        block_weights['norm2_bias'],
        dim_sqrt
    )
    
    # 計算 norm2 scaling factor (per-channel)
    norm2_sf = base_scaling_factor * block_weights['norm2_weight']  # per-channel
    
    # 使用 qact3_sf 作為目標 scaling factor
    x_norm2_int8 = requantize(x_norm2_int32, norm2_sf, block_weights['qact3_sf'], output_bits=8)
    
    # ============================================================
    # MLP (純整數)
    # ============================================================
    mlp_output_int16, mlp_output_sf = pure_integer_mlp(
        x_norm2_int8, block_weights['qact3_sf'], block_weights
    )
    
    # ============================================================
    # Residual 2 (純整數)
    # ============================================================
    # 使用 qact4_sf 作為目標 scaling factor
    x_int16 = quant_act_residual(
        mlp_output_int16, mlp_output_sf,
        x_input_for_residual2, x_sf_for_residual2,
        block_weights['qact4_sf'], output_bits=16
    )
    x_sf = block_weights['qact4_sf']
    
    return x_int16, x_sf


def pure_integer_inference(model, image_tensor, weights):
    """
    純整數推論（Transformer Blocks 部分）
    
    參數:
        model: PyTorch 模型（只用於 Input Processing 和 Classification Head）
        image_tensor: 輸入圖片 tensor
        weights: 提取的權重和 scaling factors
    
    返回:
        pred_class: 預測類別
        pred_prob: 預測概率
        logits: Logits
    """
    print("\n" + "="*80)
    print("開始純整數推論")
    print("="*80)
    
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # ============================================================
        # Stage 1: Input Processing (使用 PyTorch，提取整數)
        # ============================================================
        print(f"\n[Stage 1] Input Processing (PyTorch)")
        
        start_time = time.time()
        
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        # 轉換為整數
        x_np = x.cpu().numpy()
        x_scaling_factor_np = extract_scaling_factor(x_scaling_factor)
        x_int16 = quantize_to_int(x_np, x_scaling_factor_np, bits=16)
        
        print(f"  輸出: shape={x_int16.shape}, dtype={x_int16.dtype}")
        print(f"  整數範圍: [{x_int16.min()}, {x_int16.max()}]")
        print(f"  Scaling factor: {x_scaling_factor_np:.6f}")
        print(f"  時間: {(time.time() - start_time)*1000:.2f} ms")
        
        # ============================================================
        # Stage 2: Transformer Blocks (純整數)
        # ============================================================
        print(f"\n[Stage 2] Transformer Blocks (純整數)")
        
        dim_sqrt = np.sqrt(x_int16.shape[-1])
        x_sf = x_scaling_factor_np
        
        for block_idx in range(len(model.blocks)):
            block_start = time.time()
            
            block_weights = weights[f'block_{block_idx}']
            
            # 純整數 Transformer Block
            x_int16, x_sf = pure_integer_transformer_block(
                x_int16, x_sf, block_weights, dim_sqrt, block_idx
            )
            
            block_time = (time.time() - block_start) * 1000
            
            if block_idx % 3 == 0 or block_idx == len(model.blocks) - 1:
                print(f"  Block {block_idx}: {block_time:.2f} ms, int16 range=[{x_int16.min()}, {x_int16.max()}], sf={x_sf:.6f}")
        
        print(f"  ✓ 所有 blocks 完成（純整數運算）")
        
        # ============================================================
        # Stage 3: Classification Head (使用 PyTorch)
        # ============================================================
        print(f"\n[Stage 3] Classification Head (PyTorch)")
        
        start_time = time.time()
        
        # 轉回 PyTorch tensor
        x = torch.from_numpy(dequantize_to_float(x_int16, x_sf)).float()
        x_scaling_factor = torch.tensor(x_sf).float()
        
        # Final norm
        x, x_scaling_factor = model.norm(x, x_scaling_factor)
        
        # CLS token
        x = x[:, 0]
        x, x_scaling_factor = model.qact2(x, x_scaling_factor)
        
        # Head
        x, x_scaling_factor = model.head(x, x_scaling_factor)
        
        print(f"  時間: {(time.time() - start_time)*1000:.2f} ms")
        
        # 預測
        logits = x.cpu().numpy()[0]
        pred_class = np.argmax(logits)
        pred_prob = np.exp(logits) / np.sum(np.exp(logits))
        
        return pred_class, pred_prob[pred_class], logits


def main():
    """主函數"""
    print("="*80)
    print("完全純整數端到端推論")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    
    # 載入模型
    print("\n[1/5] 載入模型...")
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
    print("  ✓ 模型已載入")
    
    # 載入圖片
    print("\n[2/5] 載入圖片...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print("  ✓ 圖片已載入")
    
    # 提取權重和 scaling factors（會執行一次推論以獲取 runtime scaling factors）
    weights = extract_weights_and_scales(model, image_tensor)
    
    # 執行純整數推論
    print("\n[3/5] 執行純整數推論...")
    total_start = time.time()
    pred_class, pred_prob, logits = pure_integer_inference(model, image_tensor, weights)
    total_time = (time.time() - total_start) * 1000
    
    # 結果
    print("\n" + "="*80)
    print("推論結果")
    print("="*80)
    print(f"預測類別: {pred_class}")
    print(f"預測概率: {pred_prob:.4f}")
    print(f"Top-5 類別: {np.argsort(logits)[-5:][::-1]}")
    print(f"總時間: {total_time:.2f} ms")
    
    # 與 PyTorch 比較
    print("\n[4/5] 與 PyTorch 比較...")
    
    with torch.no_grad():
        pytorch_output = model(image_tensor)
        pytorch_logits = pytorch_output.cpu().numpy()[0]
        pytorch_pred_class = np.argmax(pytorch_logits)
        pytorch_pred_prob = np.exp(pytorch_logits) / np.sum(np.exp(pytorch_logits))
    
    print(f"  PyTorch 預測類別: {pytorch_pred_class}")
    print(f"  PyTorch 預測概率: {pytorch_pred_prob[pytorch_pred_class]:.4f}")
    
    # 比較 logits
    logits_diff = np.abs(logits - pytorch_logits)
    logits_corr = np.corrcoef(logits, pytorch_logits)[0, 1]
    
    print(f"\n  Logits 比較:")
    print(f"    最大差異: {logits_diff.max():.4f}")
    print(f"    平均差異: {logits_diff.mean():.4f}")
    print(f"    相關係數: {logits_corr:.6f}")
    
    # 判斷
    print("\n" + "="*80)
    print("測試結果")
    print("="*80)
    
    if pred_class == pytorch_pred_class:
        print("✓ 預測類別一致！")
        if pred_class == 1:
            print("✓ 預測正確（期望類別 1）")
            print(f"\n✓✓✓ 純整數端到端推論成功！✓✓✓")
            print(f"  - 運算圖: 完全純整數")
            print(f"  - LayerNorm: 純整數")
            print(f"  - Attention: 純整數")
            print(f"  - MLP: 純整數")
            print(f"  - Residual: 純整數")
            print(f"  - 預測類別: {pred_class}")
            print(f"  - 預測概率: {pred_prob:.4f}")
            print(f"  - Logits 相關係數: {logits_corr:.6f}")
            return True
        else:
            print(f"⚠ 預測類別一致但不是期望的類別 1（實際: {pred_class}）")
            return False
    else:
        print(f"✗ 預測類別不一致")
        print(f"  純整數 C-Model: {pred_class}")
        print(f"  PyTorch: {pytorch_pred_class}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
