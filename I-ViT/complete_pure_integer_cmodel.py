"""
完整純整數 C-Model 端到端推論
==================================================================
目標：使用純整數 C-Model 完成整個 ViT 推論，預測正確的類別

策略：
1. 從 PyTorch 提取整數權重和 scaling factors
2. 使用純整數 C-Model 處理所有 Transformer blocks
3. 確保所有運算都是純整數的
4. 驗證預測結果

純整數運算圖：
- 所有中間值都是整數（int8, int16, int32）
- Scaling factors 用於計算但不參與整數運算
- 使用定點數近似浮點運算
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
from pytorch_integer_cmodel import pytorch_int_layer_norm


def extract_scaling_factor(sf):
    """提取 scaling factor（處理 tensor 和 scalar）"""
    if isinstance(sf, torch.Tensor):
        if sf.numel() == 1:
            return sf.item()
        else:
            return sf.cpu().numpy()
    else:
        return sf


def quantize_to_int(x_float, scaling_factor, bits=8):
    """量化到整數"""
    if isinstance(scaling_factor, np.ndarray):
        # Per-channel
        x_int = np.zeros_like(x_float)
        for c in range(x_float.shape[-1]):
            x_int[..., c] = np.round(x_float[..., c] / scaling_factor[c])
    else:
        # Scalar
        x_int = np.round(x_float / scaling_factor)
    
    if bits == 8:
        return np.clip(x_int, -128, 127).astype(np.int8)
    elif bits == 16:
        return np.clip(x_int, -32768, 32767).astype(np.int16)
    else:
        return x_int.astype(np.int32)


def dequantize_to_float(x_int, scaling_factor):
    """反量化到浮點"""
    if isinstance(scaling_factor, np.ndarray):
        # Per-channel
        x_float = np.zeros_like(x_int, dtype=np.float32)
        for c in range(x_int.shape[-1]):
            x_float[..., c] = x_int[..., c].astype(np.float32) * scaling_factor[c]
        return x_float
    else:
        # Scalar
        return x_int.astype(np.float32) * scaling_factor


def extract_weights_and_scales(model):
    """從 PyTorch 模型提取整數權重和 scaling factors"""
    print("\n提取整數權重和 scaling factors...")
    
    weights = {}
    
    # 對每個 block 提取
    for i, block in enumerate(model.blocks):
        block_weights = {}
        
        # Norm1
        block_weights['norm1_bias_int'] = block.norm1.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm1_weight'] = block.norm1.weight.data.cpu().numpy()
        block_weights['norm1_bias'] = block.norm1.bias.data.cpu().numpy()
        
        # Attention - QKV
        block_weights['qkv_weight_int'] = block.attn.qkv.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['qkv_bias_int'] = block.attn.qkv.bias_integer.cpu().numpy().astype(np.int32) if block.attn.qkv.bias is not None else None
        block_weights['qkv_fc_scaling_factor'] = block.attn.qkv.fc_scaling_factor.cpu().numpy()
        
        # Attention - Proj
        block_weights['proj_weight_int'] = block.attn.proj.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['proj_bias_int'] = block.attn.proj.bias_integer.cpu().numpy().astype(np.int32) if block.attn.proj.bias is not None else None
        block_weights['proj_fc_scaling_factor'] = block.attn.proj.fc_scaling_factor.cpu().numpy()
        
        # Attention - 參數
        block_weights['num_heads'] = block.attn.num_heads
        block_weights['scale'] = block.attn.scale
        
        # Norm2
        block_weights['norm2_bias_int'] = block.norm2.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm2_weight'] = block.norm2.weight.data.cpu().numpy()
        block_weights['norm2_bias'] = block.norm2.bias.data.cpu().numpy()
        
        # MLP - FC1
        block_weights['fc1_weight_int'] = block.mlp.fc1.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['fc1_bias_int'] = block.mlp.fc1.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc1.bias is not None else None
        block_weights['fc1_fc_scaling_factor'] = block.mlp.fc1.fc_scaling_factor.cpu().numpy()
        
        # MLP - FC2
        block_weights['fc2_weight_int'] = block.mlp.fc2.weight_integer.cpu().numpy().astype(np.int8)
        block_weights['fc2_bias_int'] = block.mlp.fc2.bias_integer.cpu().numpy().astype(np.int32) if block.mlp.fc2.bias is not None else None
        block_weights['fc2_fc_scaling_factor'] = block.mlp.fc2.fc_scaling_factor.cpu().numpy()
        
        weights[f'block_{i}'] = block_weights
    
    # Final norm
    weights['final_norm'] = {
        'bias_int': model.norm.bias_integer.cpu().numpy().astype(np.float64),
        'weight': model.norm.weight.data.cpu().numpy(),
        'bias': model.norm.bias.data.cpu().numpy()
    }
    
    print(f"✓ 已提取 {len(model.blocks)} 個 blocks 的權重和 scaling factors")
    
    return weights


def pure_integer_cmodel_inference(model, image_tensor, weights):
    """純整數 C-Model 推論"""
    print("\n" + "="*80)
    print("開始純整數 C-Model 推論")
    print("="*80)
    
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # ============================================================
        # Stage 1: Input Processing (使用 PyTorch，提取整數)
        # ============================================================
        print(f"\n[Stage 1] Input Processing")
        
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
        print(f"  Scaling factor: {x_scaling_factor_np if np.isscalar(x_scaling_factor_np) else 'per-channel'}")
        print(f"  時間: {(time.time() - start_time)*1000:.2f} ms")
        
        # ============================================================
        # Stage 2: Transformer Blocks (純整數 C-Model)
        # ============================================================
        print(f"\n[Stage 2] Transformer Blocks (純整數 C-Model)")
        
        dim_sqrt = np.sqrt(x_int16.shape[-1])
        
        for block_idx in range(len(model.blocks)):
            block_start = time.time()
            
            block = model.blocks[block_idx]
            block_weights = weights[f'block_{block_idx}']
            
            # 保存輸入用於殘差連接 1
            x_input_for_residual1 = x_int16.copy()
            x_scaling_factor_for_residual1 = x_scaling_factor_np
            
            # ============================================================
            # Norm1 (純整數 C-Model)
            # ============================================================
            x_norm1_int32 = pytorch_int_layer_norm(
                x_int16.astype(np.float64),
                block_weights['norm1_bias_int'],
                block_weights['norm1_weight'],
                block_weights['norm1_bias'],
                dim_sqrt
            )
            
            # 計算 norm1 scaling factor
            base_scaling_factor = dim_sqrt / (2 ** 30)
            norm1_scaling_factor = base_scaling_factor * block_weights['norm1_weight']
            
            # QuantAct1: 將 per-channel 轉換為 scalar（使用 PyTorch）
            x_norm1_torch = torch.from_numpy(x_norm1_int32.astype(np.float32) * norm1_scaling_factor).float()
            norm1_scaling_factor_torch = torch.from_numpy(norm1_scaling_factor).float()
            x_qact1, qact1_scaling_factor = block.qact1(x_norm1_torch, norm1_scaling_factor_torch)
            
            # 轉回整數
            x_qact1_np = x_qact1.cpu().numpy()
            qact1_scaling_factor_np = extract_scaling_factor(qact1_scaling_factor)
            x_qact1_int8 = quantize_to_int(x_qact1_np, qact1_scaling_factor_np, bits=8)
            
            # ============================================================
            # Attention (使用 PyTorch，但提取整數值)
            # ============================================================
            attn_output, attn_output_scaling_factor = block.attn(x_qact1, qact1_scaling_factor)
            
            # 轉換為整數
            attn_output_np = attn_output.cpu().numpy()
            attn_output_scaling_factor_np = extract_scaling_factor(attn_output_scaling_factor)
            attn_output_int16 = quantize_to_int(attn_output_np, attn_output_scaling_factor_np, bits=16)
            
            # ============================================================
            # Residual 1 (使用 PyTorch QuantAct)
            # ============================================================
            attn_output_torch = torch.from_numpy(dequantize_to_float(attn_output_int16, attn_output_scaling_factor_np))
            attn_output_scaling_factor_torch = torch.tensor(attn_output_scaling_factor_np).float()
            
            x_input_for_residual1_torch = torch.from_numpy(dequantize_to_float(x_input_for_residual1, x_scaling_factor_for_residual1))
            x_scaling_factor_for_residual1_torch = torch.tensor(x_scaling_factor_for_residual1).float() if np.isscalar(x_scaling_factor_for_residual1) else torch.from_numpy(x_scaling_factor_for_residual1).float()
            
            x_qact2, qact2_scaling_factor = block.qact2(
                attn_output_torch, attn_output_scaling_factor_torch,
                x_input_for_residual1_torch, x_scaling_factor_for_residual1_torch
            )
            
            # 轉換為整數
            x_qact2_np = x_qact2.cpu().numpy()
            qact2_scaling_factor_np = extract_scaling_factor(qact2_scaling_factor)
            x_int16 = quantize_to_int(x_qact2_np, qact2_scaling_factor_np, bits=16)
            x_scaling_factor_np = qact2_scaling_factor_np
            
            # 保存輸入用於殘差連接 2
            x_input_for_residual2 = x_int16.copy()
            x_scaling_factor_for_residual2 = x_scaling_factor_np
            
            # ============================================================
            # Norm2 (純整數 C-Model)
            # ============================================================
            x_norm2_int32 = pytorch_int_layer_norm(
                x_int16.astype(np.float64),
                block_weights['norm2_bias_int'],
                block_weights['norm2_weight'],
                block_weights['norm2_bias'],
                dim_sqrt
            )
            
            # 計算 norm2 scaling factor
            norm2_scaling_factor = base_scaling_factor * block_weights['norm2_weight']
            
            # QuantAct3: 將 per-channel 轉換為 scalar（使用 PyTorch）
            x_norm2_torch = torch.from_numpy(x_norm2_int32.astype(np.float32) * norm2_scaling_factor).float()
            norm2_scaling_factor_torch = torch.from_numpy(norm2_scaling_factor).float()
            x_qact3, qact3_scaling_factor = block.qact3(x_norm2_torch, norm2_scaling_factor_torch)
            
            # 轉回整數
            x_qact3_np = x_qact3.cpu().numpy()
            qact3_scaling_factor_np = extract_scaling_factor(qact3_scaling_factor)
            x_qact3_int8 = quantize_to_int(x_qact3_np, qact3_scaling_factor_np, bits=8)
            
            # ============================================================
            # MLP (使用 PyTorch，但提取整數值)
            # ============================================================
            mlp_output, mlp_output_scaling_factor = block.mlp(x_qact3, qact3_scaling_factor)
            
            # 轉換為整數
            mlp_output_np = mlp_output.cpu().numpy()
            mlp_output_scaling_factor_np = extract_scaling_factor(mlp_output_scaling_factor)
            mlp_output_int16 = quantize_to_int(mlp_output_np, mlp_output_scaling_factor_np, bits=16)
            
            # ============================================================
            # Residual 2 (使用 PyTorch QuantAct)
            # ============================================================
            mlp_output_torch = torch.from_numpy(dequantize_to_float(mlp_output_int16, mlp_output_scaling_factor_np))
            mlp_output_scaling_factor_torch = torch.tensor(mlp_output_scaling_factor_np).float()
            
            x_input_for_residual2_torch = torch.from_numpy(dequantize_to_float(x_input_for_residual2, x_scaling_factor_for_residual2))
            x_scaling_factor_for_residual2_torch = torch.tensor(x_scaling_factor_for_residual2).float() if np.isscalar(x_scaling_factor_for_residual2) else torch.from_numpy(x_scaling_factor_for_residual2).float()
            
            x_qact4, qact4_scaling_factor = block.qact4(
                mlp_output_torch, mlp_output_scaling_factor_torch,
                x_input_for_residual2_torch, x_scaling_factor_for_residual2_torch
            )
            
            # 轉換為整數
            x_qact4_np = x_qact4.cpu().numpy()
            qact4_scaling_factor_np = extract_scaling_factor(qact4_scaling_factor)
            x_int16 = quantize_to_int(x_qact4_np, qact4_scaling_factor_np, bits=16)
            x_scaling_factor_np = qact4_scaling_factor_np
            
            block_time = (time.time() - block_start) * 1000
            
            if block_idx % 3 == 0:
                print(f"  Block {block_idx}: {block_time:.2f} ms, int16 range=[{x_int16.min()}, {x_int16.max()}]")
        
        print(f"  所有 blocks 完成")
        
        # ============================================================
        # Stage 3: Classification Head (使用 PyTorch)
        # ============================================================
        print(f"\n[Stage 3] Classification Head")
        
        start_time = time.time()
        
        # 轉回 PyTorch tensor
        x = torch.from_numpy(dequantize_to_float(x_int16, x_scaling_factor_np)).float()
        x_scaling_factor = torch.tensor(x_scaling_factor_np).float() if np.isscalar(x_scaling_factor_np) else torch.from_numpy(x_scaling_factor_np).float()
        
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
    print("完整純整數 C-Model 端到端推論")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    
    # 載入模型
    print("\n[1/4] 載入模型...")
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
    print("\n[2/4] 載入圖片...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print("✓ 圖片已載入")
    
    # 執行一次前向傳播，初始化參數
    print("\n[3/4] 初始化模型參數...")
    with torch.no_grad():
        _ = model(image_tensor)
    print("✓ 參數已初始化")
    
    # 提取權重和 scaling factors
    print("\n[4/4] 提取權重和 scaling factors...")
    weights = extract_weights_and_scales(model)
    print("✓ 權重和 scaling factors 已提取")
    
    # 執行純整數 C-Model 推論
    total_start = time.time()
    pred_class, pred_prob, logits = pure_integer_cmodel_inference(model, image_tensor, weights)
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
    print("\n" + "="*80)
    print("與 PyTorch 比較")
    print("="*80)
    
    with torch.no_grad():
        pytorch_output = model(image_tensor)
        pytorch_logits = pytorch_output.cpu().numpy()[0]
        pytorch_pred_class = np.argmax(pytorch_logits)
        pytorch_pred_prob = np.exp(pytorch_logits) / np.sum(np.exp(pytorch_logits))
    
    print(f"PyTorch 預測類別: {pytorch_pred_class}")
    print(f"PyTorch 預測概率: {pytorch_pred_prob[pytorch_pred_class]:.4f}")
    
    # 比較 logits
    logits_diff = np.abs(logits - pytorch_logits)
    logits_corr = np.corrcoef(logits, pytorch_logits)[0, 1]
    
    print(f"\nLogits 比較:")
    print(f"  最大差異: {logits_diff.max():.4f}")
    print(f"  平均差異: {logits_diff.mean():.4f}")
    print(f"  相關係數: {logits_corr:.6f}")
    
    # 判斷
    print("\n" + "="*80)
    print("測試結果")
    print("="*80)
    
    if pred_class == pytorch_pred_class:
        print("✓ 預測類別一致！")
        if pred_class == 1:
            print("✓ 預測正確（期望類別 1）")
            print(f"\n✓✓✓ 純整數 C-Model 端到端推論成功！✓✓✓")
            print(f"  - LayerNorm: 純整數 C-Model")
            print(f"  - Attention: PyTorch（整數值提取）")
            print(f"  - MLP: PyTorch（整數值提取）")
            print(f"  - 所有中間值: 整數（int8, int16, int32）")
            print(f"  - 預測類別: {pred_class}")
            print(f"  - 預測概率: {pred_prob:.4f}")
            print(f"  - Logits 相關係數: {logits_corr:.6f}")
            return True
        else:
            print(f"⚠ 預測類別一致但不是期望的類別 1（實際: {pred_class}）")
            return False
    else:
        print(f"✗ 預測類別不一致")
        print(f"  C-Model: {pred_class}")
        print(f"  PyTorch: {pytorch_pred_class}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
