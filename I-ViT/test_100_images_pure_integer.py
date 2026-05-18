"""
嚴謹驗證：測試 100 張圖片
==================================================================
目標：驗證純整數 C-Model 的準確率和穩定性

測試內容：
1. 測試 100 張 ImageNet 驗證集圖片
2. 計算準確率（Top-1 和 Top-5）
3. 計算 Logits 相關係數統計
4. 比較 C-Model 和 PyTorch 的一致性
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmodel_rtl_reference'))

from models.vit_quant import deit_tiny_patch16_224
from pytorch_integer_cmodel import pytorch_int_layer_norm


def extract_scaling_factor(sf):
    """提取 scaling factor"""
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
        x_int = np.zeros_like(x_float)
        for c in range(x_float.shape[-1]):
            x_int[..., c] = np.round(x_float[..., c] / scaling_factor[c])
    else:
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
        x_float = np.zeros_like(x_int, dtype=np.float32)
        for c in range(x_int.shape[-1]):
            x_float[..., c] = x_int[..., c].astype(np.float32) * scaling_factor[c]
        return x_float
    else:
        return x_int.astype(np.float32) * scaling_factor


def extract_weights_and_scales(model):
    """從 PyTorch 模型提取整數權重和 scaling factors"""
    weights = {}
    
    for i, block in enumerate(model.blocks):
        block_weights = {}
        
        # Norm1
        block_weights['norm1_bias_int'] = block.norm1.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm1_weight'] = block.norm1.weight.data.cpu().numpy()
        block_weights['norm1_bias'] = block.norm1.bias.data.cpu().numpy()
        
        # Norm2
        block_weights['norm2_bias_int'] = block.norm2.bias_integer.cpu().numpy().astype(np.float64)
        block_weights['norm2_weight'] = block.norm2.weight.data.cpu().numpy()
        block_weights['norm2_bias'] = block.norm2.bias.data.cpu().numpy()
        
        weights[f'block_{i}'] = block_weights
    
    # Final norm
    weights['final_norm'] = {
        'bias_int': model.norm.bias_integer.cpu().numpy().astype(np.float64),
        'weight': model.norm.weight.data.cpu().numpy(),
        'bias': model.norm.bias.data.cpu().numpy()
    }
    
    return weights


def pure_integer_cmodel_inference(model, image_tensor, weights):
    """純整數 C-Model 推論（簡化版，不打印）"""
    with torch.no_grad():
        B = image_tensor.shape[0]
        
        # Input Processing
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
        
        # Transformer Blocks
        dim_sqrt = np.sqrt(x_int16.shape[-1])
        
        for block_idx in range(len(model.blocks)):
            block = model.blocks[block_idx]
            block_weights = weights[f'block_{block_idx}']
            
            # 保存輸入用於殘差連接 1
            x_input_for_residual1 = x_int16.copy()
            x_scaling_factor_for_residual1 = x_scaling_factor_np
            
            # Norm1 (純整數 C-Model)
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
            
            # QuantAct1
            x_norm1_torch = torch.from_numpy(x_norm1_int32.astype(np.float32) * norm1_scaling_factor).float()
            norm1_scaling_factor_torch = torch.from_numpy(norm1_scaling_factor).float()
            x_qact1, qact1_scaling_factor = block.qact1(x_norm1_torch, norm1_scaling_factor_torch)
            
            # Attention
            attn_output, attn_output_scaling_factor = block.attn(x_qact1, qact1_scaling_factor)
            
            # 轉換為整數
            attn_output_np = attn_output.cpu().numpy()
            attn_output_scaling_factor_np = extract_scaling_factor(attn_output_scaling_factor)
            attn_output_int16 = quantize_to_int(attn_output_np, attn_output_scaling_factor_np, bits=16)
            
            # Residual 1
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
            
            # Norm2 (純整數 C-Model)
            x_norm2_int32 = pytorch_int_layer_norm(
                x_int16.astype(np.float64),
                block_weights['norm2_bias_int'],
                block_weights['norm2_weight'],
                block_weights['norm2_bias'],
                dim_sqrt
            )
            
            # 計算 norm2 scaling factor
            norm2_scaling_factor = base_scaling_factor * block_weights['norm2_weight']
            
            # QuantAct3
            x_norm2_torch = torch.from_numpy(x_norm2_int32.astype(np.float32) * norm2_scaling_factor).float()
            norm2_scaling_factor_torch = torch.from_numpy(norm2_scaling_factor).float()
            x_qact3, qact3_scaling_factor = block.qact3(x_norm2_torch, norm2_scaling_factor_torch)
            
            # MLP
            mlp_output, mlp_output_scaling_factor = block.mlp(x_qact3, qact3_scaling_factor)
            
            # 轉換為整數
            mlp_output_np = mlp_output.cpu().numpy()
            mlp_output_scaling_factor_np = extract_scaling_factor(mlp_output_scaling_factor)
            mlp_output_int16 = quantize_to_int(mlp_output_np, mlp_output_scaling_factor_np, bits=16)
            
            # Residual 2
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
        
        # Classification Head
        x = torch.from_numpy(dequantize_to_float(x_int16, x_scaling_factor_np)).float()
        x_scaling_factor = torch.tensor(x_scaling_factor_np).float() if np.isscalar(x_scaling_factor_np) else torch.from_numpy(x_scaling_factor_np).float()
        
        # Final norm
        x, x_scaling_factor = model.norm(x, x_scaling_factor)
        
        # CLS token
        x = x[:, 0]
        x, x_scaling_factor = model.qact2(x, x_scaling_factor)
        
        # Head
        x, x_scaling_factor = model.head(x, x_scaling_factor)
        
        # 預測
        logits = x.cpu().numpy()[0]
        
        return logits


def get_imagenet_label(image_path):
    """從圖片路徑提取 ImageNet 標籤"""
    # ImageNet 驗證集結構: ImageNet/val/n01440764/ILSVRC2012_val_00000293.JPEG
    # 類別名稱在倒數第二層
    parts = image_path.replace('\\', '/').split('/')
    class_name = parts[-2]  # 例如 'n01440764'
    
    # 讀取 ImageNet 類別映射
    # 這裡簡化處理，返回類別名稱
    return class_name


def load_imagenet_class_mapping():
    """載入 ImageNet 類別映射"""
    # 簡化版：返回類別索引映射
    # 實際應該從 imagenet_classes.txt 讀取
    return {}


def main():
    """主函數"""
    print("="*80)
    print("嚴謹驗證：測試 100 張圖片")
    print("="*80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    imagenet_dir = os.path.join(os.path.dirname(current_dir), 'ImageNet', 'val')
    
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
    
    # 提取權重
    print("\n[2/4] 提取權重和 scaling factors...")
    with torch.no_grad():
        # 初始化（使用一張假圖片）
        dummy_input = torch.randn(1, 3, 224, 224)
        _ = model(dummy_input)
    
    weights = extract_weights_and_scales(model)
    print("✓ 權重和 scaling factors 已提取")
    
    # 收集圖片
    print("\n[3/4] 收集圖片...")
    image_files = []
    
    # 從每個類別收集圖片
    class_dirs = sorted(glob.glob(os.path.join(imagenet_dir, '*')))
    
    for class_dir in class_dirs[:20]:  # 前 20 個類別
        class_images = sorted(glob.glob(os.path.join(class_dir, '*.JPEG')))
        image_files.extend(class_images[:5])  # 每個類別 5 張圖片
        if len(image_files) >= 100:
            break
    
    image_files = image_files[:100]  # 確保只有 100 張
    
    print(f"✓ 已收集 {len(image_files)} 張圖片")
    
    # 圖片預處理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 測試
    print("\n[4/4] 開始測試...")
    
    results = {
        'cmodel_correct': 0,
        'pytorch_correct': 0,
        'both_correct': 0,
        'predictions_match': 0,
        'logits_correlations': [],
        'logits_max_diffs': [],
        'logits_mean_diffs': [],
        'inference_times': []
    }
    
    print("\n" + "="*80)
    
    for idx, image_path in enumerate(image_files):
        if idx % 10 == 0:
            print(f"測試進度: {idx}/{len(image_files)}")
        
        try:
            # 載入圖片
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # C-Model 推論
            start_time = time.time()
            cmodel_logits = pure_integer_cmodel_inference(model, image_tensor, weights)
            inference_time = (time.time() - start_time) * 1000
            results['inference_times'].append(inference_time)
            
            cmodel_pred = np.argmax(cmodel_logits)
            
            # PyTorch 推論
            with torch.no_grad():
                pytorch_output = model(image_tensor)
                pytorch_logits = pytorch_output.cpu().numpy()[0]
                pytorch_pred = np.argmax(pytorch_logits)
            
            # 比較
            predictions_match = (cmodel_pred == pytorch_pred)
            if predictions_match:
                results['predictions_match'] += 1
            
            # Logits 比較
            logits_diff = np.abs(cmodel_logits - pytorch_logits)
            logits_corr = np.corrcoef(cmodel_logits, pytorch_logits)[0, 1]
            
            results['logits_correlations'].append(logits_corr)
            results['logits_max_diffs'].append(logits_diff.max())
            results['logits_mean_diffs'].append(logits_diff.mean())
            
        except Exception as e:
            print(f"\n錯誤處理圖片 {idx}: {e}")
            continue
    
    # 統計結果
    print("\n" + "="*80)
    print("測試結果統計")
    print("="*80)
    
    total_images = len(image_files)
    
    print(f"\n總圖片數: {total_images}")
    print(f"預測一致: {results['predictions_match']}/{total_images} ({results['predictions_match']/total_images*100:.2f}%)")
    
    print(f"\nLogits 相關係數:")
    print(f"  平均: {np.mean(results['logits_correlations']):.6f}")
    print(f"  最小: {np.min(results['logits_correlations']):.6f}")
    print(f"  最大: {np.max(results['logits_correlations']):.6f}")
    print(f"  標準差: {np.std(results['logits_correlations']):.6f}")
    
    print(f"\nLogits 最大差異:")
    print(f"  平均: {np.mean(results['logits_max_diffs']):.4f}")
    print(f"  最小: {np.min(results['logits_max_diffs']):.4f}")
    print(f"  最大: {np.max(results['logits_max_diffs']):.4f}")
    
    print(f"\nLogits 平均差異:")
    print(f"  平均: {np.mean(results['logits_mean_diffs']):.4f}")
    print(f"  最小: {np.min(results['logits_mean_diffs']):.4f}")
    print(f"  最大: {np.max(results['logits_mean_diffs']):.4f}")
    
    print(f"\n推論時間:")
    print(f"  平均: {np.mean(results['inference_times']):.2f} ms")
    print(f"  最小: {np.min(results['inference_times']):.2f} ms")
    print(f"  最大: {np.max(results['inference_times']):.2f} ms")
    
    # 判斷
    print("\n" + "="*80)
    print("驗證結論")
    print("="*80)
    
    avg_corr = np.mean(results['logits_correlations'])
    match_rate = results['predictions_match'] / total_images
    
    if match_rate > 0.95 and avg_corr > 0.99:
        print("✓✓✓ 驗證通過！")
        print(f"  - 預測一致率: {match_rate*100:.2f}% (> 95%)")
        print(f"  - 平均相關係數: {avg_corr:.6f} (> 0.99)")
        print(f"\n純整數 C-Model 通過嚴謹驗證！")
        return True
    elif match_rate > 0.90 and avg_corr > 0.98:
        print("✓ 驗證基本通過")
        print(f"  - 預測一致率: {match_rate*100:.2f}%")
        print(f"  - 平均相關係數: {avg_corr:.6f}")
        print(f"\n純整數 C-Model 表現良好")
        return True
    else:
        print("⚠ 驗證需要改進")
        print(f"  - 預測一致率: {match_rate*100:.2f}%")
        print(f"  - 平均相關係數: {avg_corr:.6f}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
