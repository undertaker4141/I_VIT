"""
提取 PyTorch 量化模型的 Golden Patterns 用於 RTL 模擬驗證
==================================================================
此腳本從已驗證的 PyTorch 量化模型中提取中間層的輸入/輸出數據，
作為 RTL 模擬的 Golden Reference。

提取的數據包括：
1. 每一層的輸入/輸出 (int8/int32)
2. 每一層的 scaling factors
3. 權重和偏置參數
4. 最終的分類結果

這些數據將用於：
- C-Model 驗證
- RTL 模擬的測試向量
- 逐層精度分析
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

# 添加模型路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.vit_quant import deit_tiny_patch16_224


class GoldenPatternExtractor:
    """提取 Golden Patterns 的類"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.patterns = OrderedDict()
        self.layer_counter = 0
        
        # 註冊 hooks 來捕獲中間層輸出
        self._register_hooks()
    
    def _register_hooks(self):
        """註冊 forward hooks 來捕獲中間層數據"""
        
        def make_hook(name):
            def hook(module, input, output):
                try:
                    # 保存輸出
                    if isinstance(output, tuple):
                        # 如果輸出是 tuple (data, scaling_factor)
                        data, scale = output
                        self.patterns[f"{name}_output"] = data.detach().cpu().numpy()
                        if isinstance(scale, torch.Tensor):
                            self.patterns[f"{name}_scale"] = scale.detach().cpu().numpy()
                        else:
                            self.patterns[f"{name}_scale"] = np.array(scale)
                        
                        # 嘗試提取整數值（如果模組有 output_integer buffer）
                        if hasattr(module, 'output_integer'):
                            try:
                                int_output = module.output_integer.detach().cpu().numpy()
                                self.patterns[f"{name}_output_int"] = int_output
                            except:
                                pass
                    else:
                        self.patterns[f"{name}_output"] = output.detach().cpu().numpy()
                    
                    # 保存輸入
                    if isinstance(input, tuple) and len(input) > 0:
                        inp = input[0]
                        if isinstance(inp, torch.Tensor):
                            self.patterns[f"{name}_input"] = inp.detach().cpu().numpy()
                except Exception as e:
                    print(f"Warning: Failed to capture {name}: {e}")
            return hook
        
        # 註冊關鍵層的 hooks
        # 1. Input quantization
        self.model.qact_input.register_forward_hook(make_hook("0_input_quant"))
        
        # 2. Patch embedding
        self.model.patch_embed.register_forward_hook(make_hook("1_patch_embed"))
        
        # 3. Position embedding
        self.model.qact1.register_forward_hook(make_hook("2_pos_embed"))
        
        # 4. Transformer blocks
        for i, block in enumerate(self.model.blocks):
            # LayerNorm 1
            block.norm1.register_forward_hook(make_hook(f"3_block{i}_norm1"))
            
            # Attention QKV
            block.attn.qkv.register_forward_hook(make_hook(f"4_block{i}_attn_qkv"))
            
            # Attention MatMul 1 (Q @ K^T)
            block.attn.matmul_1.register_forward_hook(make_hook(f"5_block{i}_attn_qk"))
            
            # Softmax
            block.attn.int_softmax.register_forward_hook(make_hook(f"6_block{i}_attn_softmax"))
            
            # Attention MatMul 2 (Attn @ V)
            block.attn.matmul_2.register_forward_hook(make_hook(f"7_block{i}_attn_out"))
            
            # Attention Projection
            block.attn.proj.register_forward_hook(make_hook(f"8_block{i}_attn_proj"))
            
            # LayerNorm 2
            block.norm2.register_forward_hook(make_hook(f"9_block{i}_norm2"))
            
            # MLP FC1
            block.mlp.fc1.register_forward_hook(make_hook(f"10_block{i}_mlp_fc1"))
            
            # GELU
            block.mlp.act.register_forward_hook(make_hook(f"11_block{i}_mlp_gelu"))
            
            # MLP FC2
            block.mlp.fc2.register_forward_hook(make_hook(f"12_block{i}_mlp_fc2"))
        
        # 5. Final norm
        self.model.norm.register_forward_hook(make_hook("13_final_norm"))
        
        # 6. Classification head
        self.model.head.register_forward_hook(make_hook("14_head"))
    
    def extract(self, image_path, output_dir):
        """
        提取單張圖片的 Golden Patterns
        
        Args:
            image_path: 輸入圖片路徑
            output_dir: 輸出目錄
        """
        # 清空之前的 patterns
        self.patterns.clear()
        
        # 載入並預處理圖片
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 保存原始輸入
        self.patterns['input_image'] = input_tensor.cpu().numpy()
        
        # 執行推理
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # 保存最終輸出
            self.patterns['final_output'] = output.cpu().numpy()
            
            # 計算預測結果
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()
            
            self.patterns['prediction'] = {
                'class': pred_class,
                'probability': pred_prob,
                'top5_classes': torch.topk(probs, 5).indices[0].cpu().numpy(),
                'top5_probs': torch.topk(probs, 5).values[0].cpu().numpy()
            }
        
        # 保存所有 patterns
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存為 .npz 格式（壓縮）
        output_file = os.path.join(output_dir, 'golden_patterns.npz')
        np.savez_compressed(output_file, **self.patterns)
        
        print(f"✓ Golden patterns 已保存到: {output_file}")
        print(f"  總共提取了 {len(self.patterns)} 個數據項")
        print(f"  預測類別: {pred_class}, 概率: {pred_prob:.4f}")
        
        # 生成可讀的摘要報告
        self._generate_report(output_dir)
        
        return self.patterns
    
    def _generate_report(self, output_dir):
        """生成可讀的摘要報告"""
        report_file = os.path.join(output_dir, 'golden_patterns_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Golden Patterns 提取報告\n")
            f.write("=" * 80 + "\n\n")
            
            # 預測結果
            pred = self.patterns['prediction']
            f.write("預測結果:\n")
            f.write(f"  類別: {pred['class']}\n")
            f.write(f"  概率: {pred['probability']:.4f}\n")
            f.write(f"  Top-5 類別: {pred['top5_classes']}\n")
            f.write(f"  Top-5 概率: {pred['top5_probs']}\n\n")
            
            # 數據統計
            f.write("提取的數據層:\n")
            f.write("-" * 80 + "\n")
            
            for key, value in self.patterns.items():
                if key == 'prediction':
                    continue
                
                if isinstance(value, np.ndarray):
                    f.write(f"{key}:\n")
                    f.write(f"  Shape: {value.shape}\n")
                    f.write(f"  Dtype: {value.dtype}\n")
                    f.write(f"  Range: [{value.min():.6f}, {value.max():.6f}]\n")
                    f.write(f"  Mean: {value.mean():.6f}, Std: {value.std():.6f}\n")
                    f.write("\n")
                elif isinstance(value, (int, float)):
                    f.write(f"{key}: {value}\n\n")
        
        print(f"✓ 報告已保存到: {report_file}")


def extract_model_weights(model, output_dir):
    """
    提取模型的所有權重和偏置參數
    
    Args:
        model: PyTorch 模型
        output_dir: 輸出目錄
    """
    weights = OrderedDict()
    
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()
    
    # 保存權重
    os.makedirs(output_dir, exist_ok=True)
    weights_file = os.path.join(output_dir, 'model_weights.npz')
    np.savez_compressed(weights_file, **weights)
    
    print(f"✓ 模型權重已保存到: {weights_file}")
    print(f"  總共 {len(weights)} 個參數")
    
    # 生成權重報告
    report_file = os.path.join(output_dir, 'model_weights_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型權重報告\n")
        f.write("=" * 80 + "\n\n")
        
        total_params = 0
        for name, param in weights.items():
            f.write(f"{name}:\n")
            f.write(f"  Shape: {param.shape}\n")
            f.write(f"  Dtype: {param.dtype}\n")
            f.write(f"  Range: [{param.min():.6f}, {param.max():.6f}]\n")
            f.write(f"  Params: {param.size}\n")
            f.write("\n")
            total_params += param.size
        
        f.write(f"總參數量: {total_params:,}\n")
    
    print(f"✓ 權重報告已保存到: {report_file}")


def main():
    """主函數"""
    print("=" * 80)
    print("提取 PyTorch 量化模型的 Golden Patterns")
    print("=" * 80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    output_dir = os.path.join(os.path.dirname(current_dir), 'golden_patterns')
    
    # 檢查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"✗ 找不到 checkpoint: {checkpoint_path}")
        print(f"  請先執行: python convert_checkpoint.py")
        return
    
    if not os.path.exists(test_image_path):
        print(f"✗ 找不到測試圖片: {test_image_path}")
        return
    
    print(f"\nStep 1: 載入模型...")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deit_tiny_patch16_224(pretrained=False)
    
    # 載入訓練好的權重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    # 過濾掉運行時 buffer
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'output_integer' in key or 'norm_scaling_factor' in key:
            continue
        filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✓ 模型已載入 (device: {device})")
    
    # 提取權重
    print(f"\nStep 2: 提取模型權重...")
    extract_model_weights(model, output_dir)
    
    # 提取 Golden Patterns
    print(f"\nStep 3: 提取 Golden Patterns...")
    print(f"  測試圖片: {test_image_path}")
    
    extractor = GoldenPatternExtractor(model, device)
    patterns = extractor.extract(test_image_path, output_dir)
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n輸出目錄: {output_dir}")
    print(f"  - golden_patterns.npz: 中間層輸入/輸出數據")
    print(f"  - golden_patterns_report.txt: 數據摘要報告")
    print(f"  - model_weights.npz: 模型權重參數")
    print(f"  - model_weights_report.txt: 權重摘要報告")
    print("\n這些文件可用於:")
    print("  1. C-Model 驗證 (與 cmodel_rtl_reference/ 中的代碼對比)")
    print("  2. RTL 模擬測試向量生成")
    print("  3. 逐層精度分析")


if __name__ == '__main__':
    main()
