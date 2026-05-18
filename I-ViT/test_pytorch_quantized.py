"""
測試 PyTorch 量化模型並提取整數中間結果
==================================================================
此腳本用於：
1. 運行 PyTorch 量化模型
2. 提取每一層的整數輸出（output_integer buffer）
3. 驗證整數推論的正確性
4. 生成可用於 C-Model 驗證的 Golden Patterns
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


class IntegerOutputExtractor:
    """提取 PyTorch 量化模型的整數輸出"""
    
    def __init__(self, model):
        self.model = model
        self.integer_outputs = OrderedDict()
        self._register_hooks()
    
    def _register_hooks(self):
        """註冊 hooks 來捕獲 output_integer"""
        
        def make_hook(name):
            def hook(module, input, output):
                try:
                    # 檢查模組是否有 output_integer buffer
                    if hasattr(module, 'output_integer'):
                        int_output = module.output_integer.detach().cpu().numpy()
                        self.integer_outputs[name] = int_output
                        print(f"  ✓ {name}: shape={int_output.shape}, range=[{int_output.min()}, {int_output.max()}]")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
            return hook
        
        # 註冊關鍵層的 hooks
        # Input quantization
        self.model.qact_input.register_forward_hook(make_hook("input_quant"))
        
        # Patch embedding
        self.model.patch_embed.register_forward_hook(make_hook("patch_embed"))
        
        # Position embedding
        self.model.qact1.register_forward_hook(make_hook("pos_embed"))
        
        # Transformer blocks
        for i, block in enumerate(self.model.blocks):
            # LayerNorm 1
            block.norm1.register_forward_hook(make_hook(f"block{i}_norm1"))
            
            # Attention QKV
            block.attn.qkv.register_forward_hook(make_hook(f"block{i}_attn_qkv"))
            
            # Attention output
            block.attn.proj.register_forward_hook(make_hook(f"block{i}_attn_proj"))
            
            # LayerNorm 2
            block.norm2.register_forward_hook(make_hook(f"block{i}_norm2"))
            
            # MLP FC1
            block.mlp.fc1.register_forward_hook(make_hook(f"block{i}_mlp_fc1"))
            
            # MLP FC2
            block.mlp.fc2.register_forward_hook(make_hook(f"block{i}_mlp_fc2"))
        
        # Final norm
        self.model.norm.register_forward_hook(make_hook("final_norm"))
        
        # Classification head
        self.model.head.register_forward_hook(make_hook("head"))
    
    def extract(self, image_tensor):
        """
        執行推理並提取整數輸出
        
        Args:
            image_tensor: 輸入圖片 tensor [1, 3, 224, 224]
        
        Returns:
            output: 模型輸出
            integer_outputs: 整數輸出字典
        """
        self.integer_outputs.clear()
        
        print("\n開始提取整數輸出...")
        with torch.no_grad():
            output = self.model(image_tensor)
        
        print(f"\n✓ 提取了 {len(self.integer_outputs)} 個整數輸出")
        return output, self.integer_outputs


def load_and_preprocess_image(image_path):
    """載入並預處理圖片"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def analyze_integer_outputs(integer_outputs):
    """分析整數輸出"""
    print("\n" + "=" * 80)
    print("整數輸出分析")
    print("=" * 80)
    
    for name, output in integer_outputs.items():
        print(f"\n{name}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Range: [{output.min()}, {output.max()}]")
        print(f"  Mean: {output.mean():.2f}, Std: {output.std():.2f}")
        
        # 檢查是否有異常值
        if np.any(np.isnan(output)):
            print(f"  ⚠ 包含 NaN 值！")
        if np.any(np.isinf(output)):
            print(f"  ⚠ 包含 Inf 值！")


def save_integer_outputs(integer_outputs, output_dir):
    """保存整數輸出"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'pytorch_integer_outputs.npz')
    np.savez_compressed(output_file, **integer_outputs)
    
    print(f"\n✓ 整數輸出已保存到: {output_file}")
    
    # 生成報告
    report_file = os.path.join(output_dir, 'pytorch_integer_outputs_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch 量化模型整數輸出報告\n")
        f.write("=" * 80 + "\n\n")
        
        for name, output in integer_outputs.items():
            f.write(f"{name}:\n")
            f.write(f"  Shape: {output.shape}\n")
            f.write(f"  Dtype: {output.dtype}\n")
            f.write(f"  Range: [{output.min()}, {output.max()}]\n")
            f.write(f"  Mean: {output.mean():.2f}, Std: {output.std():.2f}\n")
            f.write("\n")
    
    print(f"✓ 報告已保存到: {report_file}")


def main():
    """主函數"""
    print("=" * 80)
    print("測試 PyTorch 量化模型並提取整數輸出")
    print("=" * 80)
    
    # 設定路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, 'output_gpu', 'checkpoint_converted.pth')
    test_image_path = os.path.join(os.path.dirname(current_dir), 'data', 'test_image.JPEG')
    output_dir = os.path.join(os.path.dirname(current_dir), 'pytorch_integer_outputs')
    
    # 檢查文件
    if not os.path.exists(checkpoint_path):
        print(f"✗ 找不到 checkpoint: {checkpoint_path}")
        return False
    
    if not os.path.exists(test_image_path):
        print(f"✗ 找不到測試圖片: {test_image_path}")
        return False
    
    # 1. 載入圖片
    print("\n[1/4] 載入並預處理圖片...")
    image_tensor = load_and_preprocess_image(test_image_path)
    print(f"✓ 圖片 shape: {image_tensor.shape}")
    
    # 2. 載入模型
    print("\n[2/4] 載入 PyTorch 量化模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deit_tiny_patch16_224(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    # 過濾運行時 buffer
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'output_integer' in key or 'norm_scaling_factor' in key:
            continue
        filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✓ 模型已載入 (device: {device})")
    
    # 3. 提取整數輸出
    print("\n[3/4] 執行推理並提取整數輸出...")
    extractor = IntegerOutputExtractor(model)
    image_tensor = image_tensor.to(device)
    output, integer_outputs = extractor.extract(image_tensor)
    
    # 4. 分析結果
    print("\n[4/4] 分析結果...")
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0, pred_class].item()
    
    print(f"\n預測類別: {pred_class}")
    print(f"預測概率: {pred_prob:.6f}")
    
    print("\nTop-5 預測:")
    top5_probs, top5_classes = torch.topk(probs, 5)
    for i in range(5):
        print(f"  {i+1}. 類別 {top5_classes[0, i].item()}: {top5_probs[0, i].item():.6f}")
    
    # 5. 分析整數輸出
    analyze_integer_outputs(integer_outputs)
    
    # 6. 保存結果
    print("\n[5/4] 保存整數輸出...")
    save_integer_outputs(integer_outputs, output_dir)
    
    print("\n" + "=" * 80)
    print("測試完成")
    print("=" * 80)
    print(f"\n整數輸出已保存到: {output_dir}")
    print("這些整數輸出可以用於：")
    print("  1. C-Model 驗證")
    print("  2. RTL 模擬的 Golden Reference")
    print("  3. 逐層精度分析")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
