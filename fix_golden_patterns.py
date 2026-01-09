#!/usr/bin/env python3
"""
修補腳本：修正 IntLayerNorm 的 Golden Pattern 生成
===================================================

這個腳本修改 IntLayerNorm 模組，使其在 forward 過程中保存內部的 y_int，
然後重新抽取 Golden Pattern。

問題：
------
原本的 extract_patterns.py 使用 `round(float_output / scaling_factor)` 計算 int 輸出，
這會引入浮點除法的捨入誤差。

解決方案：
---------
直接保存 IntLayerNorm 內部計算的 `y_int + bias_int`。
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path

# 添加 I-ViT 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "I-ViT"))

from models import deit_tiny_patch16_224, IntLayerNorm
from torchvision import transforms
from PIL import Image


def patch_int_layer_norm():
    """
    修補 IntLayerNorm.forward() 以保存內部的整數結果
    """
    original_forward = IntLayerNorm.forward
    
    def patched_forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # 以下是原始的整數計算
        x_int = x / scaling_factor
        
        # 使用原版的 round_ste (需要導入)
        from models.quantization_utils.quant_utils import round_ste, floor_ste
        
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int_scaled = floor_ste.apply(y_int * factor / 2)
        new_scaling_factor = self.dim_sqrt / 2 ** 30

        # bias 計算
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / new_scaling_factor)

        self.bias_integer = bias_int

        # 保存內部整數結果 (這是真正的 y_int + bias_int)
        self._internal_y_int = (y_int_scaled + bias_int).detach()
        
        y_int_output = y_int_scaled + bias_int
        new_scaling_factor = new_scaling_factor * self.weight
        final_x = y_int_output * new_scaling_factor
        self.norm_scaling_factor = new_scaling_factor
        
        return final_x, new_scaling_factor
    
    IntLayerNorm.forward = patched_forward
    print("[Patch] IntLayerNorm.forward() has been patched to save internal y_int")


def extract_true_int_outputs(model, input_tensor, output_dir):
    """
    運行 forward pass 後，從 IntLayerNorm 模組中提取真正的整數輸出
    """
    golden_dir = Path(output_dir) / "golden_fixed"
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # 遍歷所有 IntLayerNorm 模組
    extracted = {}
    for name, module in model.named_modules():
        if isinstance(module, IntLayerNorm):
            if hasattr(module, '_internal_y_int'):
                int_tensor = module._internal_y_int
                
                prefix = name.replace('.', '_')
                int_data = int_tensor.cpu().numpy().astype(np.int32)
                
                # 保存
                np.save(golden_dir / f"{prefix}_int_true.npy", int_data)
                
                extracted[prefix] = {
                    'shape': list(int_data.shape),
                    'dtype': 'int32',
                    'min': int(int_data.min()),
                    'max': int(int_data.max())
                }
                
                print(f"  Saved: {prefix}_int_true.npy, shape={int_data.shape}, range=[{int_data.min()}, {int_data.max()}]")
    
    # 保存 info
    with open(golden_dir / "layer_info_fixed.json", 'w') as f:
        json.dump(extracted, f, indent=2)
    
    return extracted


def compare_patterns(output_dir):
    """
    比較修正後的 pattern 與 CModel
    """
    base_dir = Path(output_dir)
    golden_fixed_dir = base_dir / "golden_fixed"
    
    print("\n" + "=" * 60)
    print("比較 CModel 與修正後的 Golden Pattern")
    print("=" * 60)
    
    # 載入輸入和 bias
    input_int = np.load(base_dir / "golden/qact1_int.npy")
    bias_integer = np.load(base_dir / "weights/blocks_0_norm1_bias_integer.npy")
    norm_weight = np.load(base_dir / "weights/blocks_0_norm1_weight.npy")
    
    # 載入修正後的 Golden
    golden_fixed = np.load(golden_fixed_dir / "blocks_0_norm1_int_true.npy")
    
    # 運行 CModel
    my_output = run_cmodel(input_int, bias_integer)
    
    # 比較
    diff = my_output - golden_fixed
    mismatches = np.sum(diff != 0)
    
    print(f"\n修正後的比較結果:")
    print(f"  Mismatches: {mismatches} / {diff.size} ({mismatches/diff.size*100:.2f}%)")
    print(f"  Max diff: {np.abs(diff).max()}")
    
    if mismatches == 0:
        print("\n✅ 完美匹配！CModel 現在與 PyTorch 的整數運算完全一致。")
    else:
        print(f"\n⚠️ 仍有 {mismatches} 個不匹配，需要進一步調查。")
        
        # 顯示前 5 個不匹配
        indices = np.where(diff != 0)
        print("\n前 5 個不匹配:")
        for i in range(min(5, len(indices[0]))):
            idx = tuple(ind[i] for ind in indices)
            print(f"  Pos {idx}: CModel={my_output[idx]}, Golden={golden_fixed[idx]}, Diff={diff[idx]}")
    
    return mismatches == 0


def run_cmodel(x_int, bias_integer):
    """
    純整數 CModel 實現
    """
    x_val = x_int.astype(np.float64)
    
    N = x_int.shape[-1]  # 192
    
    # 1. Mean (銀行家捨入)
    mean_float = np.mean(x_val, axis=-1, keepdims=True)
    mean_int = np.around(mean_float)
    
    # 2. Centering
    y_int = x_val - mean_int
    
    # 3. Variance
    y_sq_int = y_int ** 2
    var_int = np.sum(y_sq_int, axis=-1, keepdims=True)
    
    # 4. Newton-Raphson sqrt
    k = np.full_like(var_int, 2**16, dtype=np.float64)
    for _ in range(10):
        div_result = np.floor(var_int / k)
        k = np.floor((k + div_result) / 2)
    std_int = k
    
    # 5. Factor
    MAX_INT31 = 2**31 - 1
    std_int_safe = np.maximum(std_int, 1)
    factor = np.floor(MAX_INT31 / std_int_safe)
    
    # 6. Scaling
    term = y_int * factor
    y_int_scaled = np.floor(term / 2)
    
    # 7. Add Bias
    output_int = y_int_scaled + bias_integer.astype(np.float64)
    
    return output_int.astype(np.int32)


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
    
    return tensor


def main():
    # 配置
    checkpoint_path = "I-ViT/checkpoints/qat_calibrated.pth"
    test_image_path = "test_data/test_image.JPEG"
    output_dir = "patterns"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("I-ViT Pattern 修正腳本")
    print("=" * 60)
    print(f"Device: {device}")
    
    # 先修補 IntLayerNorm
    print("\n[1/4] 修補 IntLayerNorm...")
    patch_int_layer_norm()
    
    # 載入模型
    print("\n[2/4] 載入模型...")
    model = deit_tiny_patch16_224(pretrained=True, num_classes=1000)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() 
                         if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # 載入測試圖片
    print("\n[3/4] 載入測試圖片並抽取真正的整數輸出...")
    input_tensor = load_and_preprocess_image(test_image_path, device)
    
    # 抽取真正的整數輸出
    extracted = extract_true_int_outputs(model, input_tensor, output_dir)
    print(f"\n  提取了 {len(extracted)} 個 IntLayerNorm 層的整數輸出")
    
    # 比較
    print("\n[4/4] 比較 CModel 與修正後的 Golden Pattern...")
    success = compare_patterns(output_dir)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 修正成功！")
        print("=" * 60)
        print("""
現在您可以:
1. 使用 patterns/golden_fixed/ 目錄中的 *_int_true.npy 作為新的 Golden Pattern
2. 這些 Pattern 現在與純整數 CModel 完全匹配
        """)
    else:
        print("⚠️ 仍有差異，請查看上方的詳細輸出")
        print("=" * 60)


if __name__ == "__main__":
    main()
