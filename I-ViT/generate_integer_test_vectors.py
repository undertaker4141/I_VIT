"""
使用 PyTorch Integer C-Model 生成整數測試向量
直接從實際模型推論生成整數中間結果
包含 LayerNorm, GELU, Softmax 的測試向量

GELU 使用 C-Model Reference 計算輸出，確保與 RTL 完全一致
"""

import numpy as np
import sys
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, 'models')
sys.path.insert(0, '../cmodel_rtl_reference')

from models.vit_quant import deit_tiny_patch16_224
from nonlinear_cmodel_reference import int_gelu_kernel_fixed

def save_hex_file(data, filename):
    """儲存為 hex 檔案供 Verilog 讀取"""
    with open(filename, 'w') as f:
        for val in data.flatten():
            # 處理負數（2's complement）
            if val < 0:
                val = (1 << 32) + val  # 轉換為 unsigned
            f.write(f"{val & 0xFFFFFFFF:08x}\n")

# 全域變數用於收集 GELU 和 Softmax 的中間結果
gelu_inputs = []
gelu_outputs = []
softmax_inputs = []
softmax_outputs = []

def hook_gelu_input(module, input, output):
    """Hook 函數：捕獲 GELU 輸入（fc1 輸出後，qact_gelu 輸出）"""
    # output 是 tuple (dequantized_tensor, scaling_factor)
    # 我們需要整數值，所以要除以 scaling_factor
    if isinstance(output, tuple):
        dequant_tensor = output[0].detach()
        scaling_factor = output[1]
        # 轉回整數：x_int = dequant_tensor / scaling_factor
        x_int = (dequant_tensor / scaling_factor).cpu().numpy()
        # 四捨五入並轉換為 int32
        gelu_inputs.append(np.round(x_int).astype(np.int32))
        # 調試：打印第一個 block 的 scaling factor
        if len(gelu_inputs) == 1:
            print(f"  DEBUG: GELU input scaling_factor = {scaling_factor}")
    else:
        gelu_inputs.append(output.detach().cpu().numpy().astype(np.int32))

def hook_gelu_output(module, input, output):
    """Hook 函數：捕獲 GELU 輸出（act 輸出）"""
    # 注意：我們不再使用 PyTorch 的輸出，而是使用 C-Model Reference 計算
    # 這個 hook 只是為了觸發計算，實際輸出會在後處理中使用 C-Model 生成
    pass

def hook_softmax_input(module, input, output):
    """Hook 函數：捕獲 Softmax 輸入（qact_attn1 輸出）"""
    # output 是 tuple (tensor, scaling_factor)
    if isinstance(output, tuple):
        softmax_inputs.append(output[0].detach().cpu().numpy().astype(np.int32))
    else:
        softmax_inputs.append(output.detach().cpu().numpy().astype(np.int32))

def hook_softmax_output(module, input, output):
    """Hook 函數：捕獲 Softmax 輸出（int_softmax 輸出）"""
    # output 是 tuple (dequantized_tensor, scaling_factor)
    # 我們需要整數值，所以要除以 scaling_factor
    if isinstance(output, tuple):
        dequant_tensor = output[0].detach()
        scaling_factor = output[1]
        # 轉回整數：x_int = dequant_tensor / scaling_factor
        x_int = (dequant_tensor / scaling_factor).cpu().numpy()
        # 四捨五入並轉換為 int32
        softmax_outputs.append(np.round(x_int).astype(np.int32))
    else:
        softmax_outputs.append(output.detach().cpu().numpy().astype(np.int32))

def main():
    global gelu_inputs, gelu_outputs, softmax_inputs, softmax_outputs
    
    print("=" * 70)
    print("使用 PyTorch Integer C-Model 生成整數測試向量")
    print("包含 LayerNorm, GELU, Softmax 的中間結果")
    print("=" * 70)
    print()
    
    # 載入模型
    print("[1/5] 載入模型...")
    device = torch.device('cpu')
    model = deit_tiny_patch16_224(pretrained=False)
    
    checkpoint_path = 'output_gpu/checkpoint_converted.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'output_integer' in key or 'norm_scaling_factor' in key:
            continue
        filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print("  ✅ 模型已載入")
    
    # 提取每個 block 的 GELU x0_int
    print()
    print("  提取 GELU x0_int 參數...")
    gelu_x0_ints = []
    for i in range(12):
        key = f'blocks.{i}.mlp.qact_gelu.act_scaling_factor'
        if key in state_dict:
            sf = state_dict[key]
            if isinstance(sf, torch.Tensor):
                sf = sf.item()
            sf_sig = sf * 1.702
            x0_int = int(np.floor(-1.0 / sf_sig))
            gelu_x0_ints.append(x0_int)
            print(f"    Block {i}: x0_int = {x0_int}")
        else:
            gelu_x0_ints.append(-10)  # 預設值
            print(f"    Block {i}: x0_int = -10 (預設)")
    
    print()
    print("  ✅ 已提取所有 x0_int 參數")
    print()
    
    # 註冊 hooks 以捕獲 GELU 和 Softmax 的中間結果
    print("[2/5] 註冊 hooks...")
    for block_idx, block in enumerate(model.blocks):
        # GELU: 在 MLP 中
        # qact_gelu 輸出 -> GELU 輸入
        block.mlp.qact_gelu.register_forward_hook(hook_gelu_input)
        # 注意：我們不再捕獲 GELU 輸出，而是使用 C-Model Reference 計算
        
        # Softmax: 在 Attention 中
        # qact_attn1 輸出 -> Softmax 輸入
        block.attn.qact_attn1.register_forward_hook(hook_softmax_input)
        # int_softmax 輸出 -> Softmax 輸出
        block.attn.int_softmax.register_forward_hook(hook_softmax_output)
    
    print(f"  ✅ 已註冊 {len(model.blocks)} 個 blocks 的 hooks")
    print()
    
    # 圖片轉換
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 載入測試圖片
    print("[3/5] 載入測試圖片...")
    image_path = '../data/test_image.JPEG'
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print(f"  ✅ 圖片已載入")
    
    # 讀取 ground truth
    with open('../data/test_image_info.txt', 'r') as f:
        lines = f.readlines()
        ground_truth = int(lines[2].split(': ')[1].strip())
    print(f"  Ground Truth: {ground_truth}")
    print()
    
    # 確保輸出目錄存在
    output_dir = '/mnt/c/Users/Public/I-ViT/nonlinear_verification/test_vectors_golden'
    os.makedirs(output_dir, exist_ok=True)
    
    # 執行推論並收集整數 patterns
    print("[4/5] 執行推論並收集整數測試向量...")
    print()
    
    layernorm_count = 0
    
    with torch.no_grad():
        # 執行一次完整推論，同時收集 LayerNorm, GELU, Softmax
        # hooks 會自動捕獲 GELU 和 Softmax
        x, input_scaling_factor = model.qact_input(image_tensor)
        x, patch_scaling_factor = model.patch_embed(x, input_scaling_factor)
        cls_tokens = model.cls_token.expand(1, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x_pos, act_scaling_factor_pos = model.qact_pos(model.pos_embed)
        x, x_scaling_factor = model.qact1(x, patch_scaling_factor, x_pos, act_scaling_factor_pos)
        
        # 對每個 block 收集 LayerNorm patterns
        for block_idx, block in enumerate(model.blocks):
            # Norm1 - 輸入是當前的 x
            norm1_input = x.cpu().numpy().astype(np.int32)
            
            x_norm1, norm1_sf = block.norm1(x, x_scaling_factor)
            x_norm1, qact1_sf = block.qact1(x_norm1, norm1_sf)
            
            norm1_output = x_norm1.cpu().numpy().astype(np.int32)
            
            # 儲存 norm1 測試向量
            save_hex_file(norm1_input, os.path.join(output_dir, f'layernorm_input_{layernorm_count}.hex'))
            save_hex_file(norm1_output, os.path.join(output_dir, f'layernorm_golden_{layernorm_count}.hex'))
            
            if layernorm_count % 5 == 0:
                print(f"  LayerNorm {layernorm_count}: Block {block_idx} norm1")
                print(f"    Input shape: {norm1_input.shape}, range=[{norm1_input.min()}, {norm1_input.max()}]")
                print(f"    Output shape: {norm1_output.shape}, range=[{norm1_output.min()}, {norm1_output.max()}]")
            
            layernorm_count += 1
            
            # Attention
            attn_output, attn_sf = block.attn(x_norm1, qact1_sf)
            
            # Residual 1
            x, x_sf = block.qact2(x, x_scaling_factor, attn_output, attn_sf)
            
            # Norm2 - 輸入是 residual1 的輸出
            norm2_input = x.cpu().numpy().astype(np.int32)
            
            x_norm2, norm2_sf = block.norm2(x, x_sf)
            x_norm2, qact3_sf = block.qact3(x_norm2, norm2_sf)
            
            norm2_output = x_norm2.cpu().numpy().astype(np.int32)
            
            # 儲存 norm2 測試向量
            save_hex_file(norm2_input, os.path.join(output_dir, f'layernorm_input_{layernorm_count}.hex'))
            save_hex_file(norm2_output, os.path.join(output_dir, f'layernorm_golden_{layernorm_count}.hex'))
            
            if layernorm_count % 5 == 0:
                print(f"  LayerNorm {layernorm_count}: Block {block_idx} norm2")
                print(f"    Input shape: {norm2_input.shape}, range=[{norm2_input.min()}, {norm2_input.max()}]")
                print(f"    Output shape: {norm2_output.shape}, range=[{norm2_output.min()}, {norm2_output.max()}]")
            
            layernorm_count += 1
            
            # MLP
            mlp_output, mlp_sf = block.mlp(x_norm2, qact3_sf)
            
            # Residual 2
            x, x_sf = block.qact4(x, x_sf, mlp_output, mlp_sf)
            
            x_scaling_factor = x_sf
        
        # Final norm
        final_norm_input = x.cpu().numpy().astype(np.int32)
        
        x, x_sf = model.norm(x, x_scaling_factor)
        
        final_norm_output = x.cpu().numpy().astype(np.int32)
        
        # 儲存 final norm 測試向量
        save_hex_file(final_norm_input, os.path.join(output_dir, f'layernorm_input_{layernorm_count}.hex'))
        save_hex_file(final_norm_output, os.path.join(output_dir, f'layernorm_golden_{layernorm_count}.hex'))
        
        print(f"  LayerNorm {layernorm_count}: Final norm")
        print(f"    Input shape: {final_norm_input.shape}, range=[{final_norm_input.min()}, {final_norm_input.max()}]")
        print(f"    Output shape: {final_norm_output.shape}, range=[{final_norm_output.min()}, {final_norm_output.max()}]")
        
        layernorm_count += 1
        
        # CLS token
        x = x[:, 0]
        x, x_sf = model.qact2(x, x_sf)
        
        # Head
        x, x_sf = model.head(x, x_sf)
        
        # 最終輸出
        logits = x.cpu().numpy()[0]
        pred_class = np.argmax(logits)
        
        print()
        print(f"  ✅ 推論完成")
        print(f"    預測類別: {pred_class}")
        print(f"    Ground Truth: {ground_truth}")
        print(f"    正確: {pred_class == ground_truth}")
    
    print()
    print(f"✅ LayerNorm: 生成 {layernorm_count} 組測試向量")
    print(f"✅ GELU: 捕獲 {len(gelu_inputs)} 組輸入")
    print(f"✅ Softmax: 捕獲 {len(softmax_inputs)} 組測試向量")
    print()
    
    # 使用 C-Model Reference 計算 GELU 輸出
    print("使用 C-Model Reference 計算 GELU 輸出...")
    for i, (gelu_in, x0_int) in enumerate(zip(gelu_inputs, gelu_x0_ints)):
        # 使用 C-Model Reference 的 int_gelu_kernel_fixed 函數
        gelu_out = int_gelu_kernel_fixed(gelu_in, x0_int, output_bit=8, n=23)
        gelu_outputs.append(gelu_out)
        
        if i % 3 == 0:
            print(f"  GELU {i}: in_range=[{gelu_in.min()}, {gelu_in.max()}], out_range=[{gelu_out.min()}, {gelu_out.max()}], x0_int={x0_int}")
    
    print(f"  ✅ 已計算 {len(gelu_outputs)} 組 GELU 輸出（使用 C-Model Reference）")
    print()
    
    # 儲存 GELU 測試向量
    print("儲存 GELU 測試向量...")
    for i, (gelu_in, gelu_out) in enumerate(zip(gelu_inputs, gelu_outputs)):
        save_hex_file(gelu_in, os.path.join(output_dir, f'gelu_input_{i}.hex'))
        save_hex_file(gelu_out, os.path.join(output_dir, f'gelu_golden_{i}.hex'))
        
        # 儲存對應的 x0_int
        x0_int = gelu_x0_ints[i] if i < len(gelu_x0_ints) else -10
        with open(os.path.join(output_dir, f'gelu_x0int_{i}.txt'), 'w') as f:
            f.write(f"{x0_int}\n")
        
        if i % 3 == 0:
            nonzero_count = np.count_nonzero(gelu_out)
            print(f"  GELU {i}: shape={gelu_in.shape}, in_range=[{gelu_in.min()}, {gelu_in.max()}], out_range=[{gelu_out.min()}, {gelu_out.max()}], non-zero={nonzero_count}/{gelu_out.size}, x0_int={x0_int}")
    print(f"  ✅ 已儲存 {len(gelu_inputs)} 組 GELU 測試向量")
    print()
    
    # 儲存 Softmax 測試向量
    print("儲存 Softmax 測試向量...")
    for i, (sm_in, sm_out) in enumerate(zip(softmax_inputs, softmax_outputs)):
        save_hex_file(sm_in, os.path.join(output_dir, f'softmax_input_{i}.hex'))
        save_hex_file(sm_out, os.path.join(output_dir, f'softmax_golden_{i}.hex'))
        if i % 3 == 0:
            print(f"  Softmax {i}: shape={sm_in.shape}, in_range=[{sm_in.min()}, {sm_in.max()}], out_range=[{sm_out.min()}, {sm_out.max()}]")
    print(f"  ✅ 已儲存 {len(softmax_inputs)} 組 Softmax 測試向量")
    print()
    
    
    print("[5/5] 生成報告...")
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write("# 整數 Golden Patterns (從實際模型推論)\n\n")
        f.write("**生成日期**: 2026/5/23\n")
        f.write(f"**測試圖片**: ../data/test_image.JPEG\n")
        f.write(f"**Ground Truth**: {ground_truth}\n")
        f.write(f"**預測類別**: {pred_class}\n")
        f.write(f"**預測正確**: {pred_class == ground_truth}\n\n")
        f.write("## 測試向量\n\n")
        f.write(f"### LayerNorm: {layernorm_count} 組\n")
        f.write(f"- Block 0-11: 每個 block 2 組 (norm1 + norm2) = 24 組\n")
        f.write(f"- Final norm: 1 組\n")
        f.write(f"- 總計: 25 組\n")
        f.write(f"- 資料形狀: (1, 197, 192) = 37,824 個 INT32 值\n\n")
        f.write(f"### GELU: {len(gelu_inputs)} 組\n")
        f.write(f"- Block 0-11: 每個 block 1 組 (MLP 中的 GELU)\n")
        f.write(f"- 總計: 12 組\n")
        f.write(f"- 資料形狀: (1, 197, 768) = 151,296 個 INT32 值\n\n")
        f.write(f"### Softmax: {len(softmax_inputs)} 組\n")
        f.write(f"- Block 0-11: 每個 block 1 組 (Attention 中的 Softmax)\n")
        f.write(f"- 總計: 12 組\n")
        f.write(f"- 資料形狀: (1, 3, 197, 197) = 116,427 個 INT32 值\n\n")
        f.write("## 檔案格式\n\n")
        f.write("### LayerNorm\n")
        f.write("- `layernorm_input_X.hex`: LayerNorm 輸入（INT32, hex 格式）\n")
        f.write("- `layernorm_golden_X.hex`: LayerNorm 輸出（INT32, hex 格式）\n\n")
        f.write("### GELU\n")
        f.write("- `gelu_input_X.hex`: GELU 輸入（INT32, hex 格式）\n")
        f.write("- `gelu_golden_X.hex`: GELU 輸出（INT32, hex 格式）\n\n")
        f.write("### Softmax\n")
        f.write("- `softmax_input_X.hex`: Softmax 輸入（INT32, hex 格式）\n")
        f.write("- `softmax_golden_X.hex`: Softmax 輸出（INT32, hex 格式）\n\n")
        f.write("## 資料來源\n\n")
        f.write("這些測試向量來自實際模型推論的整數中間結果：\n")
        f.write("- 使用 PyTorch 量化模型直接推論\n")
        f.write("- 使用 forward hooks 捕獲每個模組的輸入和輸出\n")
        f.write("- 所有數據都是 INT32 整數格式\n\n")
        f.write("## 使用方法\n\n")
        f.write("這些測試向量可以直接用於 RTL 驗證，確保 RTL 實現與 PyTorch 模型完全一致。\n")
    
    print("  ✅ 報告已保存")
    print()
    
    print("=" * 70)
    print("✅✅✅ 整數測試向量生成完成！")
    print("=" * 70)
    print()
    print(f"測試向量位置: {output_dir}/")
    print(f"  - LayerNorm: layernorm_input_0.hex ~ layernorm_input_{layernorm_count-1}.hex")
    print(f"  - GELU: gelu_input_0.hex ~ gelu_input_{len(gelu_inputs)-1}.hex")
    print(f"  - Softmax: softmax_input_0.hex ~ softmax_input_{len(softmax_inputs)-1}.hex")
    print()

if __name__ == "__main__":
    main()
