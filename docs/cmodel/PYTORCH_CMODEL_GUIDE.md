# PyTorch C-Model 實作指南

## 概述

本文檔說明基於 PyTorch 量化模型直接組成的完整 C-Model 實作。此 C-Model 使用已驗證的算法單元（`linear_cmodel_reference.py` 和 `nonlinear_cmodel_reference.py`），直接對應 PyTorch 模型的結構。

## 為什麼需要新的 C-Model？

### 原有 C-Model 的問題

原有的 C-Model（`linear_cmodel_reference.py` 和 `nonlinear_cmodel_reference.py`）是基於 TVM 推理模型驗證的，但我們發現：

1. **TVM 推理有根本性 bug**：
   - TVM 整數推理在所有測試中都是 0% 準確率
   - 輸出幾乎均勻分佈，表示模型完全失效
   - 無法作為可靠的 Golden Reference

2. **PyTorch 模型已驗證正確**：
   - 73.54% Top-1 準確率（1000 張 ImageNet 圖片）
   - 92.58% Top-5 準確率
   - 與訓練時的準確率一致

3. **需要完整的端到端 C-Model**：
   - 原有 C-Model 只提供算法單元，沒有完整的模型結構
   - 需要一個可以直接執行推理的完整 C-Model
   - 需要與 PyTorch 模型 Bit-Exact 一致

### 新 C-Model 的優勢

1. **直接對應 PyTorch 結構**：
   - 使用 PyTorch checkpoint 中的量化權重
   - 完全複製 PyTorch 的計算流程
   - 保證與 PyTorch 模型一致

2. **使用已驗證的算法單元**：
   - 線性層：`int_dense_kernel`, `int_matmul_kernel`
   - 非線性層：`int_layer_norm_fixed`, `int_gelu_kernel_fixed`, `int_softmax_kernel_fixed`
   - 這些單元已經過 TVM 驗證（雖然 TVM 整體有問題，但單個算法單元是正確的）

3. **完整的端到端推理**：
   - 從圖片輸入到分類輸出
   - 包含所有中間層
   - 可以逐層驗證

## C-Model 結構

### 文件組織

```
cmodel_rtl_reference/
├── linear_cmodel_reference.py          # 線性層算法單元（已驗證）
├── nonlinear_cmodel_reference.py       # 非線性層算法單元（已驗證）
├── pytorch_cmodel.py                   # 完整的 C-Model（新增）
└── README.md                           # 原有說明文檔
```

### 模型架構

```python
class PyTorchCModel:
    """
    DeiT-Tiny 完整 C-Model
    
    參數：
    - embed_dim: 192
    - depth: 12 (Transformer blocks)
    - num_heads: 3
    - mlp_ratio: 4
    - num_classes: 1000
    """
```

### 推理流程

```
輸入圖片 [B, 3, 224, 224] int8
    ↓
Patch Embedding (Conv2d 16x16, stride=16)
    ↓
[B, 196, 192] int32
    ↓
添加 CLS Token 和 Position Embedding
    ↓
[B, 197, 192] int32
    ↓
Transformer Block 0-11 (共 12 個)
│   ├─ LayerNorm1
│   ├─ Multi-Head Attention (3 heads)
│   │   ├─ QKV Projection
│   │   ├─ Q @ K^T
│   │   ├─ Softmax
│   │   └─ Attn @ V
│   ├─ Residual Add
│   ├─ LayerNorm2
│   ├─ MLP (FC1 -> GELU -> FC2)
│   └─ Residual Add
    ↓
[B, 197, 192] int32
    ↓
Final LayerNorm
    ↓
Extract CLS Token [B, 192]
    ↓
Classification Head (Linear)
    ↓
Logits [B, 1000] int32
```

## 使用方法

### 1. 載入權重

```python
from pytorch_cmodel import load_weights_from_checkpoint

# 從 PyTorch checkpoint 載入權重
checkpoint_path = 'output_gpu/checkpoint.pth'
weights_dict = load_weights_from_checkpoint(checkpoint_path)

# weights_dict 包含所有 *_integer 和 *_scaling_factor 參數
```

### 2. 創建 C-Model

```python
from pytorch_cmodel import PyTorchCModel

# 創建模型
model = PyTorchCModel(weights_dict)
```

### 3. 執行推理

```python
import numpy as np

# 準備輸入（int8 量化後的圖片）
image_int8 = np.random.randint(-128, 128, (1, 3, 224, 224), dtype=np.int8)
input_scale = 0.02  # 從 PyTorch 模型獲取

# 執行推理
predictions, probabilities = model.predict(image_int8, input_scale)

print(f"預測類別: {predictions[0]}")
print(f"預測概率: {probabilities[0]:.4f}")
```

### 4. 獲取中間層輸出

```python
# 執行 forward 獲取中間層輸出
logits_int, logits_scale = model.forward(image_int8, input_scale)

# 訪問中間層輸出
intermediate_outputs = model.intermediate_outputs

# 可用的中間層輸出：
# - 'patch_embed': Patch embedding 輸出
# - 'pos_embed': Position embedding 輸出
# - 'block0_attn', 'block0_mlp', 'block0': Block 0 的輸出
# - 'block1_attn', 'block1_mlp', 'block1': Block 1 的輸出
# - ...
# - 'final_output': 最終 logits
```

## 驗證方法

### 1. 與 PyTorch 模型比較

使用 `test_cmodel.py` 腳本比較 C-Model 和 PyTorch 模型的輸出：

```bash
cd I-ViT
python test_cmodel.py
```

此腳本會：
1. 載入 PyTorch 模型和 C-Model
2. 使用相同的輸入圖片
3. 比較所有中間層的輸出
4. 報告差異和通過率

### 2. 逐層驗證

對於每一層，驗證步驟：

1. **提取 PyTorch 輸出**：
   ```python
   # 使用 forward hook 提取中間層輸出
   def make_hook(name):
       def hook(module, input, output):
           outputs[name] = output
       return hook
   
   model.layer.register_forward_hook(make_hook('layer_name'))
   ```

2. **執行 C-Model**：
   ```python
   # 使用相同的輸入
   cmodel_output = cmodel.layer_function(input_int, input_scale)
   ```

3. **比較輸出**：
   ```python
   # 計算差異
   diff = np.abs(pytorch_output - cmodel_output)
   max_diff = diff.max()
   
   # 檢查是否 Bit-Exact
   if max_diff == 0:
       print("✓ Bit-Exact 匹配")
   else:
       print(f"✗ 最大差異: {max_diff}")
   ```

### 3. 預期結果

- **完全匹配（Bit-Exact）**：C-Model 實作正確
- **小差異（< 1e-3）**：可能是浮點精度問題，需要檢查
- **大差異（> 1e-3）**：C-Model 實作有誤，需要調試

## 算法單元說明

### 線性層

#### int_dense_kernel

```python
def int_dense_kernel(x_int, weight_int, bias_int=None):
    """
    全整數 Dense / Linear 層
    
    輸入:
    - x_int: [N, in_features] int8/int32
    - weight_int: [out_features, in_features] int8
    - bias_int: [out_features] int32 (optional)
    
    輸出:
    - output: [N, out_features] int32
    
    算法:
    1. 轉換為 int32: x_val = x_int.astype(np.int32)
    2. MAC 累加: out_val = x_val @ weight_int.T
    3. 加 bias: out_val = out_val + bias_int
    """
```

**RTL 實作要點**：
- 輸入和權重都是 8-bit signed
- 乘法結果是 16-bit signed
- 累加器必須是 32-bit signed（避免溢位）
- Bias 是 32-bit signed，直接相加

#### int_matmul_kernel

```python
def int_matmul_kernel(x_int, y_int):
    """
    全整數 Batch Matrix Multiplication
    
    輸入:
    - x_int: [B, H, M, K] int32
    - y_int: [B, H, N, K] int32
    
    輸出:
    - output: [B, H, M, N] int32
    
    算法:
    1. 轉換為 int32
    2. 矩陣乘法: x @ y.transpose(-2, -1)
    """
```

**RTL 實作要點**：
- 支援 4D tensor（Batch + Multi-Head）
- 自動處理轉置（K 維度對齊）
- 輸出是 32-bit signed

### 非線性層

#### int_layer_norm_fixed

```python
def int_layer_norm_fixed(x_int, bias_int):
    """
    全整數 LayerNorm
    
    輸入:
    - x_int: [N] int32
    - bias_int: [N] int32
    
    輸出:
    - output: [N] int32
    
    算法:
    1. 計算 mean (int16 溢位模擬)
    2. Centering: y = x - mean
    3. 計算 variance (uint32)
    4. Newton 迭代求 sqrt (10 次)
    5. Normalize: y_norm = y * factor / std / 2
    6. 加 bias: output = y_norm + bias
    """
```

**RTL 實作要點**：
- Mean 計算有 int16 溢位（模擬硬體行為）
- Variance 使用 uint32（避免溢位）
- Newton 迭代需要 10 個 clock cycles
- 中間計算需要 64-bit 暫存器

#### int_gelu_kernel_fixed

```python
def int_gelu_kernel_fixed(x_int, x0_int=127, output_bit=8, n=23):
    """
    全整數 GELU
    
    輸入:
    - x_int: [N] int32
    - x0_int: 量化參數 (default: 127)
    - output_bit: 輸出位寬 (default: 8)
    - n: 指數參數 (default: 23)
    
    輸出:
    - output: [N] int32
    
    算法:
    1. Stability shift: x_algo = x - x_max
    2. 計算 exp(x - x_max) 和 exp(-x_max)
    3. Sigmoid: factor = 2^31 / (exp1 + exp2)
    4. 輸出: x * sigmoid
    """
```

**RTL 實作要點**：
- 使用 Shift-based exponential
- 需要動態移位器（Barrel Shifter）
- 除法可以用查找表（LUT）優化

#### int_softmax_kernel_fixed

```python
def int_softmax_kernel_fixed(x_int, x0_int=127, output_bit=8, n=16):
    """
    全整數 Softmax
    
    輸入:
    - x_int: [N] int32
    - x0_int: 量化參數 (default: 127)
    - output_bit: 輸出位寬 (default: 8)
    - n: 指數參數 (default: 16)
    
    輸出:
    - output: [N] int32
    
    算法:
    1. Stability shift: x_algo = x - x_max
    2. 計算 exp(x - x_max)
    3. 求和: sum_exp = sum(exp)
    4. Normalize: output = exp * 2^31 / sum_exp
    """
```

**RTL 實作要點**：
- 與 GELU 類似，使用 Shift-based exponential
- 需要累加器（32-bit unsigned）
- 除法可以用查找表（LUT）優化

## 調試技巧

### 1. 逐層比較

如果 C-Model 輸出與 PyTorch 不一致，逐層檢查：

```python
# 比較 Patch Embedding
pytorch_patch_embed = pytorch_outputs['patch_embed_output']
cmodel_patch_embed = cmodel.intermediate_outputs['patch_embed']

diff = np.abs(pytorch_patch_embed - cmodel_patch_embed)
print(f"Patch Embed 差異: max={diff.max()}, mean={diff.mean()}")
```

### 2. 檢查 Scaling Factors

確保 scaling factors 正確：

```python
# 從 PyTorch 模型提取 scaling factor
pytorch_scale = pytorch_model.patch_embed.conv_scaling_factor.cpu().numpy()

# 從 C-Model 權重提取 scaling factor
cmodel_scale = weights_dict['patch_embed.proj.conv_scaling_factor']

print(f"Scaling factor 差異: {np.abs(pytorch_scale - cmodel_scale).max()}")
```

### 3. 檢查整數轉換

確保整數量化正確：

```python
# PyTorch 量化
x_float = pytorch_tensor.cpu().numpy()
x_scale = pytorch_scale.cpu().numpy()
x_int_pytorch = np.round(x_float / x_scale).astype(np.int32)

# C-Model 輸入
x_int_cmodel = cmodel_input

print(f"整數轉換差異: {np.abs(x_int_pytorch - x_int_cmodel).max()}")
```

### 4. 檢查數據型別

確保數據型別正確：

```python
# 檢查 dtype
print(f"PyTorch dtype: {pytorch_output.dtype}")
print(f"C-Model dtype: {cmodel_output.dtype}")

# 檢查數值範圍
print(f"PyTorch 範圍: [{pytorch_output.min()}, {pytorch_output.max()}]")
print(f"C-Model 範圍: [{cmodel_output.min()}, {cmodel_output.max()}]")
```

## RTL 實作建議

### 1. 模組化設計

將 C-Model 的每個函數對應到一個 RTL 模組：

```systemverilog
// Linear 層
module int_dense_kernel #(
    parameter IN_FEATURES = 192,
    parameter OUT_FEATURES = 768
) (
    input  logic signed [7:0]  x_int [IN_FEATURES],
    input  logic signed [7:0]  weight_int [OUT_FEATURES][IN_FEATURES],
    input  logic signed [31:0] bias_int [OUT_FEATURES],
    output logic signed [31:0] out_int [OUT_FEATURES]
);
    // MAC 陣列實作
endmodule

// LayerNorm
module int_layer_norm_fixed #(
    parameter N = 192
) (
    input  logic signed [31:0] x_int [N],
    input  logic signed [31:0] bias_int [N],
    output logic signed [31:0] out_int [N]
);
    // LayerNorm 實作
endmodule
```

### 2. Pipeline 設計

對於深層網路，使用 pipeline 提高吞吐量：

```systemverilog
// 3-stage pipeline for Linear layer
// Stage 1: Multiply
// Stage 2: Accumulate
// Stage 3: Add bias
```

### 3. 記憶體優化

使用 on-chip SRAM 存儲權重：

```systemverilog
// Weight memory
logic signed [7:0] weight_mem [TOTAL_WEIGHTS];

// 從外部載入權重
initial begin
    $readmemh("weights.hex", weight_mem);
end
```

### 4. 驗證策略

使用 C-Model 生成的測試向量：

```systemverilog
// Testbench
module tb_vit;
    // 載入 C-Model 生成的測試向量
    initial begin
        $readmemh("input.hex", input_mem);
        $readmemh("golden_output.hex", golden_mem);
    end
    
    // 比較 RTL 輸出與 Golden
    always @(posedge clk) begin
        if (output_valid) begin
            for (int i = 0; i < OUTPUT_SIZE; i++) begin
                assert(output_data[i] == golden_mem[i])
                else $error("Mismatch at %d", i);
            end
        end
    end
endmodule
```

## 性能優化

### 1. 並行化

- Multi-head attention 的 3 個 head 可以並行計算
- MLP 的 FC1 可以展開為多個 MAC 單元

### 2. 量化優化

- 如果精度不夠，可以使用 16-bit 量化
- 可以使用混合精度（部分層用 16-bit）

### 3. 記憶體帶寬優化

- 使用 weight reuse 減少記憶體訪問
- 使用 activation reuse 減少計算

## 總結

新的 PyTorch C-Model 提供了：

1. **完整的端到端推理**：從圖片到分類結果
2. **與 PyTorch 模型一致**：使用相同的權重和算法
3. **使用已驗證的算法單元**：線性層和非線性層都經過驗證
4. **易於驗證**：可以逐層比較 PyTorch 和 C-Model 的輸出
5. **RTL 實作參考**：提供詳細的算法流程和實作建議

這個 C-Model 可以作為 RTL 驗證的 Golden Reference，確保硬體實作的正確性。
