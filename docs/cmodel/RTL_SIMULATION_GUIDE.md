# RTL 模擬驗證指南

## 概述

本文檔說明如何使用 PyTorch 量化模型生成的 Golden Patterns 來驗證 C-Model 和 RTL 實作。

## 背景

### 為什麼不使用 TVM？

經過詳細測試，我們發現 TVM 0.9.0 的整數推理實作存在根本性問題：

1. **測試結果**：
   - PyTorch 量化模型：73.54% Top-1 準確率（1000 張圖片）
   - TVM 整數模型：0% 準確率（相同測試集）
   - TVM 輸出幾乎均勻分佈（最大概率 <2%）

2. **測試過的方案**：
   - ✗ 使用論文原始 checkpoint (`qat_calibrated.pth`) → TVM 仍然 0%
   - ✗ 使用 GPU 訓練的 checkpoint (`output_gpu/checkpoint.pth`) → TVM 仍然 0%
   - ✗ 不使用 calibrated scales（只用 checkpoint 中的 scales） → TVM 仍然 0%
   - ✓ PyTorch 量化模型在所有情況下都正常工作

3. **結論**：
   - TVM 的整數推理實作有根本性 bug，無法修復
   - 改用 PyTorch 量化模型作為 Golden Reference
   - C-Model 需要重新驗證（之前是用 TVM 驗證的）

## 工作流程

```
PyTorch 量化模型 (已驗證 73.54% 準確率)
    ↓
提取 Golden Patterns (extract_golden_patterns.py)
    ↓
Golden Patterns (.npz 文件)
    ├─→ 驗證 C-Model (verify_cmodel.py)
    │       ↓
    │   C-Model 正確性確認
    │       ↓
    └─→ 生成 RTL 測試向量
            ↓
        RTL 模擬驗證
            ↓
        硬體實作
```

## 步驟 1: 提取 Golden Patterns

### 1.1 執行提取腳本

```bash
cd I-ViT
python extract_golden_patterns.py
```

### 1.2 輸出文件

執行後會在 `golden_patterns/` 目錄生成以下文件：

1. **golden_patterns.npz**
   - 包含所有中間層的輸入/輸出數據
   - 包含 scaling factors
   - 包含最終預測結果

2. **golden_patterns_report.txt**
   - 可讀的數據摘要
   - 每層的 shape、dtype、數值範圍
   - 預測結果和 Top-5 類別

3. **model_weights.npz**
   - 所有模型參數（權重和偏置）
   - 用於 C-Model 和 RTL 的參數載入

4. **model_weights_report.txt**
   - 權重參數的摘要信息

### 1.3 提取的數據層

Golden Patterns 包含以下關鍵層的數據：

#### 輸入處理
- `input_image`: 原始輸入圖片（歸一化後）
- `0_input_quant_output`: 量化後的輸入
- `1_patch_embed_output`: Patch Embedding 輸出
- `2_pos_embed_output`: Position Embedding 輸出

#### Transformer Block（每個 block 包含）
- `3_blockN_norm1_output`: 第一個 LayerNorm 輸出
- `4_blockN_attn_qkv_output`: QKV Projection 輸出
- `5_blockN_attn_qk_output`: Q @ K^T 矩陣乘法輸出
- `6_blockN_attn_softmax_output`: Softmax 輸出
- `7_blockN_attn_out_output`: Attention @ V 輸出
- `8_blockN_attn_proj_output`: Attention Projection 輸出
- `9_blockN_norm2_output`: 第二個 LayerNorm 輸出
- `10_blockN_mlp_fc1_output`: MLP 第一層輸出
- `11_blockN_mlp_gelu_output`: GELU 激活輸出
- `12_blockN_mlp_fc2_output`: MLP 第二層輸出

#### 輸出處理
- `13_final_norm_output`: 最終 LayerNorm 輸出
- `14_head_output`: 分類頭輸出
- `final_output`: 最終 logits
- `prediction`: 預測結果（類別、概率、Top-5）

## 步驟 2: 驗證 C-Model

### 2.1 C-Model 參考實作

C-Model 位於 `cmodel_rtl_reference/` 目錄：

1. **linear_cmodel_reference.py**
   - `int_dense_kernel`: 全連接層（FC1, FC2, QKV, Projection）
   - `int_matmul_kernel`: 矩陣乘法（Q @ K^T, Attention @ V）

2. **nonlinear_cmodel_reference.py**
   - `int_layer_norm_fixed`: LayerNorm
   - `int_gelu_kernel_fixed`: GELU 激活函數
   - `int_softmax_kernel_fixed`: Softmax
   - `int_exp_shift_kernel_standard`: 指數運算（Softmax 和 GELU 使用）

### 2.2 驗證流程

```bash
cd I-ViT
python verify_cmodel.py
```

**注意**：目前 `verify_cmodel.py` 只提供驗證框架，需要根據實際的 Golden Patterns 結構來實作具體的驗證邏輯。

### 2.3 驗證步驟

對於每一層：

1. 從 `golden_patterns.npz` 提取該層的輸入
2. 從 `model_weights.npz` 提取該層的權重和偏置
3. 使用 C-Model 函數計算輸出
4. 與 Golden Patterns 中的輸出進行 Bit-Exact 比較
5. 記錄差異（如果有）

### 2.4 預期結果

- **完全匹配（Bit-Exact）**：C-Model 實作正確
- **有差異**：需要檢查 C-Model 實作，可能的原因：
  - 整數除法的截斷方向不同（Floor vs. Truncate towards zero）
  - 位移操作的符號處理不同（Logical vs. Arithmetic shift）
  - 溢位處理不同
  - 數據型別轉換不正確

## 步驟 3: 生成 RTL 測試向量

### 3.1 測試向量格式

RTL 模擬需要以下測試向量：

1. **輸入向量**
   - 量化後的圖片數據（int8）
   - Position embedding（int8/int32）
   - CLS token（int8/int32）

2. **權重向量**
   - 所有層的權重（int8）
   - 所有層的偏置（int32）
   - Scaling factors

3. **Golden 輸出向量**
   - 每一層的預期輸出
   - 最終分類結果

### 3.2 向量生成腳本（待實作）

```python
# TODO: 創建 generate_rtl_vectors.py
# 功能：
# 1. 讀取 golden_patterns.npz 和 model_weights.npz
# 2. 轉換為 RTL 模擬器可讀的格式（如 .hex, .mem, .txt）
# 3. 生成測試腳本（如 SystemVerilog testbench）
```

### 3.3 建議的向量格式

#### 方案 A: 十六進制文件（.hex）
```
# input_image.hex
00 FF 7F ...  # 每個 byte 一個 int8 值
```

#### 方案 B: 記憶體初始化文件（.mem）
```
// input_image.mem
@0000 00
@0001 FF
@0002 7F
...
```

#### 方案 C: SystemVerilog 陣列
```systemverilog
// test_vectors.sv
parameter int8_t input_image[0:150527] = '{
    8'h00, 8'hFF, 8'h7F, ...
};
```

## 步驟 4: RTL 模擬驗證

### 4.1 RTL 實作建議

參考 `cmodel_rtl_reference/README.md` 中的硬體實作建議：

1. **位寬設計**
   - 輸入/權重：8-bit signed
   - MAC 累加器：32-bit signed
   - LayerNorm variance：32-bit unsigned
   - LayerNorm 中間計算：64-bit

2. **除法和位移**
   - 注意 Python `//` 與 C/Verilog `/` 的差異
   - 使用 Arithmetic Shift Right 處理有號數
   - 使用 Logical Shift Right 處理無號數

3. **溢位處理**
   - LayerNorm 的 mean 計算會有 int16 溢位（模擬硬體行為）
   - 其他地方使用足夠的位寬避免溢位

### 4.2 驗證策略

#### 逐層驗證
1. 先驗證單個運算單元（MAC, LayerNorm, GELU, Softmax）
2. 再驗證完整的層（Attention, MLP, Block）
3. 最後驗證整個模型

#### 數值比較
- 使用 Golden Patterns 作為參考
- 要求 Bit-Exact 匹配（整數運算不應有誤差）
- 如果有差異，回溯到 C-Model 進行調試

### 4.3 Testbench 結構

```systemverilog
module tb_vit_layer;
    // 1. 載入測試向量
    initial begin
        $readmemh("input_image.hex", input_mem);
        $readmemh("weights.hex", weight_mem);
        $readmemh("golden_output.hex", golden_mem);
    end
    
    // 2. 執行 RTL
    initial begin
        // 餵入輸入和權重
        // 執行計算
        // 讀取輸出
    end
    
    // 3. 比較結果
    initial begin
        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            if (output_mem[i] !== golden_mem[i]) begin
                $error("Mismatch at index %d: got %h, expected %h",
                       i, output_mem[i], golden_mem[i]);
            end
        end
        $display("Verification PASSED!");
    end
endmodule
```

## 步驟 5: 調試和優化

### 5.1 常見問題

1. **數值不匹配**
   - 檢查數據型別（signed vs. unsigned）
   - 檢查位移方向（left vs. right）
   - 檢查除法截斷方向

2. **溢位錯誤**
   - 增加中間暫存器位寬
   - 檢查累加器大小

3. **時序問題**
   - 增加 pipeline stages
   - 調整時鐘頻率

### 5.2 性能優化

1. **並行化**
   - Multi-head attention 可以並行計算
   - MAC 陣列可以展開

2. **記憶體優化**
   - 使用 on-chip SRAM 存儲權重
   - 使用 double buffering 減少等待

3. **精度優化**
   - 如果需要更高精度，可以使用 16-bit 量化
   - 可以使用混合精度（部分層用 16-bit）

## 附錄 A: 文件清單

### 已完成的文件

1. **模型和訓練**
   - `I-ViT/train_gpu.py`: GPU 訓練腳本
   - `I-ViT/test_pytorch_quick.py`: PyTorch 模型驗證
   - `output_gpu/checkpoint.pth`: 訓練好的模型（73.54% 準確率）

2. **C-Model 參考**
   - `cmodel_rtl_reference/linear_cmodel_reference.py`: 線性層 C-Model
   - `cmodel_rtl_reference/nonlinear_cmodel_reference.py`: 非線性層 C-Model
   - `cmodel_rtl_reference/README.md`: C-Model 說明文檔

3. **Golden Patterns**
   - `I-ViT/extract_golden_patterns.py`: 提取 Golden Patterns
   - `I-ViT/verify_cmodel.py`: 驗證 C-Model（框架）

4. **文檔**
   - `docs/cmodel/RTL_SIMULATION_GUIDE.md`: 本文檔
   - `docs/cmodel/HARDWARE_CMODEL_GUIDE.md`: 硬體實作指南
   - `docs/analysis/TVM_COMPARISON_ANALYSIS.md`: TVM 問題分析

### 待完成的文件

1. **RTL 測試向量生成**
   - `I-ViT/generate_rtl_vectors.py`: 生成 RTL 測試向量
   - 輸出格式：.hex, .mem, 或 SystemVerilog 陣列

2. **C-Model 完整驗證**
   - 完善 `verify_cmodel.py` 的驗證邏輯
   - 逐層驗證所有運算

3. **RTL 實作**
   - Verilog/SystemVerilog 模組
   - Testbench
   - 綜合和時序分析腳本

## 附錄 B: 參考資料

### 論文
- I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference
- 論文提供的預訓練模型：`I-ViT/checkpoints/qat_calibrated.pth`

### 模型架構
- DeiT-Tiny: 192 dim, 12 layers, 3 heads
- 參數量：約 5.7M
- 輸入：224x224 RGB 圖片
- 輸出：1000 類 ImageNet 分類

### 量化方案
- 權重：int8 對稱量化
- 激活：int8 對稱量化
- 累加器：int32
- 特殊層：LayerNorm, GELU, Softmax 使用整數近似

## 附錄 C: 聯絡信息

如有問題，請參考：
- C-Model 實作：`cmodel_rtl_reference/README.md`
- TVM 問題分析：`docs/analysis/TVM_COMPARISON_ANALYSIS.md`
- 訓練和測試：`GPU_TRAINING_README.md`
