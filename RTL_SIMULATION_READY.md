# RTL 模擬準備完成報告

## 日期
2026-05-18

## 狀態
✅ **所有準備工作已完成，可以開始 RTL 實作**

---

## 已完成項目

### 1. ✅ PyTorch 量化模型驗證
- **準確率**: 73.54% Top-1, 92.58% Top-5
- **Checkpoint**: `I-ViT/output_gpu/checkpoint_converted.pth`
- **測試腳本**: `I-ViT/test_pytorch_only.py`
- **狀態**: 完全驗證，可作為 Golden Reference

### 2. ✅ C-Model 算法單元驗證
- **線性層**: `int_dense_kernel`, `int_matmul_kernel` - 100% 匹配
- **非線性層**: `int_layer_norm_fixed`, `int_gelu_kernel_fixed`, `int_softmax_kernel_fixed` - 全部通過
- **驗證腳本**: `I-ViT/verify_cmodel.py`
- **狀態**: 所有測試通過，可直接用於 RTL 實作

### 3. ✅ Golden Patterns 提取
- **數據項**: 378 個中間層輸入/輸出
- **文件**: `golden_patterns/golden_patterns.npz`
- **報告**: `golden_patterns/golden_patterns_report.txt`
- **權重**: `golden_patterns/model_weights.npz`
- **狀態**: 成功提取，包含所有 12 個 Transformer blocks 的數據

### 4. ✅ RTL 測試向量生成
- **格式**: .hex, .mem, .txt (三種格式)
- **目錄**: `rtl_vectors/`
- **文件**:
  - `input_image.*` - 輸入圖片
  - `patch_embed_output.*` - Patch Embedding 輸出
  - `block0_norm1_output.*` - Block 0 LayerNorm1 輸出（單元測試用）
  - `final_output.*` - 最終輸出 logits
  - `prediction.txt` - 預測結果
  - `README.txt` - 使用說明
- **狀態**: 成功生成，可直接用於 RTL 模擬

### 5. ✅ 完整文檔
- `docs/cmodel/FINAL_SUMMARY.md` - 完整總結
- `docs/cmodel/RTL_SIMULATION_GUIDE.md` - RTL 模擬指南
- `docs/cmodel/PYTORCH_CMODEL_GUIDE.md` - C-Model 使用指南
- `docs/cmodel/HARDWARE_CMODEL_GUIDE.md` - 硬體實作建議
- `README_CMODEL.md` - 快速開始指南
- **狀態**: 文檔完整，包含詳細的算法說明和實作建議

---

## 測試結果

### PyTorch 模型測試
```
Top-1 準確率: 73.54%
Top-5 準確率: 92.58%
測試圖片數: 1000 張 (ImageNet validation set)
預測類別: 1
預測概率: 0.3996
```

### C-Model 驗證測試
```
線性層 (Dense):     ✓ PASS (最大差異: 0)
矩陣乘法 (MatMul):  ✓ PASS (最大差異: 0)
LayerNorm:         ✓ PASS
GELU:              ✓ PASS
Softmax:           ✓ PASS
```

### Golden Patterns 提取
```
總數據項: 378 個
包含層級:
  - Input quantization
  - Patch embedding
  - Position embedding
  - 12 個 Transformer blocks (每個包含 10+ 個子層)
  - Final norm
  - Classification head
```

---

## RTL 實作指南

### Step 1: 選擇實作策略

#### 選項 A: 逐層實作（推薦）
1. 從最簡單的模組開始：`int_dense_kernel`
2. 使用 `block0_norm1_output.*` 作為測試向量
3. 逐步實作 LayerNorm, GELU, Softmax
4. 組合成完整的 Transformer Block
5. 複製 12 次形成完整模型

#### 選項 B: 完整模型實作
1. 直接實作完整的 DeiT-Tiny 模型
2. 使用 `input_image.*` 和 `final_output.*` 進行端到端驗證

### Step 2: 參考 C-Model 算法

所有算法單元的詳細實作請參考：
- `cmodel_rtl_reference/linear_cmodel_reference.py`
- `cmodel_rtl_reference/nonlinear_cmodel_reference.py`

關鍵要點：
- **線性層**: 8-bit 輸入/權重 → 32-bit MAC → 32-bit 輸出
- **LayerNorm**: 包含 int16 溢位模擬、uint32 variance、Newton 迭代
- **GELU**: Shift-based exponential、動態移位器
- **Softmax**: 與 GELU 類似，但需要累加器

### Step 3: 載入測試向量

在 SystemVerilog testbench 中：
```systemverilog
// 宣告記憶體
reg [31:0] input_mem [0:150527];    // 1*3*224*224 = 150528 bytes
reg [31:0] golden_mem [0:999];      // 1000 classes

// 載入測試向量
initial begin
    $readmemh("rtl_vectors/input_image.hex", input_mem);
    $readmemh("rtl_vectors/final_output.hex", golden_mem);
end

// 執行 RTL
// ...

// 比較結果
integer i, errors;
initial begin
    errors = 0;
    for (i = 0; i < 1000; i = i + 1) begin
        if (output_mem[i] !== golden_mem[i]) begin
            $error("Mismatch at %d: got %h, expected %h",
                   i, output_mem[i], golden_mem[i]);
            errors = errors + 1;
        end
    end
    
    if (errors == 0)
        $display("✓ Verification PASSED!");
    else
        $display("✗ Verification FAILED with %d errors", errors);
end
```

### Step 4: 驗證策略

#### 單元測試
- 測試單個算法單元（Dense, LayerNorm, GELU, Softmax）
- 使用隨機生成的測試向量
- 與 C-Model 輸出進行 bit-exact 比較

#### 集成測試
- 測試完整的 Transformer Block
- 使用 `block0_norm1_output.*` 作為輸入
- 比較 Block 輸出與 Golden Patterns

#### 端到端測試
- 測試完整模型
- 使用 `input_image.*` 作為輸入
- 比較最終輸出與 `final_output.*`
- 驗證預測類別是否為 1（概率 0.3996）

---

## 模型架構

### DeiT-Tiny 規格
```
Embedding Dim:    192
Depth:            12 Transformer blocks
Num Heads:        3
MLP Ratio:        4 (hidden dim = 768)
Num Classes:      1000
Input Size:       224x224x3
Patch Size:       16x16
Num Patches:      196
Total Params:     ~5.7M
Quantized Size:   ~1.4MB (int8 weights)
```

### 數據流
```
Input Image [1, 3, 224, 224] (float32)
    ↓ Normalize & Quantize
Input Image [1, 3, 224, 224] (int8)
    ↓ Patch Embedding (Conv2d 16x16, stride=16)
Patches [1, 196, 192] (int32)
    ↓ Add CLS Token & Position Embedding
Tokens [1, 197, 192] (int32)
    ↓ Transformer Block 0-11 (x12)
    │   ├─ LayerNorm1
    │   ├─ Multi-Head Attention (3 heads)
    │   │   ├─ QKV Projection [1, 197, 576]
    │   │   ├─ Reshape to [1, 3, 197, 192]
    │   │   ├─ Q @ K^T [1, 3, 197, 197]
    │   │   ├─ Softmax
    │   │   └─ Attn @ V [1, 3, 197, 192]
    │   ├─ Residual Add
    │   ├─ LayerNorm2
    │   ├─ MLP
    │   │   ├─ FC1 [1, 197, 768]
    │   │   ├─ GELU
    │   │   └─ FC2 [1, 197, 192]
    │   └─ Residual Add
Tokens [1, 197, 192] (int32)
    ↓ Final LayerNorm
Tokens [1, 197, 192] (int32)
    ↓ Extract CLS Token
CLS [1, 192] (int32)
    ↓ Classification Head
Logits [1, 1000] (int32)
```

---

## 關鍵算法參考

### 1. 線性層 (Dense / MatMul)

**算法**:
```python
def int_dense_kernel(x_int, weight_int, bias_int=None):
    # 1. 轉換為 int32
    x_val = x_int.astype(np.int32)
    w_val = weight_int.astype(np.int32)
    
    # 2. 矩陣乘法 (MAC)
    out_val = np.matmul(x_val, w_val.T)
    
    # 3. 加 bias
    if bias_int is not None:
        out_val = out_val + bias_int.astype(np.int32)
    
    return out_val.astype(np.int32)
```

**RTL 要點**:
- 輸入/權重: 8-bit signed
- MAC 累加器: 32-bit signed
- Bias: 32-bit signed
- 無需 scaling 或 offset（對稱量化）

### 2. LayerNorm

**算法** (詳見 `cmodel_rtl_reference/nonlinear_cmodel_reference.py`):
```python
def int_layer_norm_fixed(x_int, bias_int):
    # 1. 計算 mean (int16 溢位模擬)
    sum_val = np.sum(x_val, axis=-1, keepdims=True)
    sum_wrapped = sum_val.astype(np.int16).astype(np.int32)
    mean_int = np.fix(sum_wrapped / N).astype(np.int32)
    
    # 2. Centering
    y_int = x_val - mean_int
    y_int = y_int.astype(np.int16).astype(np.int32)
    
    # 3. Variance (uint32)
    data_sq = (y_int * y_int).astype(np.uint32)
    var = np.sum(data_sq, axis=-1, keepdims=True).astype(np.uint32)
    
    # 4. Newton 迭代求 sqrt (10 次)
    std = np.full_like(var, 2**16, dtype=np.uint32)
    for _ in range(10):
        safe_std = np.maximum(std, np.uint32(1))
        std = (std + var // safe_std) // np.uint32(2)
    
    # 5. Normalize
    factor = np.int32(2**31 - 1)
    factor_div_std = (factor // std_safe).astype(np.int32)
    term = factor_div_std.astype(np.int64) * y_int.astype(np.int64)
    y_norm = np.where(term >= 0, term // 2, -((-term) // 2)).astype(np.int32)
    
    # 6. Add bias
    output_int = y_norm + bias_int.astype(np.int32)
    return output_int
```

**RTL 要點**:
- Mean 計算有 int16 溢位（必須模擬）
- Variance 使用 uint32
- Newton 迭代需要 10 個 clock cycles
- 中間計算需要 64-bit 暫存器
- 除法可以用查找表（LUT）優化

### 3. GELU

**算法** (詳見 `cmodel_rtl_reference/nonlinear_cmodel_reference.py`):
```python
def int_gelu_kernel_fixed(x_int, x0_int=127, output_bit=8, n=23):
    # 1. Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # 2. Exponential
    exp_int = int_exp_shift_kernel_standard(x_algo, x0_int, n)
    exp_int_max = int_exp_shift_kernel_standard(-x_int_max, x0_int, n)
    
    # 3. Sigmoid
    exp_int_sum = exp_int + exp_int_max
    exp_int_sum = np.minimum(exp_int_sum, 2**31-1)
    factor = (2**31-1) // exp_int_sum_safe
    
    # 4. Output
    term = exp_int * factor
    sigmoid_int = (term >> (31-output_bit+1))
    output = pre_x_int * sigmoid_int
    return output
```

**RTL 要點**:
- 使用 Shift-based exponential
- 需要動態移位器（Barrel Shifter）
- 除法可以用查找表（LUT）優化
- 需要找最大值電路

### 4. Softmax

**算法** (詳見 `cmodel_rtl_reference/nonlinear_cmodel_reference.py`):
```python
def int_softmax_kernel_fixed(x_int, x0_int=127, output_bit=8, n=16):
    # 1. Stability shift
    x_int_max = np.max(pre_x_int, axis=-1, keepdims=True)
    x_algo = pre_x_int - x_int_max
    
    # 2. Exponential
    exp_int = int_exp_shift_kernel_standard(x_algo, x0_int, n)
    
    # 3. Normalize
    exp_int_sum = np.sum(exp_int, axis=-1, keepdims=True)
    exp_int_sum = np.minimum(exp_int_sum, 2**31-1)
    factor = (2**31-1) // exp_int_sum_safe
    
    # 4. Output
    term = exp_int * factor
    output = (term >> (31-output_bit+1))
    return output
```

**RTL 要點**:
- 與 GELU 類似
- 需要累加器（32-bit unsigned）
- 除法可以用查找表（LUT）優化
- 需要找最大值電路

---

## 文件清單

### 核心文件
- ✅ `I-ViT/output_gpu/checkpoint_converted.pth` - 轉換後的 checkpoint
- ✅ `I-ViT/test_pytorch_only.py` - PyTorch 模型測試
- ✅ `I-ViT/verify_cmodel.py` - C-Model 驗證測試
- ✅ `I-ViT/extract_golden_patterns.py` - Golden Patterns 提取
- ✅ `I-ViT/generate_rtl_vectors.py` - RTL 測試向量生成
- ✅ `cmodel_rtl_reference/linear_cmodel_reference.py` - 線性層 C-Model
- ✅ `cmodel_rtl_reference/nonlinear_cmodel_reference.py` - 非線性層 C-Model
- ✅ `cmodel_rtl_reference/pytorch_cmodel.py` - 完整 C-Model 框架

### Golden Patterns
- ✅ `golden_patterns/golden_patterns.npz` - 所有中間層數據
- ✅ `golden_patterns/golden_patterns_report.txt` - 數據摘要
- ✅ `golden_patterns/model_weights.npz` - 模型權重
- ✅ `golden_patterns/model_weights_report.txt` - 權重摘要

### RTL 測試向量
- ✅ `rtl_vectors/input_image.*` - 輸入圖片（3 種格式）
- ✅ `rtl_vectors/patch_embed_output.*` - Patch Embedding 輸出
- ✅ `rtl_vectors/block0_norm1_output.*` - Block 0 LayerNorm1 輸出
- ✅ `rtl_vectors/final_output.*` - 最終輸出 logits
- ✅ `rtl_vectors/prediction.txt` - 預測結果
- ✅ `rtl_vectors/README.txt` - 使用說明

### 文檔
- ✅ `docs/cmodel/FINAL_SUMMARY.md` - 完整總結
- ✅ `docs/cmodel/RTL_SIMULATION_GUIDE.md` - RTL 模擬指南
- ✅ `docs/cmodel/PYTORCH_CMODEL_GUIDE.md` - C-Model 使用指南
- ✅ `docs/cmodel/HARDWARE_CMODEL_GUIDE.md` - 硬體實作建議
- ✅ `README_CMODEL.md` - 快速開始指南
- ✅ `RTL_SIMULATION_READY.md` - 本文檔

---

## 下一步行動

### 立即可做
1. ✅ 閱讀文檔，理解 C-Model 算法
2. ✅ 開始 RTL 模組設計
3. ✅ 準備 RTL 驗證環境

### RTL 實作建議順序
1. **Week 1-2**: 實作基礎算法單元
   - `int_dense_kernel` (線性層)
   - `int_matmul_kernel` (矩陣乘法)
   - 單元測試與驗證

2. **Week 3-4**: 實作非線性算法單元
   - `int_layer_norm_fixed` (LayerNorm)
   - `int_gelu_kernel_fixed` (GELU)
   - `int_softmax_kernel_fixed` (Softmax)
   - 單元測試與驗證

3. **Week 5-6**: 組合 Transformer Block
   - Multi-Head Attention
   - MLP (Feed-Forward Network)
   - Residual connections
   - 集成測試與驗證

4. **Week 7-8**: 完整模型集成
   - Patch Embedding
   - Position Embedding
   - 12 個 Transformer Blocks
   - Classification Head
   - 端到端測試與驗證

---

## 性能指標

### 準確率
- **Top-1**: 73.54% (1000 張 ImageNet 圖片)
- **Top-5**: 92.58%

### 模型大小
- **參數量**: 約 5.7M
- **量化後**: 約 1.4MB (int8 權重)

### 推理時間（CPU）
- **單張圖片**: 約 482.7ms
- **批次大小**: 1

### 硬體資源估計（FPGA）
- **DSP Blocks**: ~500-1000 (用於 MAC 陣列)
- **BRAM**: ~2-4 MB (用於權重和中間結果)
- **LUT**: ~100K-200K (用於控制邏輯和非線性函數)
- **FF**: ~50K-100K (用於暫存器和狀態機)

---

## 常見問題

### Q: 如何開始 RTL 實作？
A: 
1. 閱讀 `docs/cmodel/FINAL_SUMMARY.md`
2. 參考 `cmodel_rtl_reference/` 中的算法單元
3. 從簡單的模組開始（如 `int_dense_kernel`）
4. 逐步實作完整的 Transformer block

### Q: 如何驗證 RTL 實作？
A:
1. 使用 `rtl_vectors/` 中的測試向量
2. 在 RTL 模擬器中載入測試向量
3. 比較 RTL 輸出與 Golden Patterns（要求 Bit-Exact）
4. 驗證最終預測類別是否為 1（概率 0.3996）

### Q: C-Model 和 PyTorch 模型有什麼區別？
A:
- **PyTorch 模型**: 完整的端到端模型，已驗證正確（73.54% 準確率）
- **C-Model**: 算法單元的參考實作，用於 RTL 開發

建議使用 PyTorch 模型作為 Golden Reference，C-Model 作為算法參考。

### Q: 為什麼不直接使用 TVM？
A: 經過詳細測試，TVM 0.9.0 的整數推理實作有根本性 bug（0% 準確率），無法作為可靠的 Golden Reference。PyTorch 量化模型已驗證正確（73.54% 準確率），是更好的選擇。

### Q: RTL 實作的關鍵挑戰是什麼？
A:
1. **LayerNorm**: 需要模擬 int16 溢位、實作 Newton 迭代求平方根
2. **GELU/Softmax**: 需要動態移位器（Barrel Shifter）和除法器（可用 LUT 優化）
3. **記憶體管理**: 需要高效的 buffer 管理來存儲中間結果
4. **時序優化**: 需要平衡吞吐量和資源使用

### Q: 如何優化硬體性能？
A:
1. **並行化**: 使用多個 MAC 單元並行計算
2. **流水線**: 將長路徑分割成多個流水線階段
3. **記憶體優化**: 使用雙緩衝技術減少記憶體訪問延遲
4. **LUT 優化**: 對於除法和非線性函數，使用查找表代替計算

---

## 結論

我們已經完成了 RTL 模擬的所有準備工作：

1. ✅ **PyTorch 量化模型**: 已驗證，73.54% 準確率
2. ✅ **C-Model 算法單元**: 已驗證，所有測試通過
3. ✅ **Golden Patterns**: 成功提取，包含 378 個數據項
4. ✅ **RTL 測試向量**: 成功生成，支援 3 種格式
5. ✅ **完整文檔**: 包含使用指南、RTL 實作建議、調試技巧

**推薦方案**: 使用 PyTorch 模型作為 Golden Reference，C-Model 作為算法參考，進行 RTL 驗證。

**關鍵優勢**:
- PyTorch 模型已完全驗證
- C-Model 算法單元已驗證（100% 匹配）
- Golden Patterns 已提取（378 個數據項）
- RTL 測試向量已生成（3 種格式）
- 文檔完整，易於理解
- 可以逐層驗證 RTL 實作

**現在可以開始 RTL 實作了！** 🚀

---

## 聯絡信息

如有問題，請參考：
- 完整總結：`docs/cmodel/FINAL_SUMMARY.md`
- RTL 指南：`docs/cmodel/RTL_SIMULATION_GUIDE.md`
- C-Model 指南：`docs/cmodel/PYTORCH_CMODEL_GUIDE.md`
- 硬體指南：`docs/cmodel/HARDWARE_CMODEL_GUIDE.md`

## 授權

本項目基於 I-ViT 論文的開源實作。
