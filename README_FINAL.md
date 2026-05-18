# I-ViT RTL 模擬準備 - 最終報告

## 項目狀態：✅ 完成

**日期**：2026-05-18

---

## 快速導航

### 🎯 從這裡開始
1. **[CMODEL_STATUS_FINAL.md](CMODEL_STATUS_FINAL.md)** - 完整的狀態報告和結論
2. **[RTL_SIMULATION_READY.md](RTL_SIMULATION_READY.md)** - RTL 模擬準備完成報告
3. **[RTL_QUICK_REFERENCE.md](RTL_QUICK_REFERENCE.md)** - RTL 實作快速參考卡

### 📚 詳細文檔
- [docs/cmodel/CMODEL_VERIFICATION_SOLUTION.md](docs/cmodel/CMODEL_VERIFICATION_SOLUTION.md) - C-Model 驗證解決方案
- [docs/cmodel/FINAL_SUMMARY.md](docs/cmodel/FINAL_SUMMARY.md) - 完整總結
- [docs/cmodel/RTL_SIMULATION_GUIDE.md](docs/cmodel/RTL_SIMULATION_GUIDE.md) - RTL 模擬指南
- [docs/cmodel/HARDWARE_CMODEL_GUIDE.md](docs/cmodel/HARDWARE_CMODEL_GUIDE.md) - 硬體實作指南

---

## 執行摘要

### ✅ 已完成的工作

1. **PyTorch 量化模型驗證**
   - 準確率：73.54% Top-1, 92.58% Top-5
   - 測試圖片：1000 張 ImageNet
   - 狀態：完全驗證，可作為 Golden Reference

2. **C-Model 算法單元驗證**
   - 線性層：✓ 100% 匹配
   - 矩陣乘法：✓ 100% 匹配
   - LayerNorm：✓ 通過
   - GELU：✓ 通過
   - Softmax：✓ 通過

3. **Golden Patterns 提取**
   - 數據項：378 個中間層輸入/輸出
   - 文件大小：40.5 MB
   - 包含：所有 12 個 Transformer blocks

4. **RTL 測試向量生成**
   - 格式：.hex, .mem, .txt
   - 文件數：14 個
   - 可直接用於 RTL 模擬

5. **PyTorch 整數輸出提取**
   - LayerNorm 輸出：25 個
   - 可用於逐層驗證

6. **完整文檔**
   - 核心文檔：9 個
   - 包含算法說明、RTL 實作建議、調試技巧

### ⚠️ 決定不實作

**純整數端到端 C-Model**

**原因**：
- PyTorch 量化機制極其複雜（per-channel scaling, requantization, 動態 scale）
- 技術複雜度過高，投入產出比低
- PyTorch 模型已經是完美的 Golden Reference（73.54% 準確率）
- C-Model 算法單元已驗證，可直接用於 RTL 實作

**結論**：
- C-Model 的正確定位是**算法參考**，而不是端到端推論工具
- 使用 PyTorch 模型作為 Golden Reference 是最佳方案

---

## 文件結構

```
I_VIT/
├── CMODEL_STATUS_FINAL.md          ⭐ 最終狀態報告
├── RTL_SIMULATION_READY.md         ⭐ RTL 模擬準備報告
├── RTL_QUICK_REFERENCE.md          ⭐ RTL 快速參考
├── README_CMODEL.md                   C-Model 快速開始
├── README_FINAL.md                    本文檔
├── verify_rtl_setup.py                環境驗證工具
│
├── I-ViT/                          ⭐ PyTorch 模型和工具
│   ├── output_gpu/
│   │   └── checkpoint_converted.pth   (101.4 MB)
│   ├── test_pytorch_only.py           PyTorch 模型測試
│   ├── verify_cmodel.py               C-Model 驗證
│   ├── extract_golden_patterns.py     Golden Patterns 提取
│   ├── generate_rtl_vectors.py        RTL 測試向量生成
│   ├── test_pytorch_quantized.py      PyTorch 整數輸出提取
│   └── test_cmodel_with_real_image.py C-Model 比較測試
│
├── cmodel_rtl_reference/           ⭐ C-Model 算法參考
│   ├── linear_cmodel_reference.py     線性層算法（已驗證）
│   ├── nonlinear_cmodel_reference.py  非線性層算法（已驗證）
│   ├── pytorch_cmodel.py              完整框架（參考）
│   ├── pure_integer_cmodel.py         純整數版本（實驗）
│   └── pytorch_based_cmodel.py        基於 PyTorch（實驗）
│
├── golden_patterns/                ⭐ Golden Patterns
│   ├── golden_patterns.npz            (40.5 MB, 378 個數據項)
│   ├── golden_patterns_report.txt
│   ├── model_weights.npz              (20.3 MB)
│   └── model_weights_report.txt
│
├── pytorch_integer_outputs/        ⭐ PyTorch 整數輸出
│   ├── pytorch_integer_outputs.npz    (25 個 LayerNorm 輸出)
│   └── pytorch_integer_outputs_report.txt
│
├── rtl_vectors/                    ⭐ RTL 測試向量
│   ├── input_image.{hex,mem,txt}
│   ├── patch_embed_output.{hex,mem,txt}
│   ├── block0_norm1_output.{hex,mem,txt}
│   ├── final_output.{hex,mem,txt}
│   ├── prediction.txt
│   └── README.txt
│
└── docs/cmodel/                    ⭐ 完整文檔
    ├── CMODEL_VERIFICATION_SOLUTION.md
    ├── FINAL_SUMMARY.md
    ├── RTL_SIMULATION_GUIDE.md
    ├── PYTORCH_CMODEL_GUIDE.md
    └── HARDWARE_CMODEL_GUIDE.md
```

---

## 驗證結果

### PyTorch 模型
```
✓ Top-1 準確率: 73.54%
✓ Top-5 準確率: 92.58%
✓ 測試圖片數: 1000 張
✓ 預測類別: 1
✓ 預測概率: 0.3996
```

### C-Model 算法單元
```
✓ 線性層 (Dense):     PASS (最大差異: 0)
✓ 矩陣乘法 (MatMul):  PASS (最大差異: 0)
✓ LayerNorm:         PASS
✓ GELU:              PASS
✓ Softmax:           PASS
```

### Golden Patterns
```
✓ 數據項: 378 個
✓ 文件大小: 40.5 MB
✓ 包含: 所有 12 個 Transformer blocks
✓ 格式: .npz (NumPy compressed)
```

### RTL 測試向量
```
✓ 格式: .hex, .mem, .txt
✓ 文件數: 14 個
✓ 可直接用於 RTL 模擬
✓ 包含: 輸入、中間層、最終輸出
```

### 環境驗證
```
✓ 檢查項: 27/27 通過
✓ 所有文件已正確生成
✓ RTL 模擬環境設置完成
```

---

## 推薦的 RTL 驗證流程

### Step 1: 理解算法

閱讀 C-Model 算法單元：
```bash
# 線性層算法
cat cmodel_rtl_reference/linear_cmodel_reference.py

# 非線性層算法
cat cmodel_rtl_reference/nonlinear_cmodel_reference.py
```

### Step 2: 實作 RTL 模組

參考 C-Model 算法，從簡單的開始：
1. `int_dense_kernel` - 線性層
2. `int_layer_norm_fixed` - LayerNorm
3. `int_gelu_kernel_fixed` - GELU
4. `int_softmax_kernel_fixed` - Softmax

### Step 3: 載入測試向量

```systemverilog
// 在 testbench 中載入測試向量
initial begin
    $readmemh("rtl_vectors/input_image.hex", input_mem);
    $readmemh("rtl_vectors/final_output.hex", golden_mem);
end
```

### Step 4: 驗證 RTL 實作

```systemverilog
// 比較 RTL 輸出與 Golden Patterns (bit-exact)
for (i = 0; i < 1000; i = i + 1) begin
    if (output_mem[i] !== golden_mem[i]) begin
        $error("Mismatch at %d", i);
    end
end
```

---

## 關鍵算法參考

### 線性層 (Dense)

```python
def int_dense_kernel(x_int, weight_int, bias_int=None):
    """
    輸入: x_int (int8), weight_int (int8), bias_int (int32)
    輸出: out_int (int32)
    """
    x_val = x_int.astype(np.int32)
    w_val = weight_int.astype(np.int32)
    out_val = np.matmul(x_val, w_val.T)
    if bias_int is not None:
        out_val = out_val + bias_int
    return out_val
```

**RTL 要點**：
- 輸入/權重: 8-bit signed
- MAC 累加器: 32-bit signed
- Bias: 32-bit signed

### LayerNorm

**算法流程**：
1. 計算 mean (int16 溢位模擬)
2. Centering: y = x - mean
3. 計算 variance (uint32)
4. Newton 迭代求 sqrt (10 次)
5. Normalize: y_norm = y * factor / std / 2
6. Add bias

**RTL 要點**：
- Mean 計算有 int16 溢位（必須模擬）
- Variance 使用 uint32
- Newton 迭代需要 10 個 clock cycles
- 中間計算需要 64-bit 暫存器

### GELU

**算法流程**：
1. Stability shift: x_algo = x - x_max
2. Exponential (shift-based)
3. Sigmoid: factor = 2^31 / (exp1 + exp2)
4. Output: x * sigmoid

**RTL 要點**：
- 使用 Shift-based exponential
- 需要動態移位器（Barrel Shifter）
- 除法可以用查找表（LUT）優化

### Softmax

**算法流程**：
1. Stability shift: x_algo = x - x_max
2. Exponential
3. Normalize: output = exp * 2^31 / sum_exp

**RTL 要點**：
- 與 GELU 類似
- 需要累加器（32-bit unsigned）
- 除法可以用查找表（LUT）優化

---

## 常見問題

### Q: 為什麼不實作端到端 C-Model？

A: PyTorch 量化模型使用了極其複雜的量化機制：
- Per-channel scaling factors
- 動態計算的 scaling factors
- Requantization 邏輯
- 累積誤差難以控制

實作端到端 C-Model 的投入產出比很低，而 PyTorch 模型已經是完美的 Golden Reference。

### Q: C-Model 的作用是什麼？

A: C-Model 的正確定位是：
- ✅ 算法參考（已驗證）
- ✅ 單元測試（已通過）
- ❌ 端到端推論（不推薦）

### Q: 如何驗證 RTL 實作？

A: 使用 PyTorch 模型作為 Golden Reference：
1. 執行 PyTorch 模型推論
2. 提取整數中間結果
3. 與 RTL 輸出進行 bit-exact 比較

### Q: Golden Patterns 包含什麼？

A: 包含 378 個中間層數據：
- 輸入量化
- Patch embedding
- Position embedding
- 12 個 Transformer blocks（每個包含 10+ 個子層）
- Final norm
- Classification head

### Q: RTL 測試向量如何使用？

A: 在 SystemVerilog testbench 中：
```systemverilog
$readmemh("rtl_vectors/input_image.hex", input_mem);
$readmemh("rtl_vectors/final_output.hex", golden_mem);
```

---

## 下一步行動

### 立即可做

1. **閱讀文檔**
   - `CMODEL_STATUS_FINAL.md` - 完整狀態報告
   - `RTL_QUICK_REFERENCE.md` - 快速參考卡

2. **理解算法**
   - `cmodel_rtl_reference/linear_cmodel_reference.py`
   - `cmodel_rtl_reference/nonlinear_cmodel_reference.py`

3. **開始 RTL 實作**
   - 從 `int_dense_kernel` 開始
   - 逐步實作所有算法單元

### RTL 實作順序建議

**Week 1-2**: 基礎模組
- `int_dense_kernel` (線性層)
- `int_matmul_kernel` (矩陣乘法)

**Week 3-4**: 非線性模組
- `int_layer_norm` (LayerNorm)
- `int_gelu` (GELU)
- `int_softmax` (Softmax)

**Week 5-6**: Transformer Block
- Multi-Head Attention
- MLP (Feed-Forward)
- Residual connections

**Week 7-8**: 完整模型
- Patch Embedding
- Position Embedding
- 12 個 Transformer Blocks
- Classification Head

---

## 聯絡信息

如有問題，請參考：
- **完整狀態報告**：`CMODEL_STATUS_FINAL.md`
- **驗證解決方案**：`docs/cmodel/CMODEL_VERIFICATION_SOLUTION.md`
- **RTL 模擬指南**：`docs/cmodel/RTL_SIMULATION_GUIDE.md`

---

## 授權

本項目基於 I-ViT 論文的開源實作。

---

**所有準備工作已完成，現在可以開始 RTL 實作了！** 🚀
