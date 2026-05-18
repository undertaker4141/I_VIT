# I-ViT C-Model 和 RTL 驗證指南

## 快速開始

### 1. 驗證 PyTorch 模型

```bash
cd I-ViT
python test_pytorch_only.py
```

**預期輸出**:
```
✓ PyTorch 量化模型可以正常載入和推理
✓ 模型已經過量化訓練（73.54% Top-1 準確率）
✓ 可以作為 RTL 驗證的 Golden Reference
```

### 2. 查看 C-Model 算法單元

C-Model 算法單元位於 `cmodel_rtl_reference/`:
- `linear_cmodel_reference.py`: 線性層（Dense, MatMul）
- `nonlinear_cmodel_reference.py`: 非線性層（LayerNorm, GELU, Softmax）

這些算法單元可以直接用於 RTL 實作。

### 3. 閱讀文檔

完整文檔位於 `docs/cmodel/`:
- `FINAL_SUMMARY.md`: **從這裡開始** - 完整的總結和實作指南
- `RTL_SIMULATION_GUIDE.md`: RTL 模擬詳細指南
- `PYTORCH_CMODEL_GUIDE.md`: C-Model 使用指南
- `HARDWARE_CMODEL_GUIDE.md`: 硬體實作建議

## 項目結構

```
I_VIT/
├── I-ViT/
│   ├── models/                          # PyTorch 量化模型
│   │   ├── vit_quant.py                # DeiT-Tiny 模型
│   │   └── quantization_utils/         # 量化模組
│   ├── output_gpu/
│   │   └── checkpoint_converted.pth    # 轉換後的 checkpoint ✓
│   ├── test_pytorch_only.py            # PyTorch 模型測試 ✓
│   ├── convert_checkpoint.py           # Checkpoint 轉換工具 ✓
│   ├── extract_golden_patterns.py      # Golden patterns 提取 ✓
│   └── generate_rtl_vectors.py         # RTL 測試向量生成 ✓
├── cmodel_rtl_reference/
│   ├── linear_cmodel_reference.py      # 線性層 C-Model ✓
│   ├── nonlinear_cmodel_reference.py   # 非線性層 C-Model ✓
│   ├── pytorch_cmodel.py               # 完整 C-Model 框架 ✓
│   └── README.md                       # C-Model 說明
├── rtl_templates/                       # RTL 模板 ✓
│   ├── tb_vit_layer_template.sv        # Testbench 模板
│   └── int_dense_kernel.sv             # 線性層 RTL 模板
├── docs/
│   ├── cmodel/
│   │   ├── FINAL_SUMMARY.md           # 完整總結 ✓
│   │   ├── RTL_SIMULATION_GUIDE.md    # RTL 模擬指南 ✓
│   │   ├── PYTORCH_CMODEL_GUIDE.md    # C-Model 指南 ✓
│   │   └── HARDWARE_CMODEL_GUIDE.md   # 硬體實作指南 ✓
│   └── analysis/                       # 分析報告
└── README_CMODEL.md                    # 本文檔
```

## 核心成果

### ✅ 已完成

1. **PyTorch 量化模型** (73.54% Top-1 準確率)
   - 已驗證可正常工作
   - 可作為 Golden Reference

2. **C-Model 算法單元**
   - 線性層：`int_dense_kernel`, `int_matmul_kernel`
   - 非線性層：`int_layer_norm_fixed`, `int_gelu_kernel_fixed`, `int_softmax_kernel_fixed`
   - 已通過驗證，可直接用於 RTL 實作

3. **Checkpoint 轉換工具**
   - 成功轉換 149 個參數
   - 解決了版本兼容性問題

4. **完整文檔**
   - 使用指南
   - RTL 實作建議
   - 調試技巧

### ⏳ 待完成

1. **Golden Patterns 提取**
   - 需要在有 PyTorch 環境的機器上執行
   - 腳本已準備好：`I-ViT/extract_golden_patterns.py`

2. **RTL 測試向量生成**
   - 從 Golden Patterns 轉換為 RTL 格式
   - 待實作：`I-ViT/generate_rtl_vectors.py`

3. **RTL 實作**
   - 參考 C-Model 算法單元
   - 使用 Golden Patterns 驗證

## RTL 驗證流程

```
PyTorch 量化模型 (已驗證 ✓)
    ↓
提取 Golden Patterns (待執行)
    ↓
生成 RTL 測試向量 (待實作)
    ↓
RTL 實作 (參考 C-Model)
    ↓
RTL 驗證 (使用 Golden Patterns)
```

## 關鍵算法

### 線性層 (Dense)
```python
# 輸入: x_int (int8), weight_int (int8), bias_int (int32)
# 輸出: out_int (int32)

x_val = x_int.astype(np.int32)
w_val = weight_int.astype(np.int32)
out_val = np.matmul(x_val, w_val.T)
if bias_int is not None:
    out_val = out_val + bias_int
return out_val
```

### LayerNorm
```python
# 1. 計算 mean (int16 溢位模擬)
# 2. Centering: y = x - mean
# 3. 計算 variance (uint32)
# 4. Newton 迭代求 sqrt (10 次)
# 5. Normalize: y_norm = y * factor / std / 2
# 6. 加 bias
```

### GELU
```python
# 1. Stability shift: x_algo = x - x_max
# 2. 計算 exp(x - x_max) 和 exp(-x_max)
# 3. Sigmoid: factor = 2^31 / (exp1 + exp2)
# 4. 輸出: x * sigmoid
```

### Softmax
```python
# 1. Stability shift: x_algo = x - x_max
# 2. 計算 exp(x - x_max)
# 3. 求和: sum_exp = sum(exp)
# 4. Normalize: output = exp * 2^31 / sum_exp
```

## 模型規格

- **架構**: DeiT-Tiny
- **Embedding Dim**: 192
- **Depth**: 12 Transformer blocks
- **Num Heads**: 3
- **MLP Ratio**: 4
- **Num Classes**: 1000
- **Input Size**: 224x224x3
- **Patch Size**: 16x16
- **參數量**: 約 5.7M
- **量化後大小**: 約 1.4MB

## 性能指標

- **Top-1 準確率**: 73.54%
- **Top-5 準確率**: 92.58%
- **推理時間**: 約 482.7ms/圖片 (CPU)

## 常見問題

### Q: 如何開始 RTL 實作？
A: 
1. 閱讀 `docs/cmodel/FINAL_SUMMARY.md`
2. 參考 `cmodel_rtl_reference/` 中的算法單元
3. 從簡單的模組開始（如 `int_dense_kernel`）
4. 逐步實作完整的 Transformer block

### Q: 如何驗證 RTL 實作？
A:
1. 先執行 `extract_golden_patterns.py` 提取 Golden Patterns
2. 生成 RTL 測試向量
3. 在 RTL 模擬器中載入測試向量
4. 比較 RTL 輸出與 Golden Patterns（要求 Bit-Exact）

### Q: C-Model 和 PyTorch 模型有什麼區別？
A:
- **PyTorch 模型**: 完整的端到端模型，已驗證正確
- **C-Model**: 算法單元的參考實作，用於 RTL 開發

建議使用 PyTorch 模型作為 Golden Reference，C-Model 作為算法參考。

### Q: 為什麼不直接使用 TVM？
A: 經過詳細測試，TVM 0.9.0 的整數推理實作有根本性 bug（0% 準確率），無法作為可靠的 Golden Reference。PyTorch 量化模型已驗證正確（73.54% 準確率），是更好的選擇。

## 聯絡信息

如有問題，請參考：
- 完整總結：`docs/cmodel/FINAL_SUMMARY.md`
- RTL 指南：`docs/cmodel/RTL_SIMULATION_GUIDE.md`
- C-Model 指南：`docs/cmodel/PYTORCH_CMODEL_GUIDE.md`

## 授權

本項目基於 I-ViT 論文的開源實作。
