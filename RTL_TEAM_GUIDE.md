# RTL Team 使用指南

**最後更新**: 2026/5/22  
**狀態**: ✅ 所有模組已驗證完成，可開始 RTL 設計

---

## 快速開始

### 1. 推薦使用的 C-Model

**最推薦**: `cmodel_modules/` 目錄下的模組化實現

```
cmodel_modules/
├── basic_ops/          # 基本運算
│   ├── int_dense.py    # 全連接層
│   └── int_matmul.py   # 矩陣乘法
├── normalization/      # 正規化
│   └── int_layer_norm.py  # LayerNorm
├── activation/         # 激活函數
│   ├── int_gelu.py     # GELU
│   ├── int_softmax.py  # Softmax
│   └── int_exp_shift.py  # Exponential (GELU/Softmax 內部使用)
├── quantization/       # 量化
│   ├── precompute_requant_params.py  # 預計算 M 和 S
│   ├── requantize_integer.py         # Requantize
│   └── quant_act_residual_integer.py # 殘差連接
└── utils/              # 工具
    ├── quantize_to_int.py      # 浮點數轉整數
    └── dequantize_to_float.py  # 整數轉浮點數
```

**優點**:
- ✅ 每個模組獨立，易於理解
- ✅ 包含單元測試
- ✅ 有詳細的 README 文檔
- ✅ 與 PyTorch 原始碼完全一致

### 2. 驗證資源

**Golden Patterns**: `golden_patterns/golden_patterns.npz`
- 包含 12 個 blocks 的完整中間結果
- 可用於 RTL 驗證
- 詳細資訊: `golden_patterns/golden_patterns_report.txt`

**驗證腳本**: `I-ViT/test_nonlinear_modules.py`
- 驗證 LayerNorm, GELU, Softmax
- 使用 Golden Patterns 和隨機數據

---

## 模組說明

### 1. LayerNorm (正規化層)

**檔案**: `cmodel_modules/normalization/int_layer_norm.py`

**關鍵參數**:
- Newton iteration: **10 次** (不是 20 次)
- Step 6: 使用 **floor** (不是 round)

**演算法流程**:
1. Mean 計算 (int16 overflow 模擬)
2. Centering (去均值)
3. Variance 計算 (uint32)
4. Newton-Raphson 迭代 (10 次)
5. Normalization (factor / std)
6. 加上 Bias

**驗證狀態**: ✅ 通過 (使用 Golden Patterns)

**文檔**: `cmodel_modules/normalization/README.md`

---

### 2. GELU (激活函數)

**檔案**: `cmodel_modules/activation/int_gelu.py`

**關鍵參數**:
- `x0_int = 92681`
- `output_bit = 8`
- `n = 23`

**演算法流程**:
1. Stability Shift (x - x_max)
2. Exponential Kernel (使用 `int_exp_shift`)
3. Sigmoid 運算 (除法)
4. Re-scaling (與原輸入相乘)

**驗證狀態**: ✅ 通過 (隨機數據測試)

**文檔**: `cmodel_modules/activation/README.md`

---

### 3. Softmax (激活函數)

**檔案**: `cmodel_modules/activation/int_softmax.py`

**關鍵參數**:
- `x0_int = 92681`
- `output_bit = 8`
- `n = 15` (注意: 不是 16)

**演算法流程**:
1. Stability Shift (x - x_max)
2. Exponential Kernel (使用 `int_exp_shift`)
3. Summation & Division
4. Scaling & Output

**驗證狀態**: ✅ 通過 (隨機數據測試)

**文檔**: `cmodel_modules/activation/README.md`

---

### 4. Exponential (內部模組)

**檔案**: `cmodel_modules/activation/int_exp_shift.py`

**說明**: GELU 和 Softmax 內部使用的指數運算模組

**演算法流程**:
1. Polynomial Approximation (位元平移)
2. Clamping (下界限制)
3. Integer Division (整數除法)
4. Base Calculation
5. Final Shift (變動平移)

**重要**: Shift clamp 最大 62 bit (避免溢位)

---

### 5. Dense (全連接層)

**檔案**: `cmodel_modules/basic_ops/int_dense.py`

**說明**: 全連接層 (Linear layer)

**演算法流程**:
1. 矩陣乘法 (Y = X @ W^T)
2. 加上 Bias
3. Requantize (使用 `requantize_integer`)

**文檔**: `cmodel_modules/basic_ops/README.md`

---

### 6. MatMul (矩陣乘法)

**檔案**: `cmodel_modules/basic_ops/int_matmul.py`

**說明**: 矩陣乘法 (用於 Attention 的 QK^T 和 Attn@V)

**演算法流程**:
1. 矩陣乘法 (C = A @ B)
2. Requantize (使用 `requantize_integer`)

**文檔**: `cmodel_modules/basic_ops/README.md`

---

### 7. Requantize (重新量化)

**檔案**: `cmodel_modules/quantization/requantize_integer.py`

**說明**: 純整數 Requantize (M×2^(-S) 格式)

**演算法流程**:
1. 乘上 M (int32 × int32 → int64)
2. 右移 S bit
3. 截斷回 int32

**預計算**: 使用 `precompute_requant_params.py` 預計算 M 和 S

**驗證狀態**: ✅ 與浮點版本差異 = 0 LSB

**文檔**: `cmodel_modules/quantization/README.md`

---

### 8. 殘差連接

**檔案**: `cmodel_modules/quantization/quant_act_residual_integer.py`

**說明**: 純整數殘差連接 (處理不同 scaling factor)

**演算法流程**:
1. 將兩個輸入 requantize 到相同的 scaling factor
2. 相加
3. Requantize 到輸出的 scaling factor

**驗證狀態**: ✅ 與浮點版本差異 <= 1 LSB

**文檔**: `cmodel_modules/quantization/README.md`

---

## 驗證報告

### 1. Nonlinear 模組驗證報告

**檔案**: `NONLINEAR_MODULES_VERIFICATION_REPORT.md`

**內容**:
- LayerNorm, GELU, Softmax 驗證結果
- Golden Patterns 使用說明
- RTL 驗證建議
- 關鍵參數總結

### 2. C-Model 一致性修復總結

**檔案**: `FINAL_CONSISTENCY_SUMMARY.md`

**內容**:
- 所有 C-Model 與 PyTorch 原始碼的一致性修復
- LayerNorm: 10 次迭代 + floor
- GELU/Softmax: 明確的 int64
- Softmax: n=15
- Shift clamp: 最大 62 bit

### 3. C-Model 模組拆分報告

**檔案**: `CMODEL_MODULES_COMPLETION_REPORT.md`

**內容**:
- 13 個模組的拆分過程
- 目錄結構說明
- 單元測試結果
- 使用範例

---

## 重要注意事項

### ⚠️ 不要使用的檔案

1. **`cmodel_rtl_reference/nonlinear_cmodel_reference.py`**
   - ❌ TVM 版本，準確率 0%
   - ❌ 已失效，不要使用

2. **`cmodel_rtl_reference/pure_integer_operations.py`**
   - ❌ 已標記為 DEPRECATED
   - ❌ 包含 object 類型（第 384、432 行）

### ✅ 推薦使用的檔案

1. **模組化實現** (最推薦):
   - `cmodel_modules/` 目錄下的所有模組

2. **PyTorch 一致版本**:
   - `cmodel_rtl_reference/pytorch_integer_cmodel.py`

3. **完整 C-Model**:
   - `cmodel_rtl_reference/pure_numpy_cmodel.py`

---

## 關鍵參數總結

### 量化參數
- **Weight bit-width**: 8-bit
- **Activation bit-width**: 8-bit
- **Accumulator bit-width**: 32-bit (int32)
- **Intermediate bit-width**: 64-bit (int64，用於乘法)

### LayerNorm
- **Newton iteration**: 10 次
- **Step 6**: floor (不是 round)
- **Initial std**: 2^16 = 65536

### GELU
- **x0_int**: 92681
- **output_bit**: 8
- **n**: 23

### Softmax
- **x0_int**: 92681
- **output_bit**: 8
- **n**: 15 (不是 16)

### Exponential
- **Shift clamp**: 最大 62 bit

---

## 使用範例

### 範例 1: 使用 LayerNorm 模組

```python
import numpy as np
from cmodel_modules.normalization.int_layer_norm import int_layer_norm_fixed

# 輸入數據 (int32)
x_int = np.random.randint(-1000, 1000, size=(1, 197, 192), dtype=np.int32)

# Bias (int32)
bias_int = np.random.randint(-100, 100, size=(192,), dtype=np.int32)

# 執行 LayerNorm
output = int_layer_norm_fixed(x_int, bias_int)

print(f"輸入 shape: {x_int.shape}")
print(f"輸出 shape: {output.shape}")
print(f"輸出範圍: [{output.min()}, {output.max()}]")
```

### 範例 2: 使用 GELU 模組

```python
import numpy as np
from cmodel_modules.activation.int_gelu import int_gelu_kernel_fixed

# 輸入數據 (int32)
x_int = np.random.randint(-1000, 1000, size=(1, 197, 192), dtype=np.int32)

# 參數
x0_int = 92681
output_bit = 8
n = 23

# 執行 GELU
output = int_gelu_kernel_fixed(x_int, x0_int, output_bit, n)

print(f"輸入 shape: {x_int.shape}")
print(f"輸出 shape: {output.shape}")
print(f"輸出範圍: [{output.min()}, {output.max()}]")
```

### 範例 3: 使用 Softmax 模組

```python
import numpy as np
from cmodel_modules.activation.int_softmax import int_softmax_kernel_fixed

# 輸入數據 (int32)
x_int = np.random.randint(-1000, 1000, size=(1, 3, 197, 197), dtype=np.int32)

# 參數
x0_int = 92681
output_bit = 8
n = 15  # 注意: 不是 16

# 執行 Softmax
output = int_softmax_kernel_fixed(x_int, x0_int, output_bit, n)

print(f"輸入 shape: {x_int.shape}")
print(f"輸出 shape: {output.shape}")
print(f"輸出範圍: [{output.min()}, {output.max()}]")
```

### 範例 4: 使用 Requantize 模組

```python
import numpy as np
from cmodel_modules.quantization.precompute_requant_params import precompute_requant_params
from cmodel_modules.quantization.requantize_integer import requantize_integer

# 預計算 M 和 S
sf_in = 0.1
sf_out = 0.05
M, S = precompute_requant_params(sf_in, sf_out)

print(f"M = {M}, S = {S}")

# 輸入數據 (int32)
x_int = np.random.randint(-1000, 1000, size=(1, 197, 192), dtype=np.int32)

# 執行 Requantize
output = requantize_integer(x_int, M, S)

print(f"輸入 shape: {x_int.shape}")
print(f"輸出 shape: {output.shape}")
print(f"輸出範圍: [{output.min()}, {output.max()}]")
```

---

## RTL 驗證流程建議

### Step 1: 單元測試
1. 從 `cmodel_modules/` 選擇要實作的模組
2. 閱讀對應的 README.md
3. 參考 Python 實現撰寫 RTL
4. 使用模組的單元測試驗證

### Step 2: Golden Patterns 驗證
1. 從 `golden_patterns/golden_patterns.npz` 載入測試向量
2. 將浮點數轉換為整數（使用對應的 scaling factor）
3. 輸入到 RTL 模組
4. 比較 RTL 輸出與 golden patterns

### Step 3: 端到端驗證
1. 整合所有模組
2. 使用完整的測試圖片進行推論
3. 比較最終輸出與 golden patterns 中的 logits

---

## 常見問題 (FAQ)

### Q1: 為什麼 LayerNorm 使用 10 次迭代而不是 20 次？
**A**: 為了與 PyTorch 原始碼一致。經過測試，10 次迭代已經足夠達到所需的精度。

### Q2: 為什麼 Softmax 的 n=15 而不是 16？
**A**: 為了與 PyTorch 原始碼一致。這是經過實驗調整的最佳參數。

### Q3: 為什麼要使用 int64 而不是 object？
**A**: object 類型是 Python 的任意精度整數，無法直接對應到硬體。使用明確的 int64 可以確保硬體實現的一致性。

### Q4: Shift clamp 為什麼是 62 bit？
**A**: 因為 int64 的最大值是 2^63-1，為了避免溢位，我們限制 shift 最大為 62 bit。

### Q5: 如何選擇使用哪個 C-Model？
**A**: 
- **RTL 設計**: 使用 `cmodel_modules/` (模組化，易於理解)
- **驗證參考**: 使用 `pytorch_integer_cmodel.py` (與 PyTorch 一致)
- **完整流程**: 使用 `pure_numpy_cmodel.py` (包含完整推論)

### Q6: Golden Patterns 包含哪些數據？
**A**: 
- 12 個 blocks 的中間結果 (norm1_output, attn_output, residual1_output, norm2_output, mlp_output, residual2_output)
- 最終輸出 (final_norm_output, logits, pred_class)
- 輸入 (input) 和 Ground Truth (ground_truth)

### Q7: 如何驗證我的 RTL 實現是否正確？
**A**: 
1. 先使用模組的單元測試驗證基本功能
2. 再使用 Golden Patterns 驗證 bit-exact 結果
3. 最後進行端到端驗證

---

## 聯絡資訊

如有任何問題，請參考以下文檔：
- `NONLINEAR_MODULES_VERIFICATION_REPORT.md` - 驗證報告
- `FINAL_CONSISTENCY_SUMMARY.md` - 一致性修復總結
- `CMODEL_MODULES_COMPLETION_REPORT.md` - 模組拆分報告
- `cmodel_modules/README.md` - 模組總覽
- 各子目錄的 README.md - 詳細說明

---

**祝 RTL 設計順利！** 🚀
