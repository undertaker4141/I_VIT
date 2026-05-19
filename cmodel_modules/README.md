# C-Model 運算模組

本目錄包含 C-Model 的各個運算模組，按功能分類，方便 RTL 對照實現。

## 目錄結構

```
cmodel_modules/
├── README.md                    # 本文件
├── basic_ops/                   # 基礎運算
│   ├── int_dense.py            # 整數 Dense/Linear 層
│   ├── int_matmul.py           # 整數矩陣乘法
│   └── README.md
├── normalization/               # 歸一化運算
│   ├── int_layer_norm.py       # 整數 LayerNorm
│   └── README.md
├── activation/                  # 激活函數
│   ├── int_gelu.py             # 整數 GELU
│   ├── int_softmax.py          # 整數 Softmax
│   ├── int_exp_shift.py        # 整數指數運算
│   └── README.md
├── quantization/                # 量化運算
│   ├── requantize_integer.py   # 純整數 Requantize
│   ├── quant_act_residual_integer.py  # 純整數殘差連接
│   ├── precompute_requant_params.py   # 預計算 M 和 S
│   └── README.md
└── utils/                       # 工具函數
    ├── quantize_to_int.py      # 浮點轉整數
    ├── dequantize_to_float.py  # 整數轉浮點
    └── README.md
```

## 模組說明

### 1. 基礎運算 (basic_ops/)

**int_dense.py** - 整數 Dense/Linear 層
- 輸入: int8
- 權重: int8
- 輸出: int32
- 算法: `output = (x @ W^T) + bias`

**int_matmul.py** - 整數矩陣乘法
- 輸入: int8 × int8
- 輸出: int32
- 算法: `output = A @ B`

### 2. 歸一化運算 (normalization/)

**int_layer_norm.py** - 整數 LayerNorm
- 輸入: int16
- 輸出: int32
- 算法:
  1. Mean: `round(mean(x))`
  2. Variance: `sum((x - mean)^2)`
  3. Std: Newton iteration (20 次)
  4. Normalize: `round(y * factor / 2)`

### 3. 激活函數 (activation/)

**int_gelu.py** - 整數 GELU
- 輸入: int32
- 輸出: int32
- 算法: `GELU(x) = x * sigmoid(1.702 * x)`
- 註: 使用明確的 int64 乘法（P1 修復）

**int_softmax.py** - 整數 Softmax
- 輸入: int32
- 輸出: int32
- 算法: `Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
- 註: 使用明確的 int64 乘法（P1 修復）

**int_exp_shift.py** - 整數指數運算
- 輸入: int64
- 輸出: int64
- 算法: 多項式近似 + 位移

### 4. 量化運算 (quantization/)

**precompute_requant_params.py** - 預計算 M 和 S
- 輸入: input_sf, output_sf (float)
- 輸出: M (int32), S (int)
- 算法: `scale = input_sf / output_sf = M * 2^(-S)`

**requantize_integer.py** - 純整數 Requantize（P0 修復）
- 輸入: int8/int16/int32
- 輸出: int8/int16/int32
- 算法: `output = ((x * M) + (1 << (S-1))) >> S`
- 註: 使用 int64 避免溢出

**quant_act_residual_integer.py** - 純整數殘差連接（P0 修復）
- 輸入: 兩個 int16
- 輸出: int16
- 算法: 將兩個輸入都轉換到輸出 scale，然後相加

### 5. 工具函數 (utils/)

**quantize_to_int.py** - 浮點轉整數
**dequantize_to_float.py** - 整數轉浮點

## RTL 實現對照

每個模組都對應一個 RTL 實現：

| C-Model 模組 | RTL 模組 | 狀態 |
|-------------|---------|------|
| int_dense.py | int_dense_kernel.sv | ⚠️ 需修復 |
| int_layer_norm.py | int_layer_norm.sv | ⚠️ 需修復 |
| int_gelu.py | - | ❌ 待實現 |
| int_softmax.py | - | ❌ 待實現 |
| int_matmul.py | - | ❌ 待實現 |
| requantize_integer.py | - | ❌ 待實現 |

## 使用方法

每個模組都是獨立的 Python 文件，可以單獨導入使用：

```python
from cmodel_modules.basic_ops.int_dense import int_dense
from cmodel_modules.normalization.int_layer_norm import int_layer_norm
from cmodel_modules.activation.int_gelu import int_gelu
from cmodel_modules.quantization.requantize_integer import requantize_integer

# 使用模組
output = int_dense(x_int, weight_int, bias_int)
```

## 測試

每個模組都包含單元測試，可以獨立運行：

```bash
python cmodel_modules/basic_ops/int_dense.py
python cmodel_modules/normalization/int_layer_norm.py
```

## P0 和 P1 修復

- ✅ **P0 修復**: `quantization/` 目錄下的模組使用純整數運算
- ✅ **P1 修復**: `activation/` 目錄下的模組使用明確的 int64

## 參考

- 完整 C-Model: `../cmodel_rtl_reference/pure_numpy_cmodel.py`
- RTL Templates: `../rtl_templates/`
- Golden Patterns: `../golden_patterns/`
