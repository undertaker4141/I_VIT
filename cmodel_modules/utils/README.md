# 工具函數模組

本目錄包含工具函數模組（主要用於測試和驗證）。

## 模組列表

### 1. quantize_to_int.py - 量化到整數

**算法**: `x_int = round(x_float / scaling_factor)`

**輸入**:
- `x_float`: 浮點輸入
- `scaling_factor`: scaling factor (scalar or per-channel)
- `bits`: 位元數 (8, 16, 32)

**輸出**:
- `x_int`: 整數輸出

**使用範例**:
```python
from cmodel_modules.utils import quantize_to_int

x_int = quantize_to_int(x_float, scaling_factor, bits=8)
```

**註**: 這個函數主要用於測試和驗證，不在推論路徑上。

### 2. dequantize_to_float.py - 反量化到浮點

**算法**: `x_float = x_int * scaling_factor`

**輸入**:
- `x_int`: 整數輸入
- `scaling_factor`: scaling factor (scalar or per-channel)

**輸出**:
- `x_float`: 浮點輸出

**使用範例**:
```python
from cmodel_modules.utils import dequantize_to_float

x_float = dequantize_to_float(x_int, scaling_factor)
```

**註**: 這個函數主要用於測試和驗證，不在推論路徑上。

## 使用場景

這些工具函數主要用於：

1. **測試**: 驗證整數運算的正確性
2. **調試**: 將整數結果轉換為浮點數以便檢查
3. **比較**: 與 PyTorch 浮點模型比較

## 重要說明

⚠️ **這些函數不在推論路徑上**

在實際的 C-Model 推論中，我們使用：
- `requantize_integer()` - 純整數 requantization（P0 修復）
- `quant_act_residual_integer()` - 純整數殘差連接（P0 修復）

這些工具函數只是為了方便測試和驗證。

## 測試

運行單元測試:
```bash
python cmodel_modules/utils/quantize_to_int.py
python cmodel_modules/utils/dequantize_to_float.py
```

## 量化誤差

量化會引入誤差：
- **量化誤差** = ±0.5 × scaling_factor
- **相對誤差** = 量化誤差 / 原始值

例如：
- scaling_factor = 0.01
- 量化誤差 = ±0.005
- 對於 x = 1.0，相對誤差 = 0.5%
- 對於 x = 0.1，相對誤差 = 5%

這就是為什麼我們需要 QAT (Quantization-Aware Training) 來訓練模型適應量化誤差。
