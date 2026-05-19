# 量化運算模組

本目錄包含量化運算模組（P0 修復完成）。

## 模組列表

### 1. precompute_requant_params.py - 預計算 Requantization 參數

**算法**: `scale = input_sf / output_sf = M * 2^(-S)`

**輸入**:
- `input_sf`: 輸入 scaling factor (float or array)
- `output_sf`: 輸出 scaling factor (float or array)

**輸出**:
- `M`: 整數乘數 (int32 or array of int32)
- `S`: 右移位數 (int or array of int)

**RTL 對應**: 預處理階段（離線計算）

**使用範例**:
```python
from cmodel_modules.quantization import precompute_requant_params

M, S = precompute_requant_params(input_sf, output_sf)
```

**特點**:
- 離線預計算，不在推論路徑上
- 將浮點 scale 轉換為整數 M 和位移 S
- 支援 per-channel 量化

### 2. requantize_integer.py - 純整數 Requantization

**算法**: `output = ((x * M) + (1 << (S-1))) >> S`

**輸入**:
- `x_int`: int8/int16/int32
- `M_or_input_sf`: 整數乘數 M (int32) 或輸入 scaling factor (float)
- `S_or_output_sf`: 右移位數 S (int) 或輸出 scaling factor (float)
- `output_bits`: 輸出位元數 (8/16/32)

**輸出**:
- `output_int`: 重新量化的整數輸出

**RTL 對應**: 待實現

**使用範例**:
```python
from cmodel_modules.quantization import requantize_integer

# 新接口（使用 M 和 S）
output = requantize_integer(x_int, M, S, output_bits=8)

# 舊接口（自動預計算 M 和 S）
output = requantize_integer(x_int, input_sf, output_sf, output_bits=8)
```

**✅ P0 修復**: 使用純整數 M×2^(-S) 格式，避免浮點運算。使用 int64 避免乘法溢出。

### 3. quant_act_residual_integer.py - 純整數 QuantAct with Residual

**算法**:
1. `x1_scaled = (x1 * M1) >> S1`
2. `x2_scaled = (x2 * M2) >> S2`
3. `y = x1_scaled + x2_scaled`

**輸入**:
- `x1_int`: 第一個輸入（主路徑）
- `x1_sf`: 第一個輸入的 scaling factor
- `x2_int`: 第二個輸入（殘差路徑）
- `x2_sf`: 第二個輸入的 scaling factor
- `output_sf`: 輸出 scaling factor
- `output_bits`: 輸出位元數

**輸出**:
- `output_int`: 重新量化的整數輸出

**RTL 對應**: 待實現

**使用範例**:
```python
from cmodel_modules.quantization import quant_act_residual_integer

output = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf)
```

**✅ P0 修復**: 使用純整數加法，避免浮點運算。將兩個輸入對齊到輸出 scale，然後相加。

## P0 修復說明

### 問題

原始實現使用浮點運算：
```python
# requantize (舊版本)
scale = input_sf / output_sf                        # ❌ 浮點除法
output = np.round(x_int.astype(np.float32) * scale) # ❌ 浮點乘法

# quant_act_residual (舊版本)
x1_float = dequantize_to_float(x1_int, x1_sf)       # ❌ 先反量化成浮點
x2_float = dequantize_to_float(x2_int, x2_sf)
y_float = x1_float + x2_float                        # ❌ 浮點加法
y_int = quantize_to_int(y_float, output_sf, ...)
```

### 修復

改為純整數實現：
```python
# requantize (新版本)
M, S = precompute_requant_params(input_sf, output_sf)  # ✅ 離線預計算
scaled = x_int64 * M_int64                              # ✅ 整數乘法
output = (scaled + rounding_bias) >> S                  # ✅ 整數右移

# quant_act_residual (新版本)
x1_scaled = requantize_integer(x1_int, M1, S1, ...)    # ✅ 純整數 requantize
x2_scaled = requantize_integer(x2_int, M2, S2, ...)
y = x1_scaled + x2_scaled                               # ✅ 整數加法
```

### 為什麼這樣做？

1. **RTL 可實現**: 硬體無法直接實現浮點乘除法
2. **Bit-exact**: 整數運算結果完全確定，無浮點誤差
3. **高效**: 整數乘法 + 位移比浮點運算快得多

## 測試

運行單元測試:
```bash
python cmodel_modules/quantization/precompute_requant_params.py
python cmodel_modules/quantization/requantize_integer.py
python cmodel_modules/quantization/quant_act_residual_integer.py
```

## RTL 實現注意事項

1. **預計算**: M 和 S 在推論前預計算，存儲在 ROM 或寄存器中
2. **64-bit 乘法**: 需要 64-bit 乘法器（或分解為多個 32-bit 乘法）
3. **Rounding**: 右移前加上 `1 << (S-1)` 實現四捨五入
4. **Per-channel**: 支援 per-channel 量化，每個 channel 有獨立的 M 和 S
