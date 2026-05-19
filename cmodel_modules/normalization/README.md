# 歸一化運算模組

本目錄包含歸一化運算模組。

## 模組列表

### 1. int_layer_norm.py - 整數 LayerNorm

**算法**:
1. `mean = round(mean(x))`
2. `y = x - mean`
3. `var = sum(y^2)`
4. `std = sqrt(var)` using Newton iteration (20 次)
5. `factor = floor((2^31 - 1) / std)`
6. `y_norm = round(y * factor / 2)`
7. `output = y_norm + bias_int`

**輸入**:
- `x_int`: int16 [batch, seq_len, features]
- `bias_int`: float64 [features] (實際是整數值)
- `weight`: float32 [features] (LayerNorm weight)
- `bias`: float32 [features] (LayerNorm bias)
- `dim_sqrt`: sqrt(features)

**輸出**:
- `output_int`: int32 [batch, seq_len, features]

**RTL 對應**: `int_layer_norm.sv`

**使用範例**:
```python
from cmodel_modules.normalization import int_layer_norm

output = int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt)
```

## 重要說明

### Newton Iteration 使用 float64

LayerNorm 的 Newton iteration 使用 float64 計算 sqrt，但每步都使用 `floor()`，模擬整數除法行為。

**為什麼可以這樣做？**
- variance 值遠低於 float64 精確整數表示上限（2^53）
- 不會有精度損失
- 行為與硬體一致

**RTL 實現**:
- 使用整數除法器
- 或使用查表法 (LUT) + 插值

### 迭代次數

C-Model 使用 **20 次** Newton iteration，RTL template 目前使用 10 次。

**建議**: RTL 也改為 20 次以保持一致性。

## 測試

運行單元測試:
```bash
python cmodel_modules/normalization/int_layer_norm.py
```

## RTL 實現注意事項

1. **Newton Iteration**: 需要整數除法器或查表法
2. **迭代次數**: 建議使用 20 次（與 C-Model 一致）
3. **Multiple Driver 問題**: 確保 `cnt` signal 只有一個 driver
4. **除法運算**: 避免使用 RTL 除法（`/`），改用移位或查表
