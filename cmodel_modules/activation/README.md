# 激活函數模組

本目錄包含激活函數模組。

## 模組列表

### 1. int_exp_shift.py - 整數指數運算

**算法**:
1. Polynomial approximation: `x + x/2 - x/16`
2. Clamping: `max(x, n * x0_int)`
3. Division: `q = x // x0_int, r = x - q * x0_int`
4. Base: `exp_base = r/2 - x0_int`
5. Shift: `exp_base << (n - q)` or `exp_base >> (q - n)`

**輸入**:
- `x_int`: int64
- `x0_int`: floor(-1.0 / scaling_factor) - 必須是負數 (int64)
- `n`: 精度參數 (23 for GELU, 15 for Softmax) (int64)

**輸出**:
- `exp_int`: int64

**RTL 對應**: 待實現

**使用範例**:
```python
from cmodel_modules.activation import int_exp_shift

exp_int = int_exp_shift(x_int, x0_int, n=23)
```

### 2. int_gelu.py - 整數 GELU 激活函數

**算法**: `GELU(x) = x * sigmoid(1.702 * x)`

**輸入**:
- `x_int`: int32
- `scaling_factor`: float or array
- `output_bit`: 輸出位元數 (default: 8)
- `n`: 指數運算的精度參數 (default: 23)

**輸出**:
- `output_int`: int32

**RTL 對應**: 待實現

**使用範例**:
```python
from cmodel_modules.activation import int_gelu

output = int_gelu(x_int, scaling_factor)
```

**✅ P1 修復**: 使用明確的 int64 乘法（不使用 object），最大乘積 ~2^62，不會溢出 int64。

### 3. int_softmax.py - 整數 Softmax

**算法**: `Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

**輸入**:
- `x_int`: int32
- `scaling_factor`: float or array
- `output_bit`: 輸出位元數 (default: 8)
- `n`: 指數運算的精度參數 (default: 15)

**輸出**:
- `output_int`: int32

**RTL 對應**: 待實現

**使用範例**:
```python
from cmodel_modules.activation import int_softmax

output = int_softmax(x_int, scaling_factor)
```

**✅ P1 修復**: 使用明確的 int64 乘法（不使用 object），最大乘積 ~2^62，不會溢出 int64。

## P1 修復說明

### 問題

原始實現使用 Python `object` 類型進行任意精度乘法：
```python
term = exp_int.astype(object) * factor.astype(object)  # ❌ 不明確
```

### 修復

改為使用明確的 `int64` 乘法：
```python
exp_int64 = exp_int.astype(np.int64)
factor_int64 = factor.astype(np.int64)
term = exp_int64 * factor_int64  # ✅ 明確的 64-bit 乘法
```

### 為什麼不會溢出？

- `exp_int` 最大值 ≤ 2^31
- `factor` 最大值 ≤ 2^31
- 乘積最大值 = 2^31 × 2^31 = 2^62
- int64 範圍 = [-2^63, 2^63-1]
- 2^62 < 2^63，所以不會溢出

## 測試

運行單元測試:
```bash
python cmodel_modules/activation/int_exp_shift.py
python cmodel_modules/activation/int_gelu.py
python cmodel_modules/activation/int_softmax.py
```

## RTL 實現注意事項

1. **指數運算**: 使用位移和多項式近似，避免浮點運算
2. **64-bit 乘法**: 需要 64-bit 乘法器（或分解為多個 32-bit 乘法）
3. **動態位移**: 需要支援可變位移量的 barrel shifter
4. **精度參數**: GELU 使用 n=23，Softmax 使用 n=15
