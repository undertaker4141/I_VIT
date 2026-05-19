# 基礎運算模組

本目錄包含基礎的整數運算模組。

## 模組列表

### 1. int_dense.py - 整數 Dense/Linear 層

**算法**: `output = (x @ W^T) + bias`

**輸入**:
- `x_int`: int8 [batch, ..., in_features]
- `weight_int`: int8 [out_features, in_features]
- `bias_int`: int32 [out_features]

**輸出**:
- `output_int`: int32 [batch, ..., out_features]

**RTL 對應**: `int_dense_kernel.sv`

**使用範例**:
```python
from cmodel_modules.basic_ops import int_dense

output = int_dense(x_int, weight_int, bias_int)
```

### 2. int_matmul.py - 整數矩陣乘法

**算法**: `output = A @ B`

**輸入**:
- `A_int`: int8 [M, K]
- `B_int`: int8 [K, N]

**輸出**:
- `output_int`: int32 [M, N]

**RTL 對應**: 可複用 `int_dense_kernel.sv` 的 MAC 單元

**使用範例**:
```python
from cmodel_modules.basic_ops import int_matmul

output = int_matmul(A_int, B_int)
```

## 測試

運行單元測試:
```bash
python cmodel_modules/basic_ops/int_dense.py
python cmodel_modules/basic_ops/int_matmul.py
```

## RTL 實現注意事項

1. **MAC 運算**: int8 × int8 -> int32
2. **並行度**: 需要決定 PE (Processing Element) 數量
3. **記憶體頻寬**: 權重和輸入的讀取頻寬
4. **流水線**: 可以使用多級流水線提高吞吐量
