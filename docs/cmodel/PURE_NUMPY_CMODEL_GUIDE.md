# 純 NumPy C-Model 實現指南

## 概述

這是一個**完全底層的 C-Model 實現**，只使用 NumPy，不依賴 PyTorch。所有運算都是純整數（int8, int16, int32, int64），可以直接轉換為 RTL 實現。

## 架構

### 1. 基礎運算模組 (`cmodel_rtl_reference/pure_numpy_cmodel.py`)

#### 1.1 LayerNorm
```python
def int_layer_norm(x_int, bias_int, weight, bias, dim_sqrt):
    """
    純整數 LayerNorm
    
    輸入: int16 [batch, seq_len, features]
    輸出: int32 [batch, seq_len, features]
    
    算法:
        1. mean = round(mean(x))
        2. y = x - mean
        3. var = sum(y^2)
        4. std = sqrt(var) using Newton iteration
        5. factor = floor((2^31 - 1) / std)
        6. y_norm = floor(y * factor / 2)
        7. output = y_norm + bias_int
    """
```

**特點**:
- 使用 `round(mean())` 而不是 `floor(mean())`，避免 int16 溢位
- Newton iteration 計算平方根（10 次迭代）
- 輸出範圍: int32

#### 1.2 Dense/Linear
```python
def int_dense(x_int, weight_int, bias_int):
    """
    純整數 Dense/Linear 層
    
    輸入: int8 [batch, ..., in_features]
    權重: int8 [out_features, in_features]
    輸出: int32 [batch, ..., out_features]
    
    算法:
        output = (x @ W^T) + bias
    """
```

**特點**:
- int8 @ int8 -> int32（避免溢位）
- 支援多維輸入（自動 reshape）

#### 1.3 MatMul
```python
def int_matmul(A_int, B_int):
    """
    純整數矩陣乘法
    
    輸入: int8
    輸出: int32
    """
```

#### 1.4 GELU
```python
def int_gelu(x_int, scaling_factor, output_bit=8, n=23):
    """
    純整數 GELU 激活函數
    
    輸入: int32
    輸出: int32
    
    算法:
        GELU(x) = x * sigmoid(1.702 * x)
        其中 sigmoid(x) = exp(x) / (exp(x) + exp(-x))
    """
```

**特點**:
- 使用整數指數運算 (`int_exp_shift`)
- Stability shift: `x - max(x)`
- 精度參數 n=23

#### 1.5 Softmax
```python
def int_softmax(x_int, scaling_factor, output_bit=8, n=15):
    """
    純整數 Softmax
    
    輸入: int32
    輸出: int32
    
    算法:
        Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
```

**特點**:
- 使用整數指數運算 (`int_exp_shift`)
- Stability shift: `x - max(x)`
- 精度參數 n=15

#### 1.6 整數指數運算
```python
def int_exp_shift(x_int, x0_int, n):
    """
    整數指數運算（使用位移和多項式近似）
    
    算法:
        1. Polynomial approximation: x + x/2 - x/16
        2. Clamping: max(x, n * x0_int)
        3. Division: q = x // x0_int, r = x - q * x0_int
        4. Base: exp_base = r/2 - x0_int
        5. Shift: exp_base << (n - q) or exp_base >> (q - n)
    """
```

**特點**:
- 使用位移運算（`<<`, `>>`）代替乘除法
- 多項式近似提高精度
- 動態位移避免溢位

### 2. 量化/反量化模組

#### 2.1 Quantize
```python
def quantize_to_int(x_float, scaling_factor, bits=8):
    """
    量化到整數
    
    支援:
        - Scalar scaling factor
        - Per-channel scaling factor
        - 8/16/32 bits
    """
```

#### 2.2 Dequantize
```python
def dequantize_to_float(x_int, scaling_factor):
    """
    反量化到浮點
    
    支援:
        - Scalar scaling factor
        - Per-channel scaling factor
    """
```

#### 2.3 Requantize
```python
def requantize(x_int, input_sf, output_sf, output_bits=8):
    """
    Requantization: 從一個量化範圍轉換到另一個量化範圍
    
    算法:
        1. scale = input_sf / output_sf
        2. output = round(x * scale)
        3. Clip to output range
    """
```

**用途**:
- 在不同層之間轉換量化範圍
- 統一 per-channel scaling factor 為 scalar

#### 2.4 QuantAct Residual
```python
def quant_act_residual(x1_int, x1_sf, x2_int, x2_sf, output_sf, output_bits=16):
    """
    QuantAct with Residual: 量化激活函數 + 殘差連接
    
    算法:
        1. 反量化: x1_float = x1_int * x1_sf, x2_float = x2_int * x2_sf
        2. 相加: y_float = x1_float + x2_float
        3. 量化: y_int = round(y_float / output_sf)
    """
```

**用途**:
- Transformer 的殘差連接
- 合併兩個不同 scaling factor 的整數張量

### 3. 高階模組 (`I-ViT/pure_numpy_vit_inference.py`)

#### 3.1 Attention
```python
def int_attention(x_int, x_sf, qkv_weight, qkv_bias, qkv_sf,
                  proj_weight, proj_bias, proj_sf,
                  num_heads, scale):
    """
    純整數 Multi-Head Self-Attention
    
    流程:
        1. QKV projection: int8 @ int8 -> int32
        2. Reshape: [B, N, 3*C] -> [3, B, num_heads, N, head_dim]
        3. Split: Q, K, V
        4. Q @ K^T: int8 @ int8 -> int32
        5. Scale: attn * (1/sqrt(head_dim))
        6. Softmax: int32 -> int32
        7. Attn @ V: int8 @ int8 -> int32
        8. Projection: int8 @ int8 -> int32
        9. Output: int16
    """
```

**特點**:
- 完全純整數運算
- 使用 `int_softmax` 處理注意力權重
- 多次 requantization 保持數值穩定

#### 3.2 MLP
```python
def int_mlp(x_int, x_sf, fc1_weight, fc1_bias, fc1_sf,
            fc2_weight, fc2_bias, fc2_sf):
    """
    純整數 MLP (Feed-Forward Network)
    
    流程:
        1. FC1: int8 @ int8 -> int32
        2. GELU: int32 -> int32
        3. FC2: int8 @ int8 -> int32
        4. Output: int16
    """
```

**特點**:
- 使用 `int_gelu` 處理激活函數
- 兩層全連接層

#### 3.3 Transformer Block
```python
def int_transformer_block(x_int, x_sf, block_weights, dim_sqrt):
    """
    純整數 Transformer Block
    
    流程:
        1. Norm1 -> Attention -> Residual1
        2. Norm2 -> MLP -> Residual2
    """
```

**特點**:
- 完整的 Transformer block
- 兩個殘差連接
- 所有運算都是純整數

## 數據類型設計

### 位元寬度選擇

| 模組 | 輸入 | 權重 | 中間值 | 輸出 | 原因 |
|------|------|------|--------|------|------|
| LayerNorm | int16 | - | int64 | int32 | 需要計算 variance (sum of squares) |
| Dense/Linear | int8 | int8 | int32 | int32 | int8 @ int8 最大 = 192 * 128 * 128 ≈ 3M |
| MatMul | int8 | int8 | int32 | int32 | 同上 |
| GELU | int32 | - | int64 | int32 | 指數運算需要高精度 |
| Softmax | int32 | - | int64 | int32 | 指數運算需要高精度 |
| Attention | int8 | int8 | int32 | int16 | 多層累積，使用 int16 輸出 |
| MLP | int8 | int8 | int32 | int16 | 同上 |
| Residual | int16 | - | float32 | int16 | 需要反量化相加再量化 |

### Scaling Factor 管理

每個模組都需要追蹤 scaling factor：

```python
# 輸入
x_int: int16, x_sf: float

# LayerNorm
norm_int: int32, norm_sf: per-channel float
norm_unified_int: int8, norm_unified_sf: scalar float

# Dense
dense_int: int32, dense_sf = input_sf * weight_sf

# GELU
gelu_int: int32, gelu_sf = input_sf

# Softmax
softmax_int: int32, softmax_sf = input_sf

# Residual
residual_int: int16, residual_sf = (sf1 + sf2) / 2
```

## RTL 實現建議

### 1. 模組優先級

**Phase 1: 核心模組**（2-3 週）
1. LayerNorm（最高優先級，correlation 1.0）
2. Dense/Linear
3. MatMul

**Phase 2: 非線性模組**（2-3 週）
4. GELU
5. Softmax
6. 整數指數運算

**Phase 3: 系統整合**（1-2 週）
7. Attention
8. MLP
9. Transformer Block

### 2. RTL 設計考量

#### 2.1 LayerNorm
- **輸入**: 16-bit signed integer
- **輸出**: 32-bit signed integer
- **關鍵運算**:
  - Mean: 累加器（需要 log2(N) + 16 bits）
  - Variance: 平方累加器（需要 log2(N) + 32 bits）
  - Sqrt: Newton iteration（10 次迭代）
  - Division: 使用 reciprocal approximation

#### 2.2 Dense/Linear
- **輸入**: 8-bit signed integer
- **權重**: 8-bit signed integer
- **輸出**: 32-bit signed integer
- **關鍵運算**:
  - MAC (Multiply-Accumulate): 8x8=16, 累加到 32-bit
  - 可以使用 DSP slice 加速

#### 2.3 GELU/Softmax
- **輸入**: 32-bit signed integer
- **輸出**: 32-bit signed integer
- **關鍵運算**:
  - 位移運算（`<<`, `>>`）
  - 整數除法（可以用查表或 reciprocal approximation）
  - 比較運算（max, clamp）

### 3. 記憶體需求

以 DeiT-Tiny 為例：
- **輸入**: 1 x 197 x 192 x 2 bytes = 75 KB
- **權重**: 
  - QKV: 192 x 576 x 1 byte = 110 KB
  - Proj: 192 x 192 x 1 byte = 37 KB
  - FC1: 192 x 768 x 1 byte = 147 KB
  - FC2: 768 x 192 x 1 byte = 147 KB
  - **總計**: ~440 KB per block x 12 blocks = 5.3 MB

- **中間值**: 
  - Attention: 1 x 3 x 197 x 64 x 4 bytes = 150 KB
  - MLP: 1 x 197 x 768 x 4 bytes = 605 KB

### 4. 性能估算

假設 100 MHz 時鐘頻率：

| 模組 | 運算量 | 週期數 | 時間 (ms) |
|------|--------|--------|-----------|
| LayerNorm | 197 x 192 | ~40K | 0.4 |
| QKV Dense | 197 x 192 x 576 | ~22M | 220 |
| Q@K^T | 197 x 197 x 64 | ~2.5M | 25 |
| Softmax | 197 x 197 | ~40K | 0.4 |
| Attn@V | 197 x 197 x 64 | ~2.5M | 25 |
| Proj Dense | 197 x 192 x 192 | ~7M | 70 |
| FC1 Dense | 197 x 192 x 768 | ~29M | 290 |
| GELU | 197 x 768 | ~150K | 1.5 |
| FC2 Dense | 197 x 768 x 192 | ~29M | 290 |
| **Per Block** | | ~92M | **920** |
| **12 Blocks** | | ~1.1B | **11,040** |

**總推論時間**: ~11 秒 @ 100 MHz

**加速方法**:
1. 提高時鐘頻率（200 MHz -> 5.5 秒）
2. 並行化（4x -> 2.75 秒）
3. 使用 DSP slice（2x -> 1.4 秒）

## 驗證方法

### 1. 單元測試

測試每個基礎模組：

```bash
cd /mnt/c/桌面/冠泓/大學/專題/I-ViT/I_VIT
source /home/jin25/tvm_env_310/bin/activate
python cmodel_rtl_reference/pure_numpy_cmodel.py
```

預期輸出：
```
純 NumPy C-Model 測試
================================================================================

[1] 測試 LayerNorm
  輸入: shape=(1, 10, 192), dtype=int16, range=[-1000, 999]
  輸出: shape=(1, 10, 192), dtype=int32, range=[...]

[2] 測試 Dense
  ...

✓ 所有測試完成
```

### 2. 模組測試

測試 Attention 和 MLP：

```bash
python I-ViT/pure_numpy_vit_inference.py
```

### 3. 端到端測試

（待實現）使用真實圖片測試完整推論流程。

## 與 PyTorch C-Model 的比較

| 特性 | PyTorch C-Model | 純 NumPy C-Model |
|------|----------------|------------------|
| 依賴 | PyTorch | 只有 NumPy |
| LayerNorm | ✓ 純整數 | ✓ 純整數 |
| Dense/Linear | ✗ 使用 PyTorch | ✓ 純整數 |
| Attention | ✗ 使用 PyTorch | ✓ 純整數 |
| GELU | ✓ 純整數（算法） | ✓ 純整數 |
| Softmax | ✓ 純整數（算法） | ✓ 純整數 |
| QuantAct | ✗ 使用 PyTorch | ✓ 純整數 |
| RTL 就緒 | 部分 | ✓ 完全 |

## 下一步

1. **完成端到端推論**
   - 實現完整的 ViT 推論流程
   - 使用真實圖片測試
   - 驗證預測準確度

2. **優化性能**
   - 減少 requantization 次數
   - 優化 scaling factor 管理
   - 減少記憶體使用

3. **RTL 實現**
   - 從 LayerNorm 開始
   - 逐步實現其他模組
   - 使用 golden patterns 驗證

4. **文檔完善**
   - 添加更多範例
   - 詳細的 API 文檔
   - RTL 實現指南

## 參考資料

- `cmodel_rtl_reference/pytorch_integer_cmodel.py`: PyTorch 整數 C-Model
- `cmodel_rtl_reference/pure_numpy_cmodel.py`: 純 NumPy 基礎模組
- `I-ViT/pure_numpy_vit_inference.py`: 純 NumPy ViT 推論
- `VALIDATION_REPORT_100_IMAGES.md`: 驗證報告
- `CMODEL_COMPARISON.md`: C-Model 比較分析
