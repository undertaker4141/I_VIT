# int64 使用說明

## 問題

為什麼 C-Model 需要使用 int64？不是說所有運算都是 int8/int16/int32 嗎？

## 答案

**int64 只用於 GELU 和 Softmax 的指數運算中間值，最終輸出仍然是 int32。**

---

## 詳細說明

### 1. 哪裡使用了 int64？

#### 位置 1：`int_exp_shift` 函數

```python
def int_exp_shift(x_int, x0_int, n):
    """整數指數運算"""
    # 輸入轉換為 int64
    x_int = x_int.astype(np.int64)
    x0_int = np.int64(x0_int)
    n = np.int64(n)
    
    # ... 中間計算 ...
    
    # 左移操作（可能左移最多 62 位）
    result[pos_mask] = exp_base[pos_mask] << shift_clamped
    
    # 返回 int64
    return result.astype(np.int64)
```

#### 位置 2：`int_gelu` 函數

```python
def int_gelu(x_int, scaling_factor, output_bit=8, n=23):
    """純整數 GELU"""
    # 輸入轉換為 int64
    pre_x_int = x_int.astype(np.int64)
    
    # 調用 int_exp_shift（返回 int64）
    exp_int = int_exp_shift(x_algo, x0_int, n)
    
    # 大數乘法（使用 object 避免溢出）
    term = exp_int.astype(object) * factor.astype(object)
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # 最終輸出轉換為 int32
    return output.astype(np.int32)  # ← 注意：輸出是 int32！
```

#### 位置 3：`int_softmax` 函數

```python
def int_softmax(x_int, scaling_factor, output_bit=8, n=15):
    """純整數 Softmax"""
    # 輸入轉換為 int64
    pre_x_int = x_int.astype(np.int64)
    
    # 調用 int_exp_shift（返回 int64）
    exp_int = int_exp_shift(x_algo, x0_int, n)
    
    # 大數乘法
    term = exp_int.astype(object) * factor.astype(object)
    output = (term >> shift_amt).astype(np.int64)
    
    # 最終輸出轉換為 int32
    return output.astype(np.int32)  # ← 注意：輸出是 int32！
```

---

## 為什麼需要 int64？

### 原因 1：左移操作會溢出 int32

在指數運算中，需要進行動態位移：

```python
shift = n - q  # n 可能是 23（GELU）或 15（Softmax）
result = exp_base << shift  # 左移最多 62 位
```

**問題**：
- int32 最大值：2^31 - 1 ≈ 2.1 × 10^9
- 左移 40 位：2^40 ≈ 1.1 × 10^12（**超過 int32 範圍**）
- 必須使用 int64 才能存儲

**範例**：
```python
# 假設 exp_base = 1000
# shift = 40

# 使用 int32（錯誤）
result_int32 = np.int32(1000) << 40  # 溢出！結果錯誤

# 使用 int64（正確）
result_int64 = np.int64(1000) << 40  # = 1,099,511,627,776（正確）
```

### 原因 2：大數乘法會溢出 int32

在 GELU 和 Softmax 中，需要計算：

```python
term = exp_int * factor
```

其中：
- `exp_int` 可能是 int64 的大數（例如 10^12）
- `factor` 也可能很大（例如 10^6）
- 乘法結果：10^18（**遠超 int32 範圍**）

**解決方案**：
```python
# 使用 object 類型避免溢出
term = exp_int.astype(object) * factor.astype(object)

# 然後右移縮小
output = (term >> shift_amt).astype(np.int64)
```

### 原因 3：PyTorch 的原始實現也是這樣

PyTorch 的 `IntGELU` 和 `IntSoftmax` 也使用 64-bit 中間值：

```python
# PyTorch IntGELU 源碼（簡化）
def forward(self, x):
    x_int = x.to(torch.int64)  # 轉換為 int64
    exp_int = int_exp_shift(x_int, ...)  # int64 運算
    output = (exp_int * factor) >> shift  # int64 運算
    return output.to(torch.int32)  # 轉換回 int32
```

---

## 對硬體設計的影響

### 模組分類

| 模組類型 | 需要 64-bit | 數量 | 百分比 |
|---------|------------|------|--------|
| LayerNorm | ❌ | 24 | 18% |
| Linear/Dense | ❌ | 36 | 27% |
| MatMul | ❌ | 12 | 9% |
| **GELU** | ✅ | 12 | 9% |
| **Softmax** | ✅ | 12 | 9% |
| Requantize | ❌ | 48 | 36% |
| **總計** | - | **132** | **100%** |

**結論**：
- 只有 **24 個模組**（18%）需要 64-bit 運算
- 其他 **108 個模組**（82%）只需要 32-bit 運算

### 硬體資源估算

假設：
- 32-bit ALU 面積：1x
- 64-bit ALU 面積：2.5x

**方案 A：全部使用 64-bit**
- 總面積：132 × 2.5x = **330x**
- 優點：設計簡單
- 缺點：面積浪費

**方案 B：混合使用**（推薦）
- 32-bit 模組：108 × 1x = 108x
- 64-bit 模組：24 × 2.5x = 60x
- 總面積：**168x**
- 優點：面積節省 49%
- 缺點：設計稍複雜

**方案 C：使用查找表（LUT）**
- 用 LUT 近似 GELU 和 Softmax
- 總面積：108 × 1x + 24 × 0.5x = **120x**
- 優點：面積最小
- 缺點：精度可能下降，需要重新驗證

---

## 硬體實現建議

### 推薦方案：混合 32-bit 和 64-bit

#### 1. 設計兩種運算單元

**32-bit ALU**（用於大部分模組）：
- LayerNorm
- Linear/Dense
- MatMul
- Requantize

**64-bit ALU**（只用於 GELU 和 Softmax）：
- GELU 的指數運算
- Softmax 的指數運算

#### 2. 數據流設計

```
輸入 (int32) → GELU/Softmax (內部 int64) → 輸出 (int32)
                    ↑
                64-bit ALU
```

**關鍵點**：
- 輸入和輸出都是 int32
- 只有內部計算使用 int64
- 不需要 64-bit 的數據總線

#### 3. 面積優化

如果面積受限，可以考慮：

**選項 A：時分複用**
- 一個 64-bit ALU 被多個 GELU/Softmax 共享
- 串行執行，增加延遲但減少面積

**選項 B：分段計算**
- 將 64-bit 運算拆分為兩個 32-bit 運算
- 高位和低位分別計算
- 面積不變，但控制邏輯複雜

**選項 C：近似算法**
- 使用查找表（LUT）近似指數函數
- 測試精度損失是否可接受
- 可能需要重新訓練模型

---

## 驗證建議

### 1. 精度驗證

如果使用近似算法（LUT），需要驗證：

```python
# 測試 GELU 近似精度
def test_gelu_approximation():
    # 原始 GELU
    gelu_exact = int_gelu(x, sf, output_bit=8, n=23)
    
    # 近似 GELU（使用 LUT）
    gelu_approx = int_gelu_lut(x, sf, output_bit=8)
    
    # 比較
    corr = np.corrcoef(gelu_exact.flatten(), gelu_approx.flatten())[0, 1]
    print(f"GELU 近似相關係數: {corr:.6f}")
    
    # 要求：相關係數 > 0.999
    assert corr > 0.999
```

### 2. 端到端驗證

修改 C-Model 後，必須重新測試：

```bash
# 測試所有 blocks
python I-ViT/test_all_blocks.py

# 測試驗證集
python I-ViT/test_100_images_pure_integer.py
```

**要求**：
- Block 平均精度：> 95%
- 驗證集一致率：> 95%
- Logits 相關係數：> 99%

---

## 常見問題

### Q1: 可以完全避免 int64 嗎？

**A**: 理論上可以，但會犧牲精度或增加複雜度：

1. **使用 LUT 近似**：精度可能下降
2. **分段計算**：控制邏輯複雜
3. **降低精度參數 n**：會影響 GELU/Softmax 精度

### Q2: 為什麼不直接使用浮點數？

**A**: 
1. 硬體目標是純整數運算（面積小、功耗低）
2. 浮點運算單元面積是整數的 5-10 倍
3. 量化推論的目的就是避免浮點運算

### Q3: 其他量化模型也需要 int64 嗎？

**A**: 
- 取決於激活函數的實現
- 如果使用查找表（LUT）近似，可能不需要
- 如果使用位移和多項式近似（如 PyTorch），通常需要

### Q4: 可以用 int48 嗎？

**A**: 
- 理論上可以，但硬體設計更複雜
- 大部分硬體只支持 8/16/32/64 bit
- 使用非標準位寬會增加設計和驗證成本

---

## 總結

1. **int64 只用於 GELU 和 Softmax 的中間計算**
2. **最終輸出仍然是 int32**
3. **只有 18% 的模組需要 64-bit 運算**
4. **推薦使用混合 32-bit 和 64-bit 設計**
5. **如果面積受限，可以考慮 LUT 近似**

---

**文檔版本**: 1.0  
**日期**: 2026-05-19  
**作者**: Kiro AI Assistant
