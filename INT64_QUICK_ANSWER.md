# int64 快速解答

## 問題
「所有運算都是純整數（int8, int16, int32, int64）」- int64 哪裡來的？

## 簡短答案

**int64 只用於 GELU 和 Softmax 的指數運算中間值，最終輸出仍然是 int32。**

---

## 為什麼需要 int64？

### 1. 左移操作會溢出 int32

```python
# 在指數運算中
result = exp_base << 40  # 左移 40 位

# int32 最大值：2^31 - 1 ≈ 2.1 × 10^9
# 左移 40 位：2^40 ≈ 1.1 × 10^12  ← 超過 int32！
```

### 2. 大數乘法會溢出 int32

```python
# 在 GELU/Softmax 中
term = exp_int * factor  # 可能達到 10^18

# 必須使用 int64 或 object 類型
```

### 3. 最終輸出還是 int32

```python
def int_gelu(...):
    pre_x_int = x_int.astype(np.int64)  # 中間計算用 int64
    ...
    return output.astype(np.int32)  # 輸出是 int32！

def int_softmax(...):
    pre_x_int = x_int.astype(np.int64)  # 中間計算用 int64
    ...
    return output.astype(np.int32)  # 輸出是 int32！
```

---

## 對硬體設計的影響

### 只有 18% 的模組需要 64-bit

| 模組 | 需要 64-bit | 數量 | 百分比 |
|------|------------|------|--------|
| LayerNorm, Linear, MatMul, Requantize | ❌ | 108 | 82% |
| **GELU, Softmax** | ✅ | 24 | **18%** |

### 推薦硬體設計

**方案 A：混合 32-bit 和 64-bit**（推薦）
- 大部分模組使用 32-bit ALU
- GELU 和 Softmax 使用 64-bit ALU
- 面積節省 49%

**方案 B：使用查找表（LUT）**
- 用 LUT 近似 GELU 和 Softmax
- 面積最小
- 需要驗證精度

---

## 詳細文檔

請參考：`docs/cmodel/INT64_USAGE_EXPLANATION.md`

---

**結論**：int64 是必要的，但只用於少數模組的中間計算，不影響整體架構。
