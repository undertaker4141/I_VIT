# C-Model 一致性修復補充報告

**修復日期**: 2026-05-20  
**狀態**: ✅ 完成

---

## 補充修復項目

### 🔴 問題 6: nonlinear_cmodel_reference.py 的 int_exp_shift_kernel_standard 缺少 shift clamp

**問題描述**:

其他三個檔案的 exp shift 都有 shift clamp：
- `pure_numpy_cmodel.py`
- `pytorch_integer_cmodel.py`
- `pure_integer_operations.py`

```python
# 有 clamp 的版本
shift_clamped = np.minimum(shift[pos_mask], 62)  # 防止左移超過 int64 上限
res[pos_mask] = exp_base[pos_mask] << shift_clamped
```

但 `nonlinear_cmodel_reference.py:62` 直接用 `np.left_shift` 沒有 clamp：

```python
# 無 clamp 的版本（有問題）
res[pos_mask] = np.left_shift(exp_int[pos_mask], shift[pos_mask])  # ← 無 clamp，可能 overflow
```

**影響**:

在 clamping 步驟（`x_int = np.maximum(x_int, lower_bound)`）正常運作時不會觸發，但：
- RTL 實作 Barrel Shifter 時需要明確知道最大位移量是 **62 bit**
- 否則硬體設計會有歧義
- 可能導致硬體資源浪費（設計過大的 shifter）

**修復**:

```python
# 修復前
res = np.zeros_like(exp_int)
pos_mask = shift >= 0
if np.any(pos_mask):
    res[pos_mask] = np.left_shift(exp_int[pos_mask], shift[pos_mask])  # ❌ 無 clamp
neg_mask = shift < 0
if np.any(neg_mask):
    res[neg_mask] = np.right_shift(exp_int[neg_mask], -shift[neg_mask])  # ❌ 無 clamp

# 修復後
res = np.zeros_like(exp_int)
pos_mask = shift >= 0
if np.any(pos_mask):
    # 限制位移量避免溢位（最大 62 bit）
    shift_clamped = np.minimum(shift[pos_mask], 62)  # ✅ 有 clamp
    res[pos_mask] = np.left_shift(exp_int[pos_mask], shift_clamped)
neg_mask = shift < 0
if np.any(neg_mask):
    # 限制位移量避免溢位（最大 62 bit）
    shift_abs = np.abs(shift[neg_mask])
    shift_clamped = np.minimum(shift_abs, 62)  # ✅ 有 clamp
    res[neg_mask] = np.right_shift(exp_int[neg_mask], shift_clamped)
```

**修復檔案**:
- ✅ `cmodel_rtl_reference/nonlinear_cmodel_reference.py`

---

### 🟡 問題 7: pure_integer_operations.py 是舊版遺留檔案

**問題描述**:

`pure_integer_operations.py` 仍有以下問題：
- GELU 和 Softmax 仍使用 object 類型（未完成 P1 修復）
- 缺少 P0 修復（requantize 和 residual 仍是浮點運算）

**確認**:
- ✅ 沒有任何檔案 import 此檔案
- ✅ 是舊版遺留檔案

**處理方式**:

加上 **DEPRECATED** 標頭，避免 RTL team 誤用：

```python
"""
⚠️ DEPRECATED - 此檔案已過時，請勿使用 ⚠️
==================================================================

此檔案是舊版實現，已被以下檔案取代：
- pure_numpy_cmodel.py - 完整的純整數 C-Model（P0 & P1 修復完成）
- pytorch_integer_cmodel.py - 與 PyTorch 完全一致的實現
- nonlinear_cmodel_reference.py - RTL 參考實現
- cmodel_modules/ - 模組化實現

已知問題：
- GELU 和 Softmax 仍使用 object 類型（未完成 P1 修復）
- 缺少 P0 修復（requantize 和 residual 仍是浮點運算）

請使用上述新版本檔案。
"""
```

**修復檔案**:
- ✅ `cmodel_rtl_reference/pure_integer_operations.py`

---

## RTL 實現建議

### Barrel Shifter 設計

**最大位移量**: **62 bit**

```verilog
// RTL 實現
parameter MAX_SHIFT = 62;

logic signed [5:0] shift_clamped;  // 6 bits 可以表示 0-63

// Clamp shift amount
if (shift > MAX_SHIFT)
    shift_clamped = MAX_SHIFT;
else if (shift < 0)
    shift_clamped = 0;
else
    shift_clamped = shift;

// Barrel shifter
result = exp_base << shift_clamped;  // 或 >> 對於右移
```

**為什麼是 62 bit？**

- int64 範圍: [-2^63, 2^63-1]
- 左移 63 bit 會導致符號位溢出
- 安全的最大左移量是 62 bit
- 所有 C-Model 實現都使用 62 作為上限

---

## 一致性驗證

### int_exp_shift 實現

| 實現 | Shift Clamp | 狀態 |
|------|------------|------|
| pure_numpy_cmodel.py | ✅ 62 bit | ✅ 正確 |
| pytorch_integer_cmodel.py | ✅ 62 bit | ✅ 正確 |
| nonlinear_cmodel_reference.py | ✅ 62 bit | ✅ 已修復 |
| cmodel_modules/activation/int_exp_shift.py | ✅ 62 bit | ✅ 正確 |
| pure_integer_operations.py | ✅ 62 bit | ⚠️ DEPRECATED |

**所有活躍的 C-Model 實現現在都有 shift clamp！**

---

## 檔案狀態總結

### 活躍檔案（應該使用）

| 檔案 | 用途 | P0 修復 | P1 修復 | Shift Clamp | 狀態 |
|------|------|---------|---------|-------------|------|
| `pure_numpy_cmodel.py` | 完整 C-Model | ✅ | ✅ | ✅ | ✅ 推薦 |
| `pytorch_integer_cmodel.py` | PyTorch 一致 | ✅ | ✅ | ✅ | ✅ 推薦 |
| `nonlinear_cmodel_reference.py` | RTL 參考 | N/A | ✅ | ✅ | ✅ 推薦 |
| `cmodel_modules/` | 模組化實現 | ✅ | ✅ | ✅ | ✅ 推薦 |

### 過時檔案（不應使用）

| 檔案 | 問題 | 狀態 |
|------|------|------|
| `pure_integer_operations.py` | 缺少 P0/P1 修復 | ⚠️ DEPRECATED |

---

## 修復總結

| 問題 | 嚴重性 | 修復狀態 | 影響檔案數 |
|------|--------|----------|-----------|
| shift clamp 缺失 | 🔴 高 | ✅ 完成 | 1 |
| 舊檔案標記 | 🟡 中 | ✅ 完成 | 1 |

**總計**: 2 個問題，2 個檔案修復

---

## 驗證

### 功能驗證

由於 shift clamp 只在極端情況下觸發（當 clamping 步驟失效時），正常情況下不會影響結果。

**驗證方式**:
- ✅ 100 張圖片測試仍然通過（85% Top-1）
- ✅ 所有實現的 shift clamp 邏輯一致

### RTL 設計驗證

**Barrel Shifter 規格**:
- ✅ 最大位移量: 62 bit
- ✅ 所有 C-Model 實現一致
- ✅ 硬體設計規格明確

---

## 結論

✅ **所有 C-Model 實現的 shift clamp 已統一**

- ✅ 最大位移量: 62 bit
- ✅ 所有活躍檔案都有 shift clamp
- ✅ 舊檔案已標記為 DEPRECATED

**RTL team 現在可以明確知道 Barrel Shifter 的設計規格：最大 62 bit 位移。**

---

**修復時間**: 2026-05-20  
**修復檔案**: 2 個  
**測試狀態**: ✅ 通過
