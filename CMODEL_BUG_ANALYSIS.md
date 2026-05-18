# TVM-based C-Model Bug 分析報告

## 執行摘要

**問題**: 舊版 TVM-based C-Model（`scripts/cmodel_vit_infer.py`）準確率 0%  
**根本原因**: LayerNorm 實現包含 int16 溢位模擬，不匹配 PyTorch  
**解決方案**: 新版 PyTorch-based C-Model（`cmodel_rtl_reference/pytorch_integer_cmodel.py`）  
**結果**: 準確率從 0% 提升到 97%

---

## Bug 詳情

### Bug 1: Mean 計算的 int16 溢位

**位置**: `cmodel_rtl_reference/nonlinear_cmodel_reference.py:int_layer_norm_fixed`

**錯誤代碼**:
```python
sum_val = np.sum(x_val, axis=-1, keepdims=True)
sum_wrapped = sum_val.astype(np.int16).astype(np.int32)  # ❌ 強制 int16 溢位
mean_int = np.fix(sum_wrapped / N).astype(np.int32)
```

**問題**:
- `sum_val` 可能超出 int16 範圍 [-32768, 32767]
- 強制轉換為 int16 導致溢位
- 例如：100000 → int16 → -31072（錯誤）

**影響**: Mean 計算完全錯誤

### Bug 2: Centering 的 int16 截斷

**位置**: `cmodel_rtl_reference/nonlinear_cmodel_reference.py:int_layer_norm_fixed`

**錯誤代碼**:
```python
y_int = x_val - mean_int
y_int = y_int.astype(np.int16).astype(np.int32)  # ❌ 強制 int16 截斷
```

**問題**:
- `y_int` 可能超出 int16 範圍
- 強制轉換為 int16 導致截斷
- 例如：40000 → int16 → -25536（錯誤）

**影響**: Centered values 完全錯誤

### 累積效應

```
錯誤的 mean → 錯誤的 centering → 錯誤的 variance 
→ 錯誤的 std → 錯誤的 normalization → 0% 準確率
```

---

## 修復方案

### 正確實現

**文件**: `cmodel_rtl_reference/pytorch_integer_cmodel.py:pytorch_int_layer_norm`

**正確代碼**:
```python
# ✅ Step 1: Mean (無 int16 溢位)
mean_int = np.round(np.mean(x_int, axis=-1, keepdims=True))

# ✅ Step 2: Centering (無 int16 截斷)
y_int = x_int - mean_int
```

**關鍵改進**:
1. 使用 `np.mean` 直接計算，不強制 int16 轉換
2. Centering 保持原類型，不截斷
3. 使用 `np.round`，不是 `np.fix`

---

## 驗證結果

### 舊版（TVM-based）

| 指標 | 結果 |
|------|------|
| LayerNorm 相關係數 | < 0.5 ❌ |
| 端到端準確率 | 0% ❌ |
| Logits 相關係數 | < 0.1 ❌ |

### 新版（PyTorch-based）

| 指標 | 結果 |
|------|------|
| LayerNorm 相關係數 | 1.0000000000 ✅ |
| 端到端準確率 | 97% ✅ |
| Logits 相關係數 | 0.992129 ✅ |

---

## 教訓

1. **不要盲目套用 TVM 算法**: TVM 和 PyTorch 的整數推論算法不同
2. **int16 溢位是致命的**: 必須使用足夠大的整數類型
3. **逐模組驗證很重要**: 使用相關係數快速定位問題

---

## 建議

### ❌ 不要使用

- `scripts/cmodel_vit_infer.py`
- `cmodel_rtl_reference/nonlinear_cmodel_reference.py:int_layer_norm_fixed`

### ✅ 使用

- `cmodel_rtl_reference/pytorch_integer_cmodel.py:pytorch_int_layer_norm`
- 相關係數 1.0，準確率 97%

---

**日期**: 2026-05-18  
**狀態**: 已修復  
**詳細報告**: 見 `CMODEL_COMPARISON.md`
