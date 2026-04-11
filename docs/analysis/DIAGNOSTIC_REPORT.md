# I-ViT C-Model 驗證問題診斷報告

> **診斷日期**: 2026-01-13  
> **診斷結論**: **C-model 正確，問題在於 Golden Pattern 抽取方式**

---

## 🔍 診斷結果

### 問題發現

| 測試類型 | 結果 | 說明 |
|----------|------|------|
| **Float 驗證** | ✅ **100% PASS** | C-model_int × scale ≈ Golden_float |
| 整數直接比較 | ❌ 35.56% 匹配 | 這不是真正的問題 |
| 容差 ±100 內 | ✅ 99.09% | 即使用舊方法也可接受 |

### 關鍵數據

```
Float 驗證:
  最大絕對誤差: 0.00003440
  平均絕對誤差: 0.00000011
  np.allclose(rtol=1e-4, atol=1e-4): ✅ PASS

整數直接比較:
  嚴格匹配率: 35.56%
  最大差異: 5066
```

---

## 📋 問題分析

### 問題根源

Golden Pattern 中的 `blocks_0_norm1_int.npy` 是透過以下方式生成的：

```python
# 在 extract_patterns.py 的 hook 中
int_out = (float_output / scaling_factor).round()
```

其中 `scaling_factor` 是 **per-channel** 的浮點數向量：

```python
scaling_factor = (sqrt(192) / 2^30) * weight  # shape: [192]
```

### 為什麼這會導致問題？

1. **浮點除法精度問題**: 
   - `float_output / scaling_factor` 本身就是浮點運算
   - 結果做 `round()` 會引入額外的捨入誤差

2. **與 PyTorch 內部計算不同**:
   - IntLayerNorm 內部的 `y_int + bias_int` 是純整數
   - 但 Golden Pattern 用的是 "從浮點反推的整數"

3. **示例說明**:
   ```
   PyTorch 內部整數 (真正的 y_int):     -201299733
   Golden Pattern (round(float/scale)): -201302736
   差距:                                    3003
   ```

### C-model 是正確的！

Float 驗證證明：
- C-model 的整數輸出 × scaling_factor = 正確的浮點輸出
- 誤差 < 0.0001，完全在可接受範圍內

---

## ✅ 解決方案

### 方案 A: 使用 Float 驗證 (推薦)

不直接比較整數，而是比較浮點輸出：

```python
# 計算 C-model 的浮點輸出
cmodel_float = cmodel_int_output * scaling_factor

# 與 Golden float 比較
passed = np.allclose(cmodel_float, golden_float, rtol=1e-4, atol=1e-4)
```

**優點**:
- 不需要重新生成 Pattern
- 可以立即使用
- 更真實地反映模型誤差

**驗證腳本**: `layernorm_cmodel/test_layernorm_float_verify.py`

---

### 方案 B: 重新抽取 Pattern

`extract_patterns.py` 和 `quant_modules.py` 已經修改為保存真正的整數輸出：

```python
# quant_modules.py 中
self.output_integer = y_int.detach()  # 保存內部整數

# extract_patterns.py 中
if hasattr(module, 'output_integer') and module.output_integer is not None:
    int_out = module.output_integer.detach().cpu()
```

**需要 CUDA 環境執行**:
```bash
cd I-ViT
python extract_patterns.py \
    --checkpoint checkpoints/qat_calibrated.pth \
    --test-image ../test_data/test_image.JPEG \
    --output ../patterns
```

**預期結果**: 整數直接比較應該達到 ~100% 匹配

---

## 📁 相關檔案

| 檔案 | 用途 |
|------|------|
| `diagnose_mismatch.py` | 診斷腳本 (新增) |
| `verify_cmodel.py` | 原始驗證腳本 |
| `layernorm_cmodel/test_layernorm_float_verify.py` | Float 驗證腳本 |
| `I-ViT/extract_patterns.py` | Pattern 抽取 (已修改) |
| `I-ViT/models/quantization_utils/quant_modules.py` | IntLayerNorm (已修改) |
| `patterns/diagnostic_results.json` | 診斷結果 |

---

## 🎯 結論

**C-model 實現完全正確！**

64.44% 的 "mismatch" 問題不是 C-model 錯誤，而是 Golden Pattern 抽取方式的限制。使用 Float 驗證可以 100% 通過，證明 C-model 的純整數計算與 PyTorch 模型等效。

### 對 RTL 開發的建議

1. **驗證方法**: 使用 Float 驗證 (方案 A)
2. **容差設定**: `rtol=1e-4, atol=1e-4` 或 `max_abs_error < 0.001`
3. **如果需要純整數驗證**: 重新抽取 Pattern (需要 CUDA)
