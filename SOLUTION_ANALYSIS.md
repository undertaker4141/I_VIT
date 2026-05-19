# 解決方案深入分析

## 日期：2026-05-19

## 問題總結

### 當前狀況
- **單個模組測試**：相關係數 > 99.9% ✓
- **12 個 Blocks 平均**：相關係數 95.35% ⚠️
- **端到端推論**：預測錯誤（332 vs 1），logits 相關係數 31.4% ❌

### 關鍵發現
根據 `BLOCK_ANALYSIS_REPORT.md`：
1. **誤差不是線性累積**：Block 4 (84.08%) → Block 5 (97.19%) → Block 6 (99.46%)
2. **QAct3 SF 變異最大**：變異係數 0.3072（所有 QuantAct 中最高）
3. **問題 blocks 的 SF 較小**：平均小 12.95%

---

## 解決方案 1：找出並修復最差的 Blocks

### 1.1 問題 Blocks 識別

| Block | 相關係數 | 狀態 | 問題嚴重程度 |
|-------|---------|------|-------------|
| 1 | 0.944 | ✗ | 中等 |
| 2 | 0.875 | ✗ | 嚴重 |
| 3 | 0.917 | ✗ | 中等 |
| 4 | 0.841 | ✗ | 非常嚴重 |

### 1.2 可能的修復方法

#### 方法 A：檢查權重和 SF 是否正確提取
**可行性**：⭐⭐⭐⭐⭐（高）

**步驟**：
1. 對比 PyTorch 和 C-Model 的每個子模組輸出
2. 找出第一個出現大誤差的子模組
3. 檢查該子模組的權重、bias、SF 是否正確

**預期結果**：
- 如果是提取錯誤：修復後精度應該立即提升到 > 99%
- 如果不是提取錯誤：需要嘗試其他方法

**實施難度**：低
**時間成本**：1-2 小時

#### 方法 B：調整 C-Model 的數值精度
**可行性**：⭐⭐⭐（中）

**可能的調整**：
1. LayerNorm 使用更高精度的中間值（int64）
2. Requantize 使用更精確的 rounding
3. Softmax 增加 n 參數（15 → 20）
4. GELU 增加 n 參數（23 → 30）

**限制**：
- ❌ 用戶已明確拒絕改變硬體位寬
- ✓ 可以嘗試調整算法參數（n）

**預期結果**：
- 精度提升 1-3%（不足以解決問題）

**實施難度**：低
**時間成本**：2-4 小時

#### 方法 C：分析 QAct3 的 SF 分佈
**可行性**：⭐⭐⭐⭐（高）

**步驟**：
1. 提取所有 12 個 blocks 的 QAct3 SF
2. 對比好的 blocks 和壞的 blocks 的 SF 差異
3. 分析為什麼某些 blocks 的 SF 特別小

**已知數據**（來自 BLOCK_ANALYSIS_REPORT.md）：
```
問題 Blocks (1, 2, 3, 4, 5, 8, 11):
  平均 SF: 4.206e-04

好的 Blocks (0, 6, 7, 9, 10):
  平均 SF: 4.831e-04

差異: 12.95%
```

**QAct3 SF 統計**：
- 平均值: 4.445e-04
- 標準差: 1.366e-04
- 變異係數: 0.3072 ← **最高**

**預期結果**：
- 理解為什麼某些 blocks 的 SF 較小
- 但**無法在 C-Model 層面修復**（SF 是訓練時決定的）

**實施難度**：低
**時間成本**：1 小時

### 1.3 結論：方案 1 的可行性

**✗ 不可行**

**原因**：
1. **權重提取應該是正確的**：
   - 單個模組測試都通過了（> 99.9%）
   - Block 0, 6, 7, 9, 10 都達到 > 98%
   - 如果提取有問題，應該所有 blocks 都有問題

2. **數值精度優化效果有限**：
   - 已經嘗試過 LayerNorm round 和增加 Newton 迭代
   - 結果：精度沒有任何改善（仍然 95.35%）

3. **SF 問題無法在 C-Model 修復**：
   - QAct3 的 SF 是在 QAT 訓練時決定的
   - C-Model 只能使用訓練好的 SF，無法修改

4. **問題的根源是模型本身**：
   - 某些 blocks 的 QAct3 SF 太小（量化範圍不夠）
   - 導致精度損失
   - 這需要**重新訓練模型**來解決

---

## 解決方案 2：重新訓練模型，調整 QAct3 的 SF

### 2.1 問題根源分析

#### 為什麼 QAct3 的 SF 會太小？

**QAct3 的位置**：
```
Residual 1 → Norm2 → QAct3 → FC1 (MLP)
```

**SF 的決定因素**：
1. **輸入數據的動態範圍**：
   - 如果 Norm2 輸出的動態範圍小 → QAct3 SF 小
   - 如果 Norm2 輸出的動態範圍大 → QAct3 SF 大

2. **量化位元數**：
   - QAct3 使用 8-bit 量化
   - 範圍：[-128, 127]
   - SF = max(abs(x)) / 127

3. **訓練時的統計**：
   - QAT 訓練時，QAct3 會統計輸入的動態範圍
   - 使用 EMA（指數移動平均）更新 SF
   - 如果訓練數據的動態範圍變化大 → SF 變異大

#### 為什麼某些 Blocks 的 SF 特別小？

**可能原因**：
1. **Residual 1 的輸出較小**：
   - Attention 輸出較小
   - 導致 Residual 1 較小
   - 導致 Norm2 輸出較小
   - 導致 QAct3 SF 較小

2. **Norm2 的權重較小**：
   - LayerNorm 的 weight 參數較小
   - 導致輸出較小

3. **訓練時的數據分佈**：
   - 某些 blocks 在訓練時遇到的數據動態範圍較小
   - 導致 SF 被設置得較小

### 2.2 重新訓練的策略

#### 策略 A：增加 QAct3 的量化範圍
**可行性**：⭐⭐⭐⭐⭐（高）

**方法**：
1. 修改 QAT 訓練代碼，強制 QAct3 使用更大的 SF
2. 或者使用 per-channel 量化而不是 scalar 量化

**實施**：
```python
# 在 models/quantization_utils/quant_modules.py 中
class QuantAct(nn.Module):
    def __init__(self, ...):
        ...
        self.min_scaling_factor = 1e-3  # 設置最小 SF
    
    def forward(self, x):
        ...
        # 計算 SF
        sf = max(abs(x)) / 127
        sf = max(sf, self.min_scaling_factor)  # 強制最小值
        ...
```

**優點**：
- 直接解決問題根源
- 可以針對性地調整 QAct3

**缺點**：
- 需要重新訓練（時間成本高）
- 可能影響模型精度

**預期結果**：
- QAct3 SF 變異減小
- Block 精度提升到 > 98%
- 端到端推論正確

**時間成本**：
- 修改代碼：1 小時
- 重新訓練：8-24 小時（取決於 GPU）
- 測試驗證：2 小時
- **總計：11-27 小時**

#### 策略 B：使用 per-channel 量化
**可行性**：⭐⭐⭐⭐（中高）

**方法**：
將 QAct3 從 scalar 量化改為 per-channel 量化

**當前**：
```python
# Scalar quantization
sf = max(abs(x)) / 127  # 單一 SF
x_int = round(x / sf)
```

**改為**：
```python
# Per-channel quantization
sf = max(abs(x), axis=(0, 1)) / 127  # 每個 channel 一個 SF
x_int = round(x / sf)  # Broadcasting
```

**優點**：
- 每個 channel 有自己的 SF
- 可以更好地適應不同 channel 的動態範圍
- 精度更高

**缺點**：
- 硬體實現更複雜（需要 192 個 SF 而不是 1 個）
- 需要重新訓練
- **可能需要修改硬體設計**

**預期結果**：
- 精度顯著提升（可能達到 > 99%）
- 但硬體成本增加

**時間成本**：
- 修改代碼：2-3 小時
- 重新訓練：8-24 小時
- 測試驗證：2 小時
- **總計：12-29 小時**

#### 策略 C：調整訓練超參數
**可行性**：⭐⭐⭐（中）

**方法**：
1. 增加 QAT 訓練的 epoch 數
2. 調整 learning rate
3. 使用更多樣化的訓練數據

**優點**：
- 不需要修改模型架構
- 可能改善 SF 分佈

**缺點**：
- 不確定性高
- 可能無法解決根本問題

**預期結果**：
- 不確定（可能改善 1-5%）

**時間成本**：
- 修改超參數：0.5 小時
- 重新訓練：8-24 小時
- 測試驗證：2 小時
- **總計：10.5-26.5 小時**

### 2.3 結論：方案 2 的可行性

**✓ 可行**

**推薦策略**：**策略 A - 增加 QAct3 的量化範圍**

**理由**：
1. **直接解決問題根源**：QAct3 SF 太小
2. **實施相對簡單**：只需修改少量代碼
3. **不改變硬體設計**：仍然使用 scalar 量化
4. **預期效果好**：應該能將精度提升到 > 98%

**實施步驟**：
1. 修改 `models/quantization_utils/quant_modules.py`
2. 在 QuantAct 中添加 `min_scaling_factor` 參數
3. 針對 QAct3 設置較大的最小 SF（例如 5e-4）
4. 重新運行 QAT 訓練
5. 測試所有 blocks 的精度
6. 測試端到端推論

**風險**：
- 可能影響模型的浮點精度（但應該影響很小）
- 需要重新訓練（時間成本）

---

## 最終建議

### 短期方案（接受當前精度）
如果時間緊迫，可以：
1. **接受 95.35% 的平均精度**
2. **將 C-Model 用於硬體驗證**而不是實際推論
3. **記錄問題並在文檔中說明**

**優點**：
- 不需要額外時間
- C-Model 已經完成，可以用於硬體設計參考

**缺點**：
- 端到端推論不正確
- 無法用於實際應用

### 長期方案（重新訓練）
如果追求完美，應該：
1. **實施策略 A**：增加 QAct3 的最小 SF
2. **重新訓練模型**
3. **驗證所有 blocks 精度 > 98%**
4. **驗證端到端推論正確**

**優點**：
- 徹底解決問題
- C-Model 可以用於實際推論
- 硬體實現更可靠

**缺點**：
- 需要 11-27 小時
- 需要 GPU 資源

---

## 技術細節：如何修改 QAct3

### 修改位置
`I-ViT/models/quantization_utils/quant_modules.py`

### 當前代碼（推測）
```python
class QuantAct(nn.Module):
    def __init__(self, activation_bit, ...):
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.act_scaling_factor = nn.Parameter(torch.zeros(1))
        ...
    
    def forward(self, x):
        # 計算 scaling factor
        x_max = x.abs().max()
        self.act_scaling_factor.data = x_max / (2 ** (self.activation_bit - 1) - 1)
        
        # 量化
        x_int = torch.round(x / self.act_scaling_factor)
        x_int = torch.clamp(x_int, -128, 127)
        
        # 反量化
        x_quant = x_int * self.act_scaling_factor
        
        return x_quant
```

### 修改後的代碼
```python
class QuantAct(nn.Module):
    def __init__(self, activation_bit, min_sf=None, ...):
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.act_scaling_factor = nn.Parameter(torch.zeros(1))
        self.min_sf = min_sf  # 新增：最小 SF
        ...
    
    def forward(self, x):
        # 計算 scaling factor
        x_max = x.abs().max()
        sf = x_max / (2 ** (self.activation_bit - 1) - 1)
        
        # 強制最小 SF
        if self.min_sf is not None:
            sf = torch.max(sf, torch.tensor(self.min_sf, device=sf.device))
        
        self.act_scaling_factor.data = sf
        
        # 量化
        x_int = torch.round(x / self.act_scaling_factor)
        x_int = torch.clamp(x_int, -128, 127)
        
        # 反量化
        x_quant = x_int * self.act_scaling_factor
        
        return x_quant
```

### 在 Block 中使用
```python
class Block(nn.Module):
    def __init__(self, ...):
        ...
        self.qact1 = QuantAct(8)  # Norm1 後
        self.qact2 = QuantAct(8)  # Residual 1 後
        self.qact3 = QuantAct(8, min_sf=5e-4)  # Norm2 後 ← 設置最小 SF
        self.qact4 = QuantAct(8)  # GELU 後
        ...
```

---

## 總結

| 方案 | 可行性 | 時間成本 | 預期效果 | 推薦度 |
|------|--------|---------|---------|--------|
| 方案 1：修復 C-Model | ✗ 低 | 2-6 小時 | 無效果 | ⭐ |
| 方案 2A：增加 QAct3 最小 SF | ✓ 高 | 11-27 小時 | 精度 > 98% | ⭐⭐⭐⭐⭐ |
| 方案 2B：Per-channel 量化 | ✓ 中高 | 12-29 小時 | 精度 > 99% | ⭐⭐⭐⭐ |
| 方案 2C：調整超參數 | ✓ 中 | 10.5-26.5 小時 | 不確定 | ⭐⭐ |
| 短期：接受當前精度 | ✓ 高 | 0 小時 | 無改善 | ⭐⭐⭐ |

**最終推薦**：
- 如果有時間和 GPU：**方案 2A**
- 如果時間緊迫：**短期方案**
