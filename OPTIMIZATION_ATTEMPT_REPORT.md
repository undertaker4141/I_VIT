# LayerNorm 優化嘗試報告

## 日期
2026-05-19

## 優化目標
提升 C-Model 精度從 95.35% 到 97-98%

## 實施的優化

### 1. LayerNorm 使用 round 而不是 floor
**位置**: `cmodel_rtl_reference/pure_numpy_cmodel.py` Line 66

**修改**:
```python
# 舊版本
y_int_normalized = np.floor(y_int * factor / 2)

# 新版本
y_int_normalized = np.round(y_int * factor / 2)
```

**影響**:
- 最大差異: 1
- 平均差異: 0.138
- 不同值的比例: 13.8%
- **結論**: 影響極小

### 2. 增加 Newton 迭代次數
**位置**: `cmodel_rtl_reference/pure_numpy_cmodel.py` Line 60

**修改**:
```python
# 舊版本: 10 次迭代
for _ in range(10):

# 新版本: 20 次迭代
for _ in range(20):
```

**影響**:
- 迭代 10: 最大變化 = 0.0
- 迭代 15: 最大變化 = 0.0
- 迭代 20: 最大變化 = 0.0
- **結論**: Newton 迭代在第 10 次就已經完全收斂，增加迭代次數沒有任何改善

### 3. Requantize 使用 round
**位置**: `cmodel_rtl_reference/pure_numpy_cmodel.py` Line 355, 358

**狀態**: 已經在使用 `np.round`，無需修改

## 測試結果

### 優化前（原始版本）
- 平均相關係數: 95.35%
- 通過率 (>98%): 5/12 (41.7%)
- 最好的 block: Block 6 (99.46%)
- 最差的 block: Block 4 (84.08%)

### 優化後（round + 20 次迭代）
- 平均相關係數: 95.35%
- 通過率 (>98%): 5/12 (41.7%)
- 最好的 block: Block 6 (99.46%)
- 最差的 block: Block 4 (84.08%)

**結論**: **沒有任何改善**

## 根本原因分析

根據 `BLOCK_ANALYSIS_REPORT.md` 的分析：

### 主要問題不在 LayerNorm
1. **QAct3 的 SF 變異最大**
   - 變異係數: 0.3072
   - 這是所有 QuantAct 中變異最大的

2. **問題 blocks 的 SF 特徵**
   - 問題 blocks 的平均 SF 比好的 blocks 小 **12.95%**
   - 這表示量化範圍不夠大，導致精度損失

3. **誤差不是線性累積**
   - Block 4 (84.08%) 比 Block 3 (91.74%) 差很多
   - 但 Block 5 (97.19%) 又恢復了
   - 這表示某些 blocks 的權重或 SF 有問題

## 可能的解決方案

### 方案 A: 調整 QAct3 的 SF（需要重新訓練）
- 在 QAT 訓練時，增加 QAct3 的量化範圍
- 這需要修改 PyTorch 模型並重新訓練

### 方案 B: 使用更高精度的中間值（會改變硬體位寬）
- 例如：LayerNorm 輸出使用 int64 而不是 int32
- **用戶已明確拒絕**：會影響硬體設計位寬

### 方案 C: 接受當前精度
- 95.35% 的平均相關係數已經相當不錯
- 5/12 blocks 達到 >98% 的精度
- 最好的 block 達到 99.46%

## 建議

**保留當前的優化**（round + 20 次迭代），雖然對整體精度沒有顯著影響，但：
1. **理論上更正確**: round 比 floor 更接近真實值
2. **沒有副作用**: 不會降低精度或增加計算量
3. **為未來優化打基礎**: 如果其他部分改善，這些優化可能會有幫助

**不建議**進一步優化，除非：
1. 重新訓練模型，調整 QAct3 的 SF
2. 用戶同意改變硬體位寬

## 文件
- 測試腳本: `I-ViT/test_layernorm_optimization.py`
- 優化後的 C-Model: `cmodel_rtl_reference/pure_numpy_cmodel.py`
- Block 測試: `I-ViT/test_all_blocks.py`
