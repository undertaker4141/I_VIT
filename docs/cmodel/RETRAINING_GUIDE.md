# 重新訓練指南：修復 QAct3 SF 問題

## 問題描述

當前 C-Model 的端到端推論預測錯誤，原因是：
- QAct3 的 scaling factor (SF) 變異太大（變異係數 0.3072）
- 某些 blocks 的 QAct3 SF 太小，導致量化精度損失
- 誤差累積後，最終 logits 相關係數只有 31.4%

## 解決方案

修改 QuantAct 類，為 QAct3 設置最小 SF，然後重新訓練模型。

---

## 步驟 1：修改 QuantAct 類

### 文件位置
`I-ViT/models/quantization_utils/quant_modules.py`

### 修改內容

#### 1.1 修改 `__init__` 方法

**原始代碼**（Line 162-176）：
```python
def __init__(self,
             activation_bit=8,
             act_range_momentum=0.95,
             running_stat=True,
             per_channel=False,
             quant_mode="symmetric"):
    super(QuantAct, self).__init__()

    self.activation_bit = activation_bit
    self.act_range_momentum = act_range_momentum
    self.running_stat = running_stat
    self.quant_mode = quant_mode
    self.per_channel = per_channel

    self.min_val = torch.zeros(1)
    self.max_val = torch.zeros(1)
    self.register_buffer('act_scaling_factor', torch.zeros(1))
```

**修改後**：
```python
def __init__(self,
             activation_bit=8,
             act_range_momentum=0.95,
             running_stat=True,
             per_channel=False,
             quant_mode="symmetric",
             min_scaling_factor=None):  # ← 新增參數
    super(QuantAct, self).__init__()

    self.activation_bit = activation_bit
    self.act_range_momentum = act_range_momentum
    self.running_stat = running_stat
    self.quant_mode = quant_mode
    self.per_channel = per_channel
    self.min_scaling_factor = min_scaling_factor  # ← 新增屬性

    self.min_val = torch.zeros(1)
    self.max_val = torch.zeros(1)
    self.register_buffer('act_scaling_factor', torch.zeros(1))
```

#### 1.2 修改 `forward` 方法

**原始代碼**（Line 221-223）：
```python
self.act_scaling_factor = symmetric_linear_quantization_params(
    self.activation_bit, self.min_val, self.max_val)
```

**修改後**：
```python
self.act_scaling_factor = symmetric_linear_quantization_params(
    self.activation_bit, self.min_val, self.max_val)

# ← 新增：強制最小 SF
if self.min_scaling_factor is not None:
    self.act_scaling_factor = torch.max(
        self.act_scaling_factor,
        torch.tensor(self.min_scaling_factor, device=self.act_scaling_factor.device)
    )
```

---

## 步驟 2：修改 Block 類

### 文件位置
`I-ViT/models/layers_quant.py` 或 `I-ViT/models/vit_quant.py`

### 找到 Block 的 `__init__` 方法

**原始代碼**（推測）：
```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., ...):
        super().__init__()
        ...
        self.qact1 = QuantAct(activation_bit=8)  # Norm1 後
        self.qact2 = QuantAct(activation_bit=8)  # Residual 1 後
        self.qact3 = QuantAct(activation_bit=8)  # Norm2 後
        self.qact4 = QuantAct(activation_bit=8)  # GELU 後
        ...
```

**修改後**：
```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., ...):
        super().__init__()
        ...
        self.qact1 = QuantAct(activation_bit=8)  # Norm1 後
        self.qact2 = QuantAct(activation_bit=8)  # Residual 1 後
        self.qact3 = QuantAct(activation_bit=8, min_scaling_factor=5e-4)  # ← 設置最小 SF
        self.qact4 = QuantAct(activation_bit=8)  # GELU 後
        ...
```

### 如何確定最小 SF 的值？

根據 `BLOCK_ANALYSIS_REPORT.md` 的數據：
- 好的 blocks 的平均 QAct3 SF：4.831e-04
- 問題 blocks 的平均 QAct3 SF：4.206e-04

**建議值**：`5e-4`（略高於好的 blocks 的平均值）

**替代方案**：
- 保守：`4.5e-4`（介於好的和壞的之間）
- 激進：`6e-4`（更高，但可能影響模型精度）

---

## 步驟 3：重新訓練模型

### 3.1 使用現有的訓練腳本

```bash
cd I-ViT
python train_gpu.py \
    --model deit_tiny_patch16_224 \
    --batch-size 128 \
    --epochs 100 \
    --data-path ../ImageNet \
    --output_dir output_gpu_retrained
```

### 3.2 訓練參數建議

| 參數 | 建議值 | 說明 |
|------|--------|------|
| epochs | 100 | 與原始訓練相同 |
| batch-size | 128 | 根據 GPU 記憶體調整 |
| lr | 1e-4 | 學習率 |
| weight-decay | 0.05 | 權重衰減 |

### 3.3 監控訓練

**關鍵指標**：
1. **訓練精度**：應該與原始模型相近（> 70%）
2. **QAct3 SF 分佈**：
   - 檢查所有 blocks 的 QAct3 SF
   - 確保沒有 SF < 5e-4 的 block
   - 變異係數應該 < 0.2

**檢查腳本**：
```python
# 在訓練過程中或訓練後運行
import torch

checkpoint = torch.load('output_gpu_retrained/checkpoint.pth')
model = ...  # 載入模型
model.load_state_dict(checkpoint['model'])

# 提取所有 QAct3 SF
qact3_sfs = []
for i, block in enumerate(model.blocks):
    sf = block.qact3.act_scaling_factor.item()
    qact3_sfs.append(sf)
    print(f"Block {i}: QAct3 SF = {sf:.6e}")

# 統計
import numpy as np
qact3_sfs = np.array(qact3_sfs)
print(f"\n統計:")
print(f"  平均值: {qact3_sfs.mean():.6e}")
print(f"  標準差: {qact3_sfs.std():.6e}")
print(f"  變異係數: {qact3_sfs.std() / qact3_sfs.mean():.4f}")
print(f"  最小值: {qact3_sfs.min():.6e}")
print(f"  最大值: {qact3_sfs.max():.6e}")
```

---

## 步驟 4：驗證新模型

### 4.1 轉換 checkpoint

```bash
cd I-ViT
python convert_checkpoint.py \
    --input output_gpu_retrained/checkpoint.pth \
    --output output_gpu_retrained/checkpoint_converted.pth
```

### 4.2 測試所有 blocks

```bash
# 修改 test_all_blocks.py 中的 checkpoint 路徑
# checkpoint_path = 'output_gpu_retrained/checkpoint_converted.pth'

python test_all_blocks.py
```

**預期結果**：
- 平均相關係數：> 98%（從 95.35% 提升）
- 通過率 (>98%)：> 90%（從 41.7% 提升）
- 所有 blocks 相關係數：> 95%

### 4.3 測試端到端推論

```bash
# 修改 pure_integer_end_to_end.py 中的 checkpoint 路徑
# checkpoint_path = 'output_gpu_retrained/checkpoint_converted.pth'

python pure_integer_end_to_end.py
```

**預期結果**：
- 預測類別：1（正確）
- Logits 相關係數：> 95%（從 31.4% 提升）

---

## 步驟 5：測試 100 張圖片

```bash
python test_100_images_pure_integer.py
```

**預期結果**：
- Top-1 準確率：> 65%
- Top-5 準確率：> 85%

---

## 替代方案：Per-Channel 量化

如果方案 1（最小 SF）效果不佳，可以嘗試 per-channel 量化。

### 修改 QuantAct

```python
class QuantAct(nn.Module):
    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,  # ← 改為 True
                 channel_len=None,   # ← 設置為 192
                 quant_mode="symmetric"):
        super(QuantAct, self).__init__()
        ...
        if per_channel:
            self.min_val = torch.zeros(channel_len)
            self.max_val = torch.zeros(channel_len)
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))
        else:
            self.min_val = torch.zeros(1)
            self.max_val = torch.zeros(1)
            self.register_buffer('act_scaling_factor', torch.zeros(1))
```

### 在 Block 中使用

```python
self.qact3 = QuantAct(activation_bit=8, per_channel=True, channel_len=192)
```

**注意**：
- Per-channel 量化需要修改 C-Model
- 硬體實現更複雜
- 但精度會更高（可能 > 99%）

---

## 時間估計

| 步驟 | 時間 |
|------|------|
| 修改代碼 | 0.5-1 小時 |
| 重新訓練 | 8-24 小時（取決於 GPU） |
| 轉換和驗證 | 1-2 小時 |
| **總計** | **9.5-27 小時** |

---

## 風險評估

### 風險 1：模型精度下降
**可能性**：低
**影響**：中
**緩解措施**：
- 先用較小的 min_sf（4.5e-4）測試
- 監控訓練過程中的驗證精度
- 如果精度下降 > 2%，調整 min_sf

### 風險 2：訓練不收斂
**可能性**：極低
**影響**：高
**緩解措施**：
- 使用與原始訓練相同的超參數
- 從預訓練模型開始 fine-tune

### 風險 3：QAct3 SF 仍然變異大
**可能性**：低
**影響**：中
**緩解措施**：
- 增加 min_sf 到 6e-4
- 或改用 per-channel 量化

---

## 成功標準

✅ **必須達成**：
1. 所有 blocks 相關係數 > 95%
2. 平均相關係數 > 98%
3. 端到端推論預測正確（類別 1）
4. Logits 相關係數 > 90%

✅ **期望達成**：
1. 通過率 (>98%) > 90%
2. 端到端 logits 相關係數 > 95%
3. 100 張圖片 Top-1 準確率 > 65%

---

## 參考資料

- `SOLUTION_ANALYSIS.md`：詳細的解決方案分析
- `BLOCK_ANALYSIS_REPORT.md`：Block 誤差分析
- `CURRENT_STATUS.md`：當前狀態總結
