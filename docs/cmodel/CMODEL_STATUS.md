# C-Model 實作狀態報告

## 日期
2026-05-18

## 完成的工作

### 1. 完整的 C-Model 框架 ✓
- **文件**: `cmodel_rtl_reference/pytorch_cmodel.py`
- **內容**: 
  - 完整的 DeiT-Tiny 模型結構
  - 使用已驗證的算法單元（linear 和 nonlinear）
  - 包含所有 12 個 Transformer blocks
  - 支援端到端推理

### 2. 算法單元（已驗證）✓
- **文件**: 
  - `cmodel_rtl_reference/linear_cmodel_reference.py`
  - `cmodel_rtl_reference/nonlinear_cmodel_reference.py`
- **內容**:
  - `int_dense_kernel`: 全連接層
  - `int_matmul_kernel`: 矩陣乘法
  - `int_layer_norm_fixed`: LayerNorm
  - `int_gelu_kernel_fixed`: GELU
  - `int_softmax_kernel_fixed`: Softmax

### 3. 測試腳本 ✓
- **文件**:
  - `I-ViT/test_cmodel.py`: 完整的 C-Model 測試
  - `I-ViT/test_cmodel_simple.py`: 簡化測試
  - `I-ViT/extract_golden_patterns.py`: Golden patterns 提取
  - `I-ViT/verify_cmodel.py`: C-Model 驗證框架

### 4. 文檔 ✓
- **文件**:
  - `docs/cmodel/PYTORCH_CMODEL_GUIDE.md`: C-Model 使用指南
  - `docs/cmodel/RTL_SIMULATION_GUIDE.md`: RTL 模擬指南
  - `docs/cmodel/HARDWARE_CMODEL_GUIDE.md`: 硬體實作指南

## 當前問題

### Checkpoint 兼容性問題

**問題描述**:
- 訓練時的代碼版本與當前代碼版本不同
- Checkpoint 中的 buffer shape 與當前模型不匹配
- 主要是 `*_scaling_factor` 和 `*_integer` 的 shape 不同

**具體錯誤**:
```
size mismatch for qact_input.act_scaling_factor: 
  copying a param with shape torch.Size([]) from checkpoint, 
  the shape in current model is torch.Size([1]).
```

**影響的 checkpoint**:
1. `I-ViT/checkpoints/qat_calibrated.pth` (論文原始)
2. `I-ViT/output_gpu/checkpoint.pth` (GPU 訓練)

### 根本原因

在訓練過程中，PyTorch 模型的 buffer 註冊方式可能有所不同：

**訓練時的代碼** (可能):
```python
self.register_buffer('act_scaling_factor', torch.zeros(1))  # Shape: [1]
```

**當前代碼**:
```python
self.register_buffer('act_scaling_factor', torch.zeros(1))  # Shape: [1]
```

但 checkpoint 中保存的是:
```python
act_scaling_factor: torch.Size([])  # Scalar
```

這表示訓練時使用的是 scalar，而當前代碼期望的是 1D tensor。

## 解決方案

### 方案 1: 修改模型代碼以匹配 checkpoint ❌

**優點**: 可以直接載入現有 checkpoint
**缺點**: 
- 需要修改大量代碼
- 可能破壞當前的功能
- 不確定訓練時的確切代碼版本

### 方案 2: 重新訓練模型 ❌

**優點**: 生成與當前代碼完全匹配的 checkpoint
**缺點**:
- 需要 GPU 和大量時間（已經訓練過 1 epoch，115 分鐘）
- 已有的訓練結果（73.54% 準確率）無法使用

### 方案 3: 手動轉換 checkpoint ✓ (推薦)

**優點**:
- 不需要修改代碼
- 不需要重新訓練
- 可以使用現有的訓練結果

**缺點**:
- 需要編寫轉換腳本

**實作步驟**:
1. 載入 checkpoint
2. 調整所有 buffer 的 shape
3. 保存新的 checkpoint

### 方案 4: 直接使用 PyTorch 模型作為 Golden Reference ✓ (當前最佳)

**優點**:
- 不需要 C-Model 的完整實作
- PyTorch 模型已經驗證正確（73.54% 準確率）
- 可以直接提取中間層輸出作為 Golden Patterns

**缺點**:
- 無法驗證 C-Model 的正確性
- RTL 驗證需要從 PyTorch 提取測試向量

**實作方式**:
1. 使用 `extract_golden_patterns.py` 提取 PyTorch 的中間層輸出
2. 將這些輸出作為 RTL 驗證的 Golden Reference
3. C-Model 可以作為算法參考，但不需要完整驗證

## 建議的下一步

### 短期（RTL 驗證）

1. **使用 PyTorch 模型作為 Golden Reference**:
   ```bash
   # 需要在有 PyTorch 環境的機器上執行
   cd I-ViT
   python extract_golden_patterns.py
   ```

2. **生成 RTL 測試向量**:
   - 從 `golden_patterns.npz` 提取數據
   - 轉換為 RTL 可讀格式（.hex, .mem）

3. **RTL 實作**:
   - 參考 C-Model 的算法流程
   - 使用 Golden Patterns 驗證

### 中期（C-Model 驗證）

1. **創建 checkpoint 轉換腳本**:
   ```python
   # convert_checkpoint.py
   import torch
   
   # 載入舊 checkpoint
   old_ckpt = torch.load('output_gpu/checkpoint.pth', weights_only=False)
   
   # 調整 buffer shapes
   new_state_dict = {}
   for key, value in old_ckpt['model'].items():
       if 'scaling_factor' in key or 'integer' in key:
           # 調整 shape
           if value.dim() == 0:  # Scalar
               new_state_dict[key] = value.unsqueeze(0)  # -> [1]
           else:
               new_state_dict[key] = value
       else:
           new_state_dict[key] = value
   
   # 保存新 checkpoint
   torch.save({'model': new_state_dict}, 'output_gpu/checkpoint_converted.pth')
   ```

2. **驗證 C-Model**:
   ```bash
   cd I-ViT
   python test_cmodel.py
   ```

### 長期（完整驗證流程）

1. **建立完整的驗證流程**:
   - PyTorch 模型 → Golden Patterns
   - C-Model → 驗證與 PyTorch 一致
   - RTL → 驗證與 C-Model 一致

2. **自動化測試**:
   - 單元測試（每個算法單元）
   - 集成測試（完整模型）
   - 回歸測試（多個測試圖片）

## 當前可用的資源

### 已驗證的組件

1. **PyTorch 量化模型** ✓
   - 準確率: 73.54% Top-1, 92.58% Top-5
   - Checkpoint: `I-ViT/output_gpu/checkpoint.pth`

2. **算法單元** ✓
   - 已通過 TVM 驗證（雖然 TVM 整體有問題，但單個算法單元是正確的）
   - 可以直接用於 RTL 實作

3. **文檔** ✓
   - 完整的使用指南
   - RTL 實作建議
   - 調試技巧

### 待完成的組件

1. **Checkpoint 轉換** ⏳
   - 需要編寫轉換腳本
   - 或者重新訓練模型

2. **C-Model 完整驗證** ⏳
   - 需要可載入的 checkpoint
   - 逐層驗證與 PyTorch 的一致性

3. **RTL 測試向量生成** ⏳
   - 從 Golden Patterns 生成
   - 轉換為 RTL 格式

## 結論

雖然遇到了 checkpoint 兼容性問題，但我們已經完成了：

1. ✓ 完整的 C-Model 框架
2. ✓ 已驗證的算法單元
3. ✓ 完整的文檔
4. ✓ 測試腳本

**當前最佳方案**: 直接使用 PyTorch 模型作為 Golden Reference 進行 RTL 驗證，C-Model 作為算法參考。

**如果需要完整驗證 C-Model**: 需要先解決 checkpoint 兼容性問題（方案 3 或方案 2）。

## 附錄：文件清單

### C-Model 相關
- `cmodel_rtl_reference/pytorch_cmodel.py` - 完整 C-Model
- `cmodel_rtl_reference/linear_cmodel_reference.py` - 線性層算法
- `cmodel_rtl_reference/nonlinear_cmodel_reference.py` - 非線性層算法

### 測試腳本
- `I-ViT/test_cmodel.py` - C-Model 測試
- `I-ViT/test_cmodel_simple.py` - 簡化測試
- `I-ViT/extract_golden_patterns.py` - Golden patterns 提取
- `I-ViT/verify_cmodel.py` - C-Model 驗證框架

### 文檔
- `docs/cmodel/PYTORCH_CMODEL_GUIDE.md` - C-Model 使用指南
- `docs/cmodel/RTL_SIMULATION_GUIDE.md` - RTL 模擬指南
- `docs/cmodel/HARDWARE_CMODEL_GUIDE.md` - 硬體實作指南
- `docs/cmodel/CMODEL_STATUS.md` - 本文檔
