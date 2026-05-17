# GPU 訓練指南

## 📋 目標

使用 GPU 訓練 DeiT-Tiny 模型，使其適應 integer-only 計算，以便部署到 TVM 和 RTL。

## 🎯 當前狀況

### 已完成
- ✅ 修改代碼使用 integer-only 計算（`I-ViT/models/quantization_utils/quant_modules.py`）
- ✅ CPU 訓練 100 batches 測試成功
- ✅ PyTorch 準確率：95%（100 張圖片測試）
- ✅ TVM 準確率：0%（需要更多訓練）

### 需要做的
- ⏳ 使用 GPU 訓練完整 1 epoch（val 數據集，50K 圖片）
- ⏳ 預期時間：10-15 分鐘（GPU）
- ⏳ 目標：TVM 準確率 > 60%

## 🚀 快速開始

### 1. 環境設置

```bash
# 檢查 GPU
nvidia-smi

# 激活環境
source /home/jin25/tvm_env_310/bin/activate

# 確認 PyTorch 可以使用 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 訓練模型

```bash
cd I-ViT

# 使用 GPU 訓練完整 1 epoch（val 數據集）
python train_gpu.py
```

**配置**：
- 數據集：ImageNet val（50K 圖片）
- Batch size：128（GPU 可以用更大的）
- Learning rate：5e-7
- Epochs：1
- Device：CUDA

**預期時間**：10-15 分鐘

### 3. 測試結果

訓練完成後，自動測試：

```bash
# 測試 100 張圖片
cd TVM_benchmark
python test_100_images_gpu.py
```

**成功標準**：
- PyTorch Top-1 準確率：> 90%
- TVM Top-1 準確率：> 60%

## 📁 重要文件

### 訓練腳本
- `I-ViT/train_gpu.py` - GPU 訓練腳本（主要使用）
- `I-ViT/quant_train.py` - 原始訓練腳本（參考）

### 測試腳本
- `I-ViT/TVM_benchmark/test_100_images_gpu.py` - 測試 100 張圖片
- `I-ViT/TVM_benchmark/convert_model.py` - 轉換模型到 TVM
- `I-ViT/TVM_benchmark/generate_calibrated_scales.py` - 生成 scales

### 模型文件
- `I-ViT/checkpoints/qat_calibrated.pth` - 原始 checkpoint
- `I-ViT/output_gpu/checkpoint.pth` - 訓練後的 checkpoint（會生成）

### 代碼修改
- `I-ViT/models/quantization_utils/quant_modules.py` - Integer-only 實現

## 🔧 訓練參數調整

如果需要調整參數，編輯 `train_gpu.py`：

```python
# 數據集大小
use_full_dataset = True  # True = 50K, False = 可以設置更小

# Batch size（根據 GPU 記憶體調整）
batch_size = 128  # 可以嘗試 256 或更大

# Learning rate
lr = 5e-7  # 可以嘗試 1e-6 或 2e-7

# Epochs
epochs = 1  # 可以增加到 3-5
```

## 📊 預期結果

### 訓練後
- **PyTorch Top-1**：90-95%
- **TVM Top-1**：60-70%
- **訓練時間**：10-15 分鐘（1 epoch）

### 如果 TVM 準確率 > 60%
✅ **成功！** 可以繼續：
1. 轉換到 TVM
2. 提取 golden patterns
3. 驗證 C-Model
4. 部署到 RTL

### 如果 TVM 準確率 < 60%
⚠️ **需要更多訓練**：
1. 增加到 3-5 epochs
2. 或使用完整 train 數據集（1.28M 圖片）

## 🐛 常見問題

### Q1: CUDA out of memory
**解決**：減少 batch size
```python
batch_size = 64  # 或 32
```

### Q2: 訓練太慢
**檢查**：
```bash
nvidia-smi  # 確認 GPU 使用率
```

### Q3: TVM 準確率仍然是 0%
**原因**：訓練不夠
**解決**：增加 epochs 到 3-5

## 📝 訓練完成後

### 1. 保存結果
```bash
# Checkpoint 位置
I-ViT/output_gpu/checkpoint.pth

# TVM params
I-ViT/TVM_benchmark/params.npy

# Calibrated scales
I-ViT/TVM_benchmark/calibrated_scales_gpu.npy
```

### 2. 測試報告
訓練完成後會生成：
- `training_results.txt` - 訓練結果
- `test_results.txt` - 測試結果

### 3. 打包給其他組員
```bash
# 只需要這些文件
I-ViT/output_gpu/checkpoint.pth
I-ViT/TVM_benchmark/params.npy
I-ViT/TVM_benchmark/calibrated_scales_gpu.npy
```

## 🎯 成功標準

- ✅ 訓練完成（無錯誤）
- ✅ PyTorch 準確率 > 90%
- ✅ TVM 準確率 > 60%
- ✅ 兩者預測一致性 > 50%

達到這些標準後，就可以繼續 RTL 部署了！

## 📞 聯繫

如果遇到問題，檢查：
1. GPU 是否可用（`nvidia-smi`）
2. CUDA 版本是否匹配
3. 數據集路徑是否正確

---

**祝訓練順利！** 🚀
