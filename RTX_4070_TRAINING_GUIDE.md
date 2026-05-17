# RTX 4070 訓練指南

## 🎮 硬體規格

**RTX 4070**:
- CUDA Cores: 5888
- Tensor Cores: 184 (第 4 代)
- VRAM: 12 GB GDDR6X
- Memory Bandwidth: 504 GB/s
- TDP: 200W

**非常適合訓練 DeiT-Tiny！** ✅

## ⚡ 性能預估

### 訓練速度

| 配置 | 預估時間 | 說明 |
|------|---------|------|
| **1 epoch, 50K 圖片, batch_size=128** | **8-10 分鐘** | 推薦配置 ⭐ |
| 1 epoch, 50K 圖片, batch_size=256 | 6-8 分鐘 | 更快，但可能 OOM |
| 3 epochs, 50K 圖片, batch_size=128 | 24-30 分鐘 | 更好的準確率 |
| 5 epochs, 50K 圖片, batch_size=128 | 40-50 分鐘 | 最佳準確率 |

### 記憶體使用

| Batch Size | VRAM 使用 | 狀態 |
|-----------|----------|------|
| 64 | ~4 GB | ✅ 安全 |
| 128 | ~6 GB | ✅ 推薦 |
| 256 | ~10 GB | ✅ 可行 |
| 512 | ~18 GB | ❌ 會 OOM |

**建議**: 使用 **batch_size=128** 或 **256**

## 🚀 優化配置

### 方案 A：快速驗證（推薦）⭐

```python
# 在 train_gpu.py 中設置
batch_size = 128
epochs = 1
lr = 5e-7
num_workers = 8  # RTX 4070 可以用更多 workers
```

**預期**:
- 訓練時間: **8-10 分鐘**
- PyTorch 準確率: 90-95%
- TVM 準確率: 60-70%

### 方案 B：更好的準確率

```python
batch_size = 128
epochs = 3
lr = 5e-7
num_workers = 8
```

**預期**:
- 訓練時間: **24-30 分鐘**
- PyTorch 準確率: 92-96%
- TVM 準確率: 65-75%

### 方案 C：最大化 batch size

```python
batch_size = 256  # 利用 12GB VRAM
epochs = 1
lr = 5e-7
num_workers = 8
```

**預期**:
- 訓練時間: **6-8 分鐘**（更快！）
- 準確率: 與方案 A 相同

## 🔧 train_gpu.py 優化設置

修改 `train_gpu.py` 以充分利用 RTX 4070：

```python
def main():
    # ... 前面的代碼 ...
    
    # 優化配置
    batch_size = 256  # RTX 4070 可以用更大的 batch size
    lr = 5e-7
    epochs = 1
    
    # 檢查 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 啟用 TF32 加速（RTX 4070 支持）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 加速已啟用")
    
    # DataLoader 優化
    num_workers = 8  # RTX 4070 可以處理更多並行
    pin_memory = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True  # 加速數據加載
    )
```

## 📊 性能對比

### CPU vs RTX 4070

| 項目 | CPU (我們測試的) | RTX 4070 |
|------|-----------------|----------|
| 100 batches | 22 分鐘 | ~30 秒 |
| 1 epoch (50K) | 10-15 小時 | **8-10 分鐘** |
| 3 epochs | 30-45 小時 | **24-30 分鐘** |
| 速度提升 | 1x | **~50-100x** 🚀 |

## 🎯 推薦訓練策略

### 階段 1：快速驗證（必做）

```bash
# 使用 batch_size=128, 1 epoch
python train_gpu.py
```

**時間**: 8-10 分鐘  
**目標**: TVM 準確率 > 60%

### 階段 2：如果需要更高準確率（可選）

```bash
# 修改 train_gpu.py: epochs = 3
python train_gpu.py
```

**時間**: 24-30 分鐘  
**目標**: TVM 準確率 > 70%

### 階段 3：最大化性能（可選）

```bash
# 修改 train_gpu.py: batch_size = 256
python train_gpu.py
```

**時間**: 6-8 分鐘  
**優勢**: 最快的訓練速度

## 💡 RTX 4070 特殊優化

### 1. 啟用 TF32

```python
# 在 train_gpu.py 開頭添加
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**效果**: 提速 20-30%，幾乎無精度損失

### 2. 使用混合精度訓練（可選）

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在訓練循環中
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**效果**: 再提速 30-50%，VRAM 使用減半

### 3. 增加 num_workers

```python
num_workers = 8  # 或更多，取決於 CPU
```

**效果**: 減少數據加載瓶頸

## 🐛 常見問題

### Q1: CUDA out of memory

**解決方案**:
```python
# 減少 batch_size
batch_size = 128  # 或 64
```

### Q2: 訓練速度慢

**檢查**:
```bash
nvidia-smi  # 確認 GPU 使用率接近 100%
```

**如果 GPU 使用率低**:
- 增加 `num_workers`
- 啟用 `pin_memory=True`
- 啟用 `persistent_workers=True`

### Q3: 想要更快

**方案**:
1. 增加 `batch_size` 到 256
2. 啟用 TF32
3. 使用混合精度訓練

## 📈 預期結果

### 1 Epoch 訓練後

**PyTorch**:
- Top-1: 90-95%
- Top-5: 98-99%

**TVM**:
- Top-1: 60-70%
- Top-5: 85-90%

### 3 Epochs 訓練後

**PyTorch**:
- Top-1: 92-96%
- Top-5: 99%

**TVM**:
- Top-1: 65-75%
- Top-5: 88-92%

## ✅ 成功標準

- ✅ 訓練完成（8-10 分鐘）
- ✅ TVM 準確率 > 60%
- ✅ GPU 使用率 > 90%
- ✅ 無 OOM 錯誤

## 🚀 開始訓練

```bash
# 1. 檢查 GPU
nvidia-smi

# 2. 確認 CUDA 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. 開始訓練
cd I-ViT
python train_gpu.py

# 4. 預期輸出
# Device: cuda
# GPU: NVIDIA GeForce RTX 4070
# GPU Memory: 12.00 GB
# ✅ TF32 加速已啟用
# ...
# Epoch 1/1 completed in 480.0s (8 分鐘)
# Train Acc: 93.45%
# Val Acc: 91.23%
```

## 🎉 總結

**RTX 4070 非常適合這個任務！**

- ✅ 12GB VRAM 足夠
- ✅ 訓練速度快（8-10 分鐘）
- ✅ 可以用大 batch size (256)
- ✅ 支持 TF32 加速

**預計總時間**: 不到 15 分鐘就能完成訓練和測試！🚀

---

**祝訓練順利！有 RTX 4070 真好！** 🎮✨
