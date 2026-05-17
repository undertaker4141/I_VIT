# 🚀 快速開始指南

> **一鍵設置 GPU 訓練環境**

---

## 📋 方法 1: 自動化腳本（推薦）

### Linux/macOS/WSL:
```bash
# 下載並執行設置腳本
curl -O https://raw.githubusercontent.com/undertaker4141/I_VIT/gpu-training-package/setup_gpu_training.sh
chmod +x setup_gpu_training.sh
./setup_gpu_training.sh
```

### Windows PowerShell:
```powershell
# 下載並執行設置腳本
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/undertaker4141/I_VIT/gpu-training-package/setup_gpu_training.ps1" -OutFile "setup_gpu_training.ps1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_gpu_training.ps1
```

---

## 📋 方法 2: 手動設置（5 個指令）

```bash
# 1. 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone 專案
git clone -b gpu-training-package https://github.com/undertaker4141/I_VIT.git
cd I_VIT

# 3. 創建環境並安裝套件
uv venv --python 3.10 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 4. 安裝 PyTorch 和依賴
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install timm numpy pillow tqdm pyyaml

# 5. 開始訓練
cd I-ViT
python train_gpu.py
```

---

## ⚡ 訓練指令

### 標準訓練（8-10 分鐘）
```bash
cd I-ViT
python train_gpu.py
```

### 快速訓練（batch_size=256, 6-8 分鐘）
```bash
# 編輯 train_gpu.py，將 batch_size 改為 256
python train_gpu.py
```

### 高準確率訓練（3 epochs, 24-30 分鐘）
```bash
# 編輯 train_gpu.py，將 epochs 改為 3
python train_gpu.py
```

---

## 🧪 測試指令

```bash
cd I-ViT/TVM_benchmark

# 1. 轉換模型
python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12

# 2. 生成 scales
python generate_calibrated_scales.py \
    --model-path '../output_gpu/checkpoint.pth' \
    --output calibrated_scales_gpu.npy \
    --image '../../data/test_image.JPEG'

# 3. 測試準確率
python test_100_images_gpu.py
```

---

## 📊 預期結果

| 項目 | 目標 |
|------|------|
| **訓練時間** | 8-10 分鐘 (RTX 4070, 1 epoch) |
| **PyTorch Top-1** | 90-95% |
| **TVM Top-1** | 60-70% ✅ |
| **GPU 使用率** | >90% |
| **VRAM 使用** | ~6GB (batch_size=128) |

---

## 🔧 常見問題

### Q: CUDA out of memory
```python
# 編輯 train_gpu.py
batch_size = 64  # 減少 batch size
```

### Q: ImageNet 路徑錯誤
```python
# 編輯 train_gpu.py (第 23 行)
imagenet_path = '/your/path/to/ImageNet'
```

### Q: 訓練速度慢
```bash
# 檢查 GPU 使用率
nvidia-smi -l 1
# 應該接近 100%
```

---

## 📞 需要詳細說明？

查看完整文檔：
- **詳細設置指南**: `REMOTE_GPU_SETUP.md`
- **訓練指南**: `GPU_TRAINING_README.md`
- **RTX 4070 優化**: `RTX_4070_TRAINING_GUIDE.md`
- **修改總結**: `MODIFICATION_SUMMARY.md`

---

**祝訓練順利！** 🎉
