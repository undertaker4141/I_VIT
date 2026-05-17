# 遠端 GPU 訓練環境設置指南

> **目標**: 透過 VS Code Tunnel 連接到 RTX 4070 電腦進行訓練  
> **分支**: gpu-training-package  
> **預計時間**: 10-15 分鐘

---

## 📋 前置檢查

### 1. 確認 GPU 可用
```bash
nvidia-smi
```

**預期輸出**: 應該看到 RTX 4070 的資訊

### 2. 確認作業系統
```bash
uname -a  # Linux
# 或
ver       # Windows
```

---

## 🚀 完整安裝步驟

### Step 1: 安裝 uv (Python 套件管理器)

**Linux/macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**驗證安裝**:
```bash
uv --version
```

---

### Step 2: Clone 專案

```bash
# 進入工作目錄
cd ~
# 或選擇其他目錄，例如:
# cd /home/username/projects

# Clone 專案並切換到 gpu-training-package 分支
git clone -b gpu-training-package https://github.com/undertaker4141/I_VIT.git
cd I_VIT
```

**驗證分支**:
```bash
git branch
# 應該顯示: * gpu-training-package
```

---

### Step 3: 創建 Python 虛擬環境 (使用 uv)

```bash
# 使用 uv 創建 Python 3.10 虛擬環境
uv venv --python 3.10 .venv

# 啟動虛擬環境
# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat
```

**驗證環境**:
```bash
python --version
# 應該顯示: Python 3.10.x
```

---

### Step 4: 安裝 PyTorch (GPU 版本)

```bash
# 使用 uv 安裝 PyTorch with CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果是 CUDA 12.1，使用:
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**驗證 CUDA 可用**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**預期輸出**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

---

### Step 5: 安裝其他依賴套件

```bash
# 安裝 timm (PyTorch Image Models)
uv pip install timm

# 安裝其他必要套件
uv pip install numpy pillow tqdm pyyaml

# 如果需要 TensorBoard (可選)
uv pip install tensorboard
```

---

### Step 6: 驗證 ImageNet 數據集路徑

```bash
# 檢查 ImageNet 路徑是否存在
ls ../ImageNet/val

# 如果路徑不同，需要修改 train_gpu.py 中的路徑
```

**如果路徑不同**，編輯 `I-ViT/train_gpu.py`:
```python
# 第 23 行左右
imagenet_path = '/path/to/your/ImageNet'  # 修改為實際路徑
```

---

### Step 7: 測試環境

```bash
cd I-ViT

# 執行環境測試腳本
python test_environment.py
```

**預期輸出**:
```
============================================================
環境測試
============================================================

Test 1: 檢查 PyTorch...
  ✅ PyTorch 版本: 2.x.x

Test 2: 檢查 CUDA...
  ✅ CUDA 可用
  ✅ GPU: NVIDIA GeForce RTX 4070
  ✅ GPU 記憶體: 12.00 GB

Test 3: 檢查其他依賴...
  ✅ torchvision: x.x.x
  ✅ timm: x.x.x
  ✅ numpy: x.x.x
  ✅ Pillow 已安裝

Test 4: 檢查模型...
  ✅ 模型導入成功

Test 5: 檢查預訓練模型...
  ✅ Checkpoint 存在: checkpoints/qat_calibrated.pth
  ✅ 檔案大小: xx.xx MB

Test 6: 檢查 ImageNet 數據集...
  ✅ ImageNet 路徑存在: ../ImageNet/val
  ✅ 類別數量: 1000

Test 7: GPU 記憶體測試...
  ✅ GPU 計算測試通過

============================================================
✅ 所有測試通過！環境設置正確
============================================================

下一步:
  python train_gpu.py
```

---

## 🎯 開始訓練

### 方案 A: 快速訓練 (推薦)

```bash
cd I-ViT
python train_gpu.py
```

**配置**:
- Batch size: 128
- Epochs: 1
- 預計時間: 8-10 分鐘

---

### 方案 B: 最大化 RTX 4070 性能

編輯 `train_gpu.py`:
```python
# 第 24 行
batch_size = 256  # 從 128 改為 256

# 第 26 行 (可選，更好的準確率)
epochs = 3  # 從 1 改為 3
```

然後執行:
```bash
python train_gpu.py
```

**配置**:
- Batch size: 256
- Epochs: 1 或 3
- 預計時間: 6-8 分鐘 (1 epoch) 或 18-24 分鐘 (3 epochs)

---

## 📊 訓練完成後

### 1. 檢查訓練結果

```bash
# 查看訓練結果
cat I-ViT/output_gpu/training_results.txt
```

### 2. 測試模型準確率

```bash
cd I-ViT/TVM_benchmark

# 轉換模型到 TVM 格式
python convert_model.py \
    --model-path ../output_gpu/checkpoint.pth \
    --params-path . \
    --depth 12

# 生成 calibrated scales
python generate_calibrated_scales.py \
    --model-path '../output_gpu/checkpoint.pth' \
    --output calibrated_scales_gpu.npy \
    --image '../../data/test_image.JPEG'

# 測試 100 張圖片
python test_100_images_gpu.py
```

### 3. 預期結果

```
PyTorch 結果:
  Top-1 準確率: 90-95%
  Top-5 準確率: 98-99%

TVM 結果:
  Top-1 準確率: 60-70%  ✅ (目標 >60%)
  Top-5 準確率: 85-90%
```

---

## 🔧 故障排除

### 問題 1: CUDA out of memory

**解決方案**:
```python
# 編輯 train_gpu.py，減少 batch size
batch_size = 64  # 或 32
```

### 問題 2: ImageNet 路徑錯誤

**解決方案**:
```bash
# 找到 ImageNet 位置
find ~ -name "ImageNet" -type d 2>/dev/null

# 或
locate ImageNet

# 然後修改 train_gpu.py 中的路徑
```

### 問題 3: 訓練速度慢

**檢查**:
```bash
# 訓練時在另一個終端執行
nvidia-smi -l 1

# GPU 使用率應該接近 100%
```

**如果 GPU 使用率低**:
```python
# 編輯 train_gpu.py
num_workers = 8  # 增加 workers
pin_memory = True  # 確保啟用
```

### 問題 4: 模組找不到

**解決方案**:
```bash
# 確認在正確的目錄
pwd
# 應該在 I_VIT/I-ViT 目錄

# 確認虛擬環境已啟動
which python
# 應該指向 .venv/bin/python
```

---

## 📦 訓練完成後的檔案

訓練成功後，會生成以下檔案：

```
I-ViT/
├── output_gpu/
│   ├── checkpoint.pth              # 訓練後的模型
│   ├── training_results.txt        # 訓練結果
│   └── test_results.txt            # 測試結果 (執行 test_100_images_gpu.py 後)
│
└── TVM_benchmark/
    ├── params.npy                  # TVM 模型參數
    └── calibrated_scales_gpu.npy   # Calibrated scales
```

---

## 🎓 重要提醒

### 訓練時注意事項

1. **不要關閉終端或 VS Code**
   - 訓練需要 8-10 分鐘持續運行
   - 如果連線中斷，訓練會停止

2. **監控 GPU 溫度**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - 溫度應該在 60-80°C 之間

3. **確保有足夠的磁碟空間**
   ```bash
   df -h
   ```
   - 至少需要 5GB 可用空間

### 訓練後

1. **備份重要檔案**
   ```bash
   # 壓縮訓練結果
   tar -czf training_results.tar.gz I-ViT/output_gpu I-ViT/TVM_benchmark/params.npy I-ViT/TVM_benchmark/calibrated_scales_gpu.npy
   ```

2. **傳回本地**
   - 使用 VS Code 的檔案傳輸功能
   - 或使用 `scp` / `rsync`

---

## 📞 需要幫助？

如果遇到問題：

1. **檢查日誌**
   ```bash
   # 查看最近的錯誤
   tail -n 50 I-ViT/output_gpu/training_results.txt
   ```

2. **檢查 GPU 狀態**
   ```bash
   nvidia-smi
   ```

3. **檢查 Python 環境**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

---

## ✅ 快速檢查清單

訓練前確認：
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] CUDA 可用 (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] ImageNet 路徑正確
- [ ] 虛擬環境已啟動
- [ ] 磁碟空間充足 (>5GB)

訓練後確認：
- [ ] `output_gpu/checkpoint.pth` 存在
- [ ] PyTorch 準確率 > 90%
- [ ] TVM 準確率 > 60%
- [ ] 檔案已備份

---

**祝訓練順利！** 🚀
