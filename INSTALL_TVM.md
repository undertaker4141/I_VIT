# TVM 安裝指南（用於 TVM_benchmark）

訓練完成後，需要使用 TVM 進行推理測試。以下是安裝步驟：

## 方法 1: 使用 pip 安裝 TVM v0.14（推薦）

```bash
# 在 I-ViT 目錄下，確保使用 .venv 環境
source .venv/bin/activate

# 安裝 TVM v0.14（有 relay 支援）
pip install apache-tvm==0.14.0

# 降級 NumPy 以相容 TVM v0.14
pip install numpy==1.26.4

# 驗證安裝
python -c "from tvm import relay; print('TVM installed successfully!')"
```

## 方法 2: 使用 uv 安裝（更快）

```bash
cd ~/khchung/I_VIT/I-ViT

# 安裝 TVM
uv pip install apache-tvm==0.14.0

# 降級 NumPy
uv pip install numpy==1.26.4

# 驗證安裝
uv run python -c "from tvm import relay; print('TVM installed successfully!')"
```

## 安裝後執行 TVM 推理

```bash
cd I-ViT/TVM_benchmark

# Step 1: 轉換模型
uv run python convert_model.py --model-path ../output_gpu/checkpoint.pth --params-path . --depth 12

# Step 2: 生成 calibrated scales
uv run python generate_calibrated_scales.py --model-path '../output_gpu/checkpoint.pth' --output calibrated_scales_gpu.npy --image '../../data/test_image.JPEG'

# Step 3: 測試 100 張圖片
uv run python test_100_images_gpu.py
```

## 如果安裝失敗

如果 `apache-tvm==0.14.0` 無法安裝，可以嘗試：

```bash
# 安裝最新的 TVM（可能沒有 relay）
pip install apache-tvm

# 或從 conda 安裝
conda install -c conda-forge tvm
```

## 注意事項

- TVM v0.14 是最後一個包含 `relay` 模組的版本
- 較新版本的 TVM 已經移除 `relay`，會導致 `convert_model.py` 失敗
- NumPy 需要降級到 1.26.4 以避免相容性問題
