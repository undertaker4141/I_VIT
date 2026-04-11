# TVM 參數抽取 Walkthrough

> **日期**: 2026-03-03  
> **目的**: 從 TVM 全整數推理中抽取所有中間層 golden patterns，供 C-model 驗證使用

---

## 背景

原先使用 `extract_patterns.py` 從 PyTorch QAT 模型抽取的量化整數值，是 **float32 模擬** 的結果（`round_ste`, `floor_ste`），並非真正的整數運算。C-model 基於論文架構的全整數運算圖設計，其 `requantize`、`IntLayerNorm`、`IntGELU`、`IntSoftmax` 等運算使用整數除法和右移，與 PyTorch 的浮點模擬有根本差異。

TVM benchmark 程式碼（`I-ViT/TVM_benchmark/`）實作了真正的整數計算圖（使用 `relay.qnn.op.requantize`），因此從 TVM 抽取的中間整數值才是 C-model 應對齊的正確 golden reference。

---

## 環境安裝

### 系統依賴（已安裝）

```bash
sudo apt install -y cmake llvm-15 llvm-15-dev zlib1g-dev libxml2-dev libzstd-dev libpolly-15-dev
```

### TVM 版本

| 組件 | 版本 | 備註 |
|------|------|------|
| TVM (from source) | v0.24.dev0 | 已移除 relay，僅供備用 |
| TVM (pip in venv) | **v0.14.dev273** | ✅ 實際使用，有 relay 支援 |
| Python | 3.10.18 | I_VIT venv |
| NumPy | 1.26.4 | 降級以相容 TVM v0.14 |

### 環境設定

```bash
cd /home/undertaker4141/I_VIT
source .venv/bin/activate
unset PYTHONPATH TVM_HOME TVM_LIBRARY_PATH
```

> [!IMPORTANT]
> **不需要** 設 `TVM_HOME` / `PYTHONPATH`，直接用 venv 中的 TVM v0.14 即可。

---

## 抽取方式

### 腳本位置

```
I-ViT/TVM_benchmark/extract_tvm_patterns.py
```

### 執行命令

```bash
cd I-ViT/TVM_benchmark
python extract_tvm_patterns.py \
    --checkpoint ../checkpoints/qat_calibrated.pth \
    --test-image ../../test_data/test_image.JPEG \
    --output ../../patterns_tvm
```

### 抽取原理

1. 用 `convert_model.load_qconfig()` 從 checkpoint 載入 scaling factors
2. 用 `convert_model.save_params()` 轉換權重為 TVM 格式
3. 在 relay 圖中重建完整的 `Q_Block`（包含所有中間節點）
4. 為每個中間層分別建立 relay sub-model，用 `graph_executor` 在 CPU (`llvm` target) 上跑 inference
5. 儲存每層的整數輸出為 `.npy` 和 `.txt` (hex)

---

## 抽取結果

| 類別 | 數量 | 位置 |
|------|------|------|
| Golden outputs | 172 層 (320 .npy files) | `patterns_tvm/golden/` |
| Weight tensors | 127 | `patterns_tvm/weights/` |
| Scaling factors | 447 | `patterns_tvm/scales/` |
| Predicted class | 377 | — |

### 每個 Block 的層結構

| 層名 | Dtype | Shape | 說明 |
|------|-------|-------|------|
| `norm1` | int32 | (1,197,192) | LayerNorm 整數輸出 |
| `req_norm1_to_qkv` | int8 | (1,197,192) | requantize → QKV 輸入 |
| `qkv` | int32 | (1,197,576) | QKV Dense 輸出 |
| `req_qkv_to_matmul1` | int8 | (1,197,576) | requantize → MatMul 輸入 |
| `attn_matmul1` | int32 | (1,3,197,197) | Q@K^T |
| `softmax` | int8 | (1,3,197,197) | IntSoftmax 整數輸出 |
| `attn_matmul2` | int32 | (1,197,192) | Softmax@V |
| `proj` | int32 | (1,197,192) | Projection Dense 輸出 |
| `add1` | int16 | (1,197,192) | 殘差連接 |
| `norm2` | int32 | (1,197,192) | LayerNorm2 |
| `fc1` | int32 | (1,197,768) | MLP FC1 |
| `gelu` | int32 | (1,197,768) | IntGELU |
| `fc2` | int32 | (1,197,192) | MLP FC2 |
| `add2` | int16 | (1,197,192) | 殘差連接 |

---

## 輸出目錄結構

```
patterns_tvm/
├── config.json              # 抽取元資料
├── input/                   # 輸入影像 + 量化輸入
├── weights/                 # INT8 weights + INT32 biases
├── scales/                  # Scaling factors (float + M/S) + all_scales.json
├── golden/                  # 每層 TVM 整數輸出 + layer_info.json
└── tvm_params/              # TVM 格式 params.npy
```
