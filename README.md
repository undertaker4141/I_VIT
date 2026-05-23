# I-ViT Pattern Extraction for Hardware Accelerator

本專案從量化的 I-ViT (DeiT-Tiny) 模型中抽取 INT8 權重、INT32 Bias、Scaling Factors 和每層的 Golden Outputs，供硬體加速器 C-model 開發使用。

## 快速開始

```bash
# 1. 安裝依賴
uv sync

# 2. 抽取 Patterns (使用已訓練的 checkpoint)
cd I-ViT
python extract_patterns.py \
    --checkpoint checkpoints/qat_calibrated.pth \
    --test-image ../test_data/test_image.JPEG \
    --output ../patterns

# 3. 驗證輸出
python verify_patterns.py --patterns ../patterns
```

## 專案結構

```
I_VIT/
├── I-ViT/                          # I-ViT 核心程式碼
│   ├── checkpoints/
│   │   └── qat_calibrated.pth      # QAT 校準後的 checkpoint
│   ├── quick_qat.py                # 快速 QAT 腳本
│   ├── extract_patterns.py         # Pattern 抽取腳本
│   └── verify_patterns.py          # 驗證腳本
├── patterns/                        # 輸出的硬體 Pattern (285MB)
├── test_data/                       # 固定的測試圖片
├── docs/                            # 文件
│   ├── implementation_plan.md      # 實作計畫
│   ├── task.md                     # 任務清單
│   └── walkthrough.md              # 實作紀錄
└── HARDWARE_CMODEL_GUIDE.md        # ⭐ C-model 完整開發指南
```

## 輸出 Pattern 說明

| 目錄 | 內容 | 格式 |
|------|------|------|
| `patterns/weights/` | INT8 權重 + INT32 Bias | `.npy` + `.txt` (Hex) |
| `patterns/scales/` | M (Mantissa) + S (Shift) | `.npy` |
| `patterns/golden/` | 每層 Golden Output | Float + Int |
| `patterns/embeddings/` | cls_token + pos_embed | INT8 量化版 |

## 量化規格

- **Weight**: INT8 (signed, [-127, 127])
- **Bias**: INT32 (signed)
- **Scale**: M × 2^(-S) 格式，M 已乘以 2^31
- **Shift 方向**: 統一右移

## 文件

- 📖 **[HARDWARE_CMODEL_GUIDE.md](HARDWARE_CMODEL_GUIDE.md)** - 完整的 C-model 開發指南
- 📋 **[docs/implementation_plan.md](docs/implementation_plan.md)** - 實作計畫
- ✅ **[docs/task.md](docs/task.md)** - 任務清單
- 📝 **[docs/walkthrough.md](docs/walkthrough.md)** - 實作紀錄

## 驗證結果

### 模型準確率
```
QAT Accuracy: 73.35% (Top-1), 91.80% (Top-5)
Pattern Files: 1829
M/S Reconstruction Error: 0.000000%
```

> ⚠️ **注意**: 上述準確率高於論文 (72.24%) 是因為我們使用 ImageNet Validation Set 同時作為訓練和驗證集 (Data Leakage)。這不影響 Pattern 抽取的正確性，但不代表真實泛化能力。如需真實準確率，請使用完整的 ImageNet Train Set (138GB) 進行訓練。

### RTL 驗證結果 ✅

所有非線性模組的 RTL 實現已通過完整驗證（2026/5/23）：

| 模組 | 測試案例 | 總資料量 | 錯誤數 | 狀態 |
|------|---------|---------|--------|------|
| LayerNorm | 25 | 945,600 | 0 | ✅ 通過 (Bit-exact) |
| GELU | 12 | 1,815,552 | 0 | ✅ 通過 (Bit-exact) |
| Softmax | 12 | 1,397,124 | 0 | ✅ 通過 (Bit-exact) |
| **總計** | **49** | **4,158,276** | **0** | ✅ **全部通過** |

詳細報告：[NONLINEAR_VERIFICATION_COMPLETE.md](NONLINEAR_VERIFICATION_COMPLETE.md)

## License

Based on [I-ViT](https://github.com/zkkli/I-ViT) project.
