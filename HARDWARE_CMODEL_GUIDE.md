# I-ViT Pattern Extraction - 硬體加速器 C-model 開發文件

> **建立日期**: 2026-01-08  
> **模型**: DeiT-Tiny (5.72M parameters)  
> **用途**: 提供硬體加速器 C-model 開發所需的量化參數與 Golden Reference

---

## 目錄

1. [專案架構](#專案架構)
2. [量化規格總覽](#量化規格總覽)
3. [Pattern 輸出結構](#pattern-輸出結構)
4. [C-model 對照指南](#c-model-對照指南)
5. [腳本使用說明](#腳本使用說明)
6. [DeiT-Tiny 模型結構](#deit-tiny-模型結構)
7. [關鍵實作細節](#關鍵實作細節)
8. [驗證結果](#驗證結果)

---

## 專案架構

```
I_VIT/
├── I-ViT/                          # I-ViT 核心程式碼
│   ├── models/
│   │   ├── vit_quant.py            # ViT 量化模型定義
│   │   ├── layers_quant.py         # 量化層 (PatchEmbed, Mlp)
│   │   └── quantization_utils/
│   │       ├── quant_modules.py    # QuantLinear, QuantAct, IntLayerNorm 等
│   │       └── quant_utils.py      # batch_frexp, SymmetricQuantFunction 等
│   ├── checkpoints/
│   │   ├── qat_calibrated.pth      # ⭐ QAT 校準後的 checkpoint (46MB)
│   │   └── qat_calibrated_final.pth
│   ├── quick_qat.py                # ⭐ 快速 QAT 腳本
│   ├── extract_patterns.py         # ⭐ Pattern 抽取腳本
│   └── verify_patterns.py          # ⭐ 驗證腳本
│
├── patterns/                        # ⭐ 輸出的硬體 Pattern (285MB)
│   ├── config.json
│   ├── input/
│   ├── embeddings/
│   ├── weights/
│   ├── scales/
│   └── golden/
│
├── test_data/
│   └── test_image.JPEG             # 固定的 Goldfish 測試圖片
│
├── ImageNet/                        # ImageNet 驗證集
│   └── val/                        # 1000 類別
│
└── pyproject.toml                  # 專案依賴配置
```

---

## 量化規格總覽

### 資料型態

| 參數類型 | 位元寬度 | 格式 | 說明 |
|---------|---------|------|------|
| **Input** | INT8 | Signed | 圖片經 normalize 後量化 |
| **Weight** | INT8 | Signed | 範圍 [-127, 127] |
| **Bias** | INT32 | Signed | 確保累加精度 |
| **Activation** | INT8/INT32 | Signed | 依層而定 |
| **Scale M** | INT64 | Unsigned | Mantissa (已乘 2^31) |
| **Scale S** | INT32 | Signed | Shift amount |

### Scaling Factor 格式

I-ViT 使用定點算術，Scale 拆解為 **M (Mantissa)** 和 **S (Shift)**：

```
硬體計算: result = (input * M) >> S
```

- **M**: 由 `frexp()` 的 mantissa (0.5~1.0) 乘以 2^31 得到的整數
- **S**: 31 - exponent
- **Direction**: 統一為右移 (除法)

### Shift 方向

分析 I-ViT 的 `fixedpoint_mul.apply()`：
```python
output = z_int * m
output = torch.round(output / (2.0 ** e))  # 除法 = 右移
```

**結論**: 所有 re-quantization 都是**右移** (`direction = -1`)

---

## Pattern 輸出結構

### 目錄說明

```
patterns/
├── config.json                     # 抽取配置與統計
│
├── input/                          # 輸入圖片
│   ├── test_image.png              # 原始測試圖
│   ├── input_float.npy             # [1,3,224,224] FP32 (normalized)
│   ├── input_int8.npy              # [1,3,224,224] INT8 (quantized)
│   ├── input_int8.txt              # Hex 格式
│   └── input_scale.json            # 輸入 scale {float_scale, M, S}
│
├── embeddings/                     # 嵌入層參數
│   ├── cls_token_float.npy         # [1,1,192] FP32 原始值
│   ├── cls_token_int8.npy          # [1,1,192] INT8 量化值
│   ├── cls_token_scale_M.npy       # Mantissa
│   ├── cls_token_scale_S.npy       # Shift
│   ├── pos_embed_float.npy         # [1,197,192] FP32
│   ├── pos_embed_int8.npy          # [1,197,192] INT8
│   └── ...
│
├── weights/                        # 權重與 Bias
│   ├── patch_embed_proj_weight.npy # [192,3,16,16] INT8
│   ├── patch_embed_proj_weight.txt # Hex 格式
│   ├── patch_embed_proj_bias.npy   # [192] INT32
│   ├── blocks_0_attn_qkv_weight.npy
│   ├── blocks_0_attn_qkv_bias.npy
│   ├── blocks_0_norm1_weight.npy   # LayerNorm weight (FP32)
│   ├── blocks_0_norm1_bias_integer.npy  # LayerNorm bias (INT)
│   └── ... (共 150 個檔案)
│
├── scales/                         # Scaling Factors (M/S 格式)
│   ├── all_scales.json             # 所有 scale 的浮點原值
│   ├── qact_input_act_M.npy        # INT64
│   ├── qact_input_act_S.npy        # INT32
│   ├── qact_input_act_direction.npy # INT8 (-1=右移)
│   └── ... (共 260 組)
│
└── golden/                         # 每層 Golden Output
    ├── layer_info.json             # 所有層的 shape/dtype 資訊
    ├── qact_input_float.npy        # FP32 版本 (debug 用)
    ├── qact_input_int.npy          # INT 版本 (對照用)
    ├── qact_input_int.txt          # Hex 格式
    └── ... (共 260 對)
```

### 檔案統計

| 目錄 | 檔案數 | 說明 |
|------|--------|------|
| input/ | 5 | 測試圖片 + 量化輸入 |
| embeddings/ | 11 | cls_token + pos_embed |
| weights/ | 150 | INT8 weights + INT32 bias |
| scales/ | 781 | M + S + direction |
| golden/ | 521 | Float + Int 輸出對 |
| **總計** | **1829** | |

---

## C-model 對照指南

### 1. 載入權重

```c
// C-model 範例
int8_t qkv_weight[576][192];
int32_t qkv_bias[576];

// 從 NPY 讀取 (Python 生成後轉為 binary)
load_npy("patterns/weights/blocks_0_attn_qkv_weight.npy", qkv_weight);
load_npy("patterns/weights/blocks_0_attn_qkv_bias.npy", qkv_bias);
```

### 2. 載入 Scale (M/S 格式)

```c
// Scale 拆分為 M 和 S
int64_t M;
int32_t S;

load_npy("patterns/scales/blocks_0_qact1_act_M.npy", &M);
load_npy("patterns/scales/blocks_0_qact1_act_S.npy", &S);

// Re-quantization 計算
int64_t temp = (int64_t)accumulator * M;
int32_t output = (int32_t)(temp >> S);  // 右移
```

### 3. 驗證每層輸出

```c
// 載入 Golden Reference
int32_t golden_output[1][197][192];
load_npy("patterns/golden/blocks_0_norm1_int.npy", golden_output);

// 比對 C-model 輸出
for (int i = 0; i < 197; i++) {
    for (int j = 0; j < 192; j++) {
        if (c_model_output[i][j] != golden_output[0][i][j]) {
            printf("Mismatch at [%d][%d]: got %d, expected %d\n",
                   i, j, c_model_output[i][j], golden_output[0][i][j]);
        }
    }
}
```

### 4. Hex 檔案格式 (Verilog 用)

**INT8 (權重)**: 空格分隔，每行 16 個值
```
7F 3A 2B 1C 00 FF FE 01 02 03 04 05 06 07 08 09
A2 B3 C4 D5 E6 F7 08 19 2A 3B 4C 5D 6E 7F 80 91
```

**INT32 (Bias)**: 一行一個值，Big-Endian
```verilog
// Verilog 載入
reg [31:0] bias_mem [0:575];
initial $readmemh("blocks_0_attn_qkv_bias.txt", bias_mem);
```

### 5. 層對照表

| C-model 模組 | Python 中間層名稱 | Golden 檔案 |
|-------------|------------------|-------------|
| Patch Embed Conv | `patch_embed.proj` | `patch_embed_proj_float/int` |
| Add cls_token + pos | `qact1` | `qact1_float/int` |
| Block 0 LayerNorm1 | `blocks.0.norm1` | `blocks_0_norm1_float/int` |
| Block 0 QKV Linear | `blocks.0.attn.qkv` | `blocks_0_attn_qkv_float/int` |
| Block 0 Softmax | `blocks.0.attn.int_softmax` | `blocks_0_attn_int_softmax_float/int` |
| Block 0 GELU | `blocks.0.mlp.act` | `blocks_0_mlp_act_float/int` |
| Final LayerNorm | `norm` | `norm_float/int` |
| Head (Classifier) | `head` | `head_float/int` |

---

## 腳本使用說明

### quick_qat.py - QAT 校準

```bash
cd I-ViT
python quick_qat.py \
    --model deit_tiny \
    --data ../ImageNet \
    --epochs 1 \
    --batch-size 32 \
    --output checkpoints/qat_calibrated.pth
```

**輸出**: `checkpoints/qat_calibrated.pth` (包含模型權重 + scaling factors)

### extract_patterns.py - Pattern 抽取

```bash
python extract_patterns.py \
    --checkpoint checkpoints/qat_calibrated.pth \
    --test-image ../test_data/test_image.JPEG \
    --output ../patterns
```

**參數說明**:
- `--checkpoint`: QAT checkpoint 路徑
- `--test-image`: 固定的測試圖片
- `--output`: Pattern 輸出目錄
- `--model`: 模型類型 (deit_tiny/deit_small/deit_base)

### verify_patterns.py - 驗證輸出

```bash
python verify_patterns.py --patterns ../patterns
```

**驗證項目**:
- 目錄結構完整性
- Weight dtype = int8
- Bias dtype = int32
- M/S 重建誤差 < 1%
- Hex 檔案格式正確

---

## DeiT-Tiny 模型結構

### 整體架構

```
DeiT-Tiny (5.72M params)
│
├── qact_input                      # 輸入量化
├── patch_embed
│   └── proj: QuantConv2d [3→192, k=16, s=16]
│
├── cls_token: [1, 1, 192]          # 可訓練參數
├── pos_embed: [1, 197, 192]        # 位置編碼
├── qact_pos                        # pos_embed 量化
├── qact1                           # cat(cls, patch) + pos 量化
│
├── blocks[0-11]                    # 12 個 Transformer Block
│   ├── norm1: IntLayerNorm [192]
│   ├── qact1
│   ├── attn: Attention
│   │   ├── qkv: QuantLinear [192→576]
│   │   ├── qact1
│   │   ├── matmul_1: Q @ K^T
│   │   ├── qact_attn1
│   │   ├── int_softmax: IntSoftmax
│   │   ├── matmul_2: Attn @ V
│   │   ├── qact2
│   │   ├── proj: QuantLinear [192→192]
│   │   └── qact3
│   ├── qact2 (residual)
│   ├── norm2: IntLayerNorm [192]
│   ├── qact3
│   ├── mlp: Mlp
│   │   ├── fc1: QuantLinear [192→768]
│   │   ├── qact_gelu
│   │   ├── act: IntGELU
│   │   ├── qact1
│   │   ├── fc2: QuantLinear [768→192]
│   │   └── qact2
│   └── qact4 (residual)
│
├── norm: IntLayerNorm [192]        # Final LayerNorm
├── qact2
└── head: QuantLinear [192→1000]    # Classifier
```

### 每層維度

| 層 | 輸入維度 | 輸出維度 | 參數量 |
|----|---------|---------|--------|
| patch_embed.proj | [1,3,224,224] | [1,196,192] | 110K |
| blocks.*.attn.qkv | [1,197,192] | [1,197,576] | 111K |
| blocks.*.attn.proj | [1,197,192] | [1,197,192] | 37K |
| blocks.*.mlp.fc1 | [1,197,192] | [1,197,768] | 148K |
| blocks.*.mlp.fc2 | [1,197,768] | [1,197,192] | 148K |
| head | [1,192] | [1,1000] | 193K |

---

## 關鍵實作細節

### 1. QuantLinear 運算流程

```python
# quant_modules.py QuantLinear.forward()
# 1. 量化權重
weight_integer = quantize(weight, 8bit, fc_scaling_factor)  # INT8

# 2. 量化 Bias
bias_scaling_factor = fc_scaling_factor * input_scaling_factor
bias_integer = quantize(bias, 32bit, bias_scaling_factor)  # INT32

# 3. 整數運算
x_int = x / input_scaling_factor
output_int = x_int @ weight_integer + bias_integer

# 4. 還原為 float (硬體不需要這步)
output = output_int * bias_scaling_factor
```

### 2. IntLayerNorm 特殊處理

```python
# quant_modules.py IntLayerNorm.forward()
# LayerNorm 的 bias 處理方式不同
bias_adjusted = bias / weight  # 先除 weight
bias_integer = floor(bias_adjusted / scaling_factor)

# 輸出
y = (y_int + bias_integer) * scaling_factor * weight
```

**C-model 注意**: LayerNorm 輸出 `weight` (FP32) 和 `bias_integer` (INT)

### 3. IntSoftmax / IntGELU

這兩個層使用 **Shift-based** 近似：
- 將浮點指數運算轉為整數 shift
- 使用 `floor_ste` 做整數除法
- 輸出 scale 固定為 `1/2^(output_bit-1)`

### 4. batch_frexp 實作

```python
# quant_utils.py
def batch_frexp(inputs, max_bit=31):
    m, e = frexp(inputs)  # m ∈ [0.5, 1.0), e = exponent
    
    # 轉為整數 (已乘 2^31)
    int_m = int(m * (2 ** max_bit))
    
    return int_m, e
```

---

## 驗證結果

### QAT 訓練結果 (2026-01-08)

```
Model: DeiT-Tiny
Epochs: 1
Train samples: 50000 (Validation set as training)
Val samples: 50000

Final Results:
  Acc@1: 73.35%
  Acc@5: 91.80%
```

> ⚠️ **注意**: 上述準確率 (73.35%) 高於 I-ViT 論文報告的 DeiT-T 結果 (72.24%)，這是因為：
> - 我們使用 ImageNet **Validation Set** 同時作為訓練和驗證集
> - 這導致 **Data Leakage**，模型「看過」驗證資料
> - **這不影響 Pattern 抽取的正確性**，權重和 Scale 仍然有效
> - 如需真實泛化準確率，請使用完整的 ImageNet Train Set (138GB) 進行訓練

### Pattern 抽取結果

```
Checkpoint: I-ViT/checkpoints/qat_calibrated.pth
Test Image: Goldfish (class n01443537)
Predicted Class: 115

Statistics:
  - Total files: 1829
  - Weights: 150 (INT8 + INT32)
  - Scales: 260 (M + S + direction)
  - Golden outputs: 260 (Float + Int pairs)
  - Directory size: 285MB
```

### 驗證通過項目

| 項目 | 狀態 |
|------|------|
| 目錄結構完整 | ✅ |
| Weight dtype = int8 | ✅ |
| Bias dtype = int32 | ✅ |
| M/S 重建誤差 | ✅ 0.000000% |
| Hex 格式正確 | ✅ |
| Golden output 完整 | ✅ |

---

## 快速開始

```bash
# 1. 啟動環境
cd I_VIT
source .venv/bin/activate  # 或使用 uv

# 2. (可選) 重新執行 QAT
cd I-ViT
python quick_qat.py --model deit_tiny --data ../ImageNet --epochs 1

# 3. 抽取 Pattern
python extract_patterns.py \
    --checkpoint checkpoints/qat_calibrated.pth \
    --test-image ../test_data/test_image.JPEG \
    --output ../patterns

# 4. 驗證
python verify_patterns.py --patterns ../patterns

# 5. 使用 Pattern
# 將 patterns/ 目錄複製給 C-model 開發團隊
```

---

## 聯絡與問題

如有問題，請檢查：
1. `patterns/config.json` 確認抽取配置
2. `patterns/golden/layer_info.json` 確認各層資訊
3. `patterns/scales/all_scales.json` 確認 scale 原始值

**常見問題**:
- Overflow: 檢查是否使用正確的 accumulator 位寬 (建議 INT64)
- Shift 錯誤: 確認 M/S 的使用順序 (先乘 M，再右移 S)
- 對不上: 檢查是否有漏掉的 QuantAct 層
