# I-ViT Pattern Extraction for Hardware Accelerator Cmodel

## 確認的執行方案

| 項目 | 決定 |
|------|------|
| **模型** | DeiT-Tiny (5.7M params, ~5.7MB INT8) |
| **測試圖片** | 固定 Goldfish 圖片 (`n01443537`) |
| **輸出格式** | `.npy` (主要) + `.txt` Hex (Verilog TB) |
| **QAT 策略** | 1 Epoch on Validation Set (無 Mixup/CutMix) |
| **Golden Output** | INT + FP32 兩者都輸出 |

---

## ⚠️ 硬體關鍵設計決策

### 1. Weight & Bias 位元寬度

| 參數類型 | 位元寬度 | 程式碼來源 |
|---------|---------|-----------|
| Weight | **INT8** | `weight_bit=8` (QuantLinear, line 32) |
| Bias | **INT32** | `bias_bit=32` (QuantLinear, line 33) |

### 2. Scale 拆解為 M (Mantissa) + S (Shift)

> [!IMPORTANT]
> **I-ViT 的 `batch_frexp()` 已經輸出整數 M！**  
> 
> 程式碼 (`quant_utils.py` line 167-169):
> ```python
> int_m_shifted = int(Decimal(m * (2 ** max_bit)).quantize(...))  # max_bit=31
> ```
> 所以 M 是 **0.5~1.0 乘以 2^31 後的整數**，不需要額外轉換。

**硬體計算公式**: `result = (input * M) >> S`

| 輸出檔案 | dtype | 說明 |
|---------|-------|------|
| `*_M.npy` | **int64** | Mantissa (已乘 2^31 的整數) |
| `*_S.npy` | **int32** | Shift amount (exponent) |
| `*_direction.npy` | **int8** | `+1`=左移, `-1`=右移 |

### 3. Shift 方向

分析 `fixedpoint_mul.apply()` (`quant_utils.py` line 220-230):
```python
output = z_int.type(torch.double) * m.type(torch.double)
output = torch.round(output / (2.0 ** e))  # 右移 (除法)
```

**結論**: I-ViT 的 re-quantization 都是**右移 (除法)**。`direction = -1`

---

## 🆕 遺漏的參數 (已補齊)

### 1. Class Token & Position Embedding

程式碼來源 (`vit_quant.py` line 186-188):
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1, 1, 192]
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # [1, 197, 192]
```

**Forward 流程** (`vit_quant.py` line 259-265):
```python
cls_tokens = self.cls_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)  # Append class token
x_pos, act_scaling_factor_pos = self.qact_pos(self.pos_embed)  # Quantize pos_embed
x, act_scaling_factor = self.qact1(x, ..., x_pos, act_scaling_factor_pos)  # Add
```

> [!WARNING]
> **cls_token 和 pos_embed 需要量化！**  
> 它們經過 `qact_pos` 和 `qact1` 做量化處理，需要輸出 INT8 版本和對應的 scale。

### 2. LayerNorm Weight & Bias

程式碼來源 (`quant_modules.py` IntLayerNorm, line 333-386):
- `self.weight`: 繼承自 `nn.LayerNorm`，形狀 `[192]`
- `self.bias`: 繼承自 `nn.LayerNorm`，形狀 `[192]`

**特殊處理** (line 377-380):
```python
bias = self.bias.data.detach() / (self.weight.data.detach())
bias_int = floor_ste.apply(bias / scaling_factor)  # 先除 weight 再量化
self.bias_integer = bias_int
```

**結論**: LayerNorm 的量化方式不同，需要單獨處理：
- `weight`: 直接作為 scale 因子使用 (FP32 → 需轉 M/S)
- `bias_integer`: 已量化的 bias (INT)

### 3. Final Head

程式碼來源 (`vit_quant.py` line 233-239):
```python
self.head = QuantLinear(self.num_features, num_classes)  # [192 → 1000]
```

---

## 完整輸出目錄結構

```
patterns/
├── config.json                         # 模型設定總覽
│
├── input/
│   ├── test_image.JPEG                 # 原始圖片
│   ├── input_int8.npy                  # [1, 3, 224, 224] INT8
│   ├── input_int8.txt                  # Hex 格式
│   ├── input_float.npy                 # [1, 3, 224, 224] FP32
│   └── input_scale.json                # scale 值
│
├── embeddings/                         # 🆕 嵌入層參數
│   ├── cls_token_int8.npy              # [1, 1, 192] INT8
│   ├── cls_token_scale_M.npy           # M (int64)
│   ├── cls_token_scale_S.npy           # S (int32)
│   ├── pos_embed_int8.npy              # [1, 197, 192] INT8
│   ├── pos_embed_scale_M.npy
│   └── pos_embed_scale_S.npy
│
├── weights/                            # 權重 & Bias
│   │── # Patch Embed
│   ├── patch_embed_proj_weight.npy     # [192, 3, 16, 16] INT8
│   ├── patch_embed_proj_weight.txt     # Hex
│   ├── patch_embed_proj_bias.npy       # [192] INT32
│   ├── patch_embed_proj_bias.txt       # Hex
│   │
│   │── # Block 0-11 (每個 block 相同結構)
│   ├── block_0_attn_qkv_weight.npy     # [576, 192] INT8
│   ├── block_0_attn_qkv_bias.npy       # [576] INT32
│   ├── block_0_attn_proj_weight.npy    # [192, 192] INT8
│   ├── block_0_attn_proj_bias.npy      # [192] INT32
│   ├── block_0_mlp_fc1_weight.npy      # [768, 192] INT8
│   ├── block_0_mlp_fc1_bias.npy        # [768] INT32
│   ├── block_0_mlp_fc2_weight.npy      # [192, 768] INT8
│   ├── block_0_mlp_fc2_bias.npy        # [192] INT32
│   │
│   │── # 🆕 LayerNorm 參數
│   ├── block_0_norm1_weight.npy        # [192] FP32 (作為 scale)
│   ├── block_0_norm1_bias_integer.npy  # [192] INT (量化後)
│   ├── block_0_norm2_weight.npy        # [192] FP32
│   ├── block_0_norm2_bias_integer.npy  # [192] INT
│   │
│   │── # ... Block 1-11
│   │
│   │── # Final LayerNorm & Head
│   ├── final_norm_weight.npy           # [192] FP32
│   ├── final_norm_bias_integer.npy     # [192] INT
│   ├── head_weight.npy                 # [1000, 192] INT8
│   └── head_bias.npy                   # [1000] INT32
│
├── scales/                             # Scaling Factors (M/S 格式)
│   ├── all_scales.json                 # 所有 scale 的浮點原值 (參考用)
│   │
│   │── # Input Quantization
│   ├── qact_input_M.npy                # int64
│   ├── qact_input_S.npy                # int32
│   │
│   │── # Block 0 (每個 block 相同結構)
│   ├── block_0_qact1_M.npy             # after LayerNorm1
│   ├── block_0_qact1_S.npy
│   ├── block_0_attn_qkv_fc_M.npy       # weight scale
│   ├── block_0_attn_qkv_fc_S.npy
│   ├── block_0_attn_matmul1_M.npy      # Q @ K^T
│   ├── block_0_attn_matmul1_S.npy
│   ├── block_0_softmax_M.npy           # Softmax output scale
│   ├── block_0_softmax_S.npy
│   ├── block_0_gelu_M.npy
│   ├── block_0_gelu_S.npy
│   └── ...
│
└── golden/                             # 每層 Golden Output
    │── layer_info.json                 # 所有層的 shape 和 dtype
    │
    │── # Patch Embedding
    ├── patch_embed_int.npy             # [1, 196, 192] INT
    ├── patch_embed_float.npy           # [1, 196, 192] FP32
    ├── patch_embed_int.txt             # Hex
    │
    │── # After Adding cls_token + pos_embed
    ├── after_pos_embed_int.npy         # [1, 197, 192] INT
    ├── after_pos_embed_float.npy
    │
    │── # Block 0 (共 12 blocks)
    ├── block_0_norm1_int.npy           # LayerNorm 後
    ├── block_0_norm1_float.npy
    ├── block_0_attn_qkv_int.npy        # QKV Linear 後
    ├── block_0_attn_qkv_float.npy
    ├── block_0_attn_score_int.npy      # Q @ K^T (pre-softmax)
    ├── block_0_attn_score_float.npy
    ├── block_0_softmax_int.npy         # ⭐ Softmax 後
    ├── block_0_softmax_float.npy
    ├── block_0_attn_output_int.npy     # Attention 完整輸出
    ├── block_0_attn_output_float.npy
    ├── block_0_residual1_int.npy       # 第一個殘差加法後
    ├── block_0_residual1_float.npy
    ├── block_0_norm2_int.npy           # LayerNorm2 後
    ├── block_0_norm2_float.npy
    ├── block_0_mlp_fc1_int.npy         # MLP FC1 後
    ├── block_0_mlp_fc1_float.npy
    ├── block_0_gelu_int.npy            # ⭐ GELU 後
    ├── block_0_gelu_float.npy
    ├── block_0_mlp_fc2_int.npy         # MLP FC2 後
    ├── block_0_mlp_fc2_float.npy
    ├── block_0_output_int.npy          # Block 完整輸出 (殘差後)
    ├── block_0_output_float.npy
    │
    │── # ... Block 1-11 同樣結構
    │
    │── # Final
    ├── final_norm_int.npy              # 最終 LayerNorm 後
    ├── final_norm_float.npy
    ├── cls_token_output_int.npy        # 取出 cls token [1, 192]
    └── final_output.npy                # [1, 1000] logits
```

---

## Hex 格式規範

### INT8 權重 (人類可讀，空格分隔)
```
# block_0_attn_qkv_weight.txt
# Shape: [576, 192]
# Dtype: int8
# Format: row-major, space-separated, 2-digit hex per byte
#
# Row 0 (192 values)
7F 3A 2B 1C 00 FF FE 01 02 03 ...
# Row 1
A2 B3 C4 D5 E6 F7 08 19 2A 3B ...
```

### INT32 Bias (Big-Endian，一行一個數值)
```
# block_0_attn_qkv_bias.txt
# Shape: [576]
# Dtype: int32
# Format: one value per line, 8-digit hex, big-endian (human-readable)
# Compatible with Verilog $readmemh
#
00001234
FFFFFFFE
00000042
...
```

> [!TIP]
> **Verilog 使用方式**:
> ```verilog
> reg [31:0] bias_mem [0:575];
> initial $readmemh("block_0_attn_qkv_bias.txt", bias_mem);
> ```

---

## DeiT-Tiny 完整結構 (供參考)

```
DeiT-Tiny
├── qact_input                          # 輸入量化
├── patch_embed
│   ├── proj: QuantConv2d [3 → 192, k=16, s=16]
│   └── qact
├── cls_token: Parameter [1, 1, 192]    # 🆕
├── pos_embed: Parameter [1, 197, 192]  # 🆕
├── qact_pos                            # pos_embed 量化
├── qact1                               # cls+pos add 量化
│
├── blocks[0-11]
│   ├── norm1: IntLayerNorm [192]       # weight + bias_integer
│   ├── qact1
│   ├── attn
│   │   ├── qkv: QuantLinear [192 → 576]
│   │   ├── qact1
│   │   ├── matmul_1 (Q @ K^T)
│   │   ├── qact_attn1
│   │   ├── int_softmax (IntSoftmax)
│   │   ├── matmul_2 (attn @ V)
│   │   ├── qact2
│   │   ├── proj: QuantLinear [192 → 192]
│   │   └── qact3
│   ├── qact2                           # residual add 量化
│   ├── norm2: IntLayerNorm [192]
│   ├── qact3
│   ├── mlp
│   │   ├── fc1: QuantLinear [192 → 768]
│   │   ├── qact_gelu
│   │   ├── act: IntGELU
│   │   ├── qact1
│   │   ├── fc2: QuantLinear [768 → 192]
│   │   └── qact2
│   └── qact4                           # residual add 量化
│
├── norm: IntLayerNorm [192]            # Final LayerNorm
├── qact2
└── head: QuantLinear [192 → 1000]      # Classifier
```

---

## 執行流程

```bash
# Step 1: 準備測試圖片
mkdir -p test_data
cp ImageNet/val/n01443537/ILSVRC2012_val_00000293.JPEG test_data/test_image.JPEG

# Step 2: Quick QAT (約 5-10 分鐘)
cd I-ViT
python quick_qat.py \
    --model deit_tiny \
    --data ../ImageNet \
    --epochs 1 \
    --batch-size 32 \
    --no-mixup \
    --output checkpoints/qat_calibrated.pth

# Step 3: 抽取 Patterns
python extract_patterns.py \
    --checkpoint checkpoints/qat_calibrated.pth \
    --test-image ../test_data/test_image.JPEG \
    --output ../patterns

# Step 4: 驗證
python verify_patterns.py --patterns ../patterns
```

---

## 驗證 Checklist

- [ ] Weight dtype = int8, range [-128, 127]
- [ ] Bias dtype = int32
- [ ] M dtype = int64 (batch_frexp 輸出)
- [ ] S dtype = int32
- [ ] cls_token 和 pos_embed 已量化
- [ ] LayerNorm 的 weight 和 bias_integer 都有輸出
- [ ] Hex INT32 採用 Big-Endian (一行一個值)
- [ ] Golden output 每層都有 INT + Float 兩個版本
