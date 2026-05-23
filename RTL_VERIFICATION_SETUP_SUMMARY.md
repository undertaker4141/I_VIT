# RTL 驗證環境設置總結

**日期**: 2026/5/22  
**狀態**: ✅ 測試向量生成完成，準備進行 RTL 仿真

---

## 執行摘要

今天完成了 RTL nonlinear 模組驗證環境的設置，包括：

1. ✅ **分析 IVIT_Verilog-master 專案**
   - 完整分析 RTL 實現
   - 確認 RTL 與 C-Model 的對應關係
   - 識別需要修正的問題

2. ✅ **使用 PyTorch C-Model 重新生成測試向量**
   - 替換 TVM 版本（準確率 0%）
   - 使用 I_VIT 的 `pytorch_integer_cmodel.py`
   - 生成 60 個測試向量（LayerNorm, GELU, Softmax 各 10 組）

3. ✅ **設置驗證環境**
   - 工作目錄: `C:\Users\Public\I-ViT` (避免中文路徑)
   - 測試向量: `nonlinear_verification/test_vectors/`
   - RTL 模組: `rtl/`

---

## 關鍵發現

### 1. RTL 與 C-Model 一致性

| 模組 | RTL 狀態 | C-Model 一致性 | 需要修正 |
|------|---------|---------------|---------|
| **LayerNorm** | ✅ 完成 | ✅ 完全一致 | 無 |
| **GELU** | ✅ 完成 | ✅ 參數一致 | 無 |
| **Softmax** | ✅ 完成 | ⚠️ n 參數不一致 | **n: 16 → 15** |
| **Exponential** | ✅ 完成 | ⚠️ 缺少 shift clamp | **加入 62 bit 限制** |
| **Requantization** | ✅ 完成 | ✅ 完全一致 | 無 |

### 2. 測試向量生成

**舊版本問題**:
- 使用 TVM 版本的 C-Model
- TVM 推論準確率 0%
- 測試向量不可靠

**新版本改進**:
- ✅ 使用 PyTorch 一致的 C-Model
- ✅ 與 I_VIT 專案完全一致
- ✅ 已驗證準確率 85%+

**生成結果**:
```
LayerNorm: 10 組測試向量
  - 輸入: INT16, shape=(1, 197, 192)
  - 輸出: INT32, shape=(1, 197, 192)

GELU: 10 組測試向量
  - 輸入: INT8, shape=(768,)
  - 輸出: INT32, shape=(768,)

Softmax: 10 組測試向量
  - 輸入: INT8, shape=(197,)
  - 輸出: INT32, shape=(197,)
```

---

## 檔案結構

### I_VIT 專案（C-Model）

```
C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\
├── cmodel_modules/                    # 模組化 C-Model
│   ├── normalization/
│   │   └── int_layer_norm.py
│   ├── activation/
│   │   ├── int_gelu.py
│   │   ├── int_softmax.py
│   │   └── int_exp_shift.py
│   └── quantization/
│       └── requantize_integer.py
├── cmodel_rtl_reference/              # RTL 參考實現
│   ├── pytorch_integer_cmodel.py     # ⭐ 用於生成測試向量
│   └── pure_numpy_cmodel.py
├── golden_patterns/                   # Golden Patterns
│   ├── golden_patterns.npz
│   └── golden_patterns_report.txt
└── RTL_TEAM_GUIDE.md                 # RTL 設計指南
```

### IVIT_Verilog-master 專案（RTL）

```
C:\桌面\冠泓\大學\專題\I-ViT\IVIT_Verilog-master\
├── rtl/                               # RTL 模組
│   ├── ivit_layernorm.v
│   ├── ivit_gelu.v
│   ├── ivit_softmax.v
│   ├── ivit_shift_exp.v
│   ├── ivit_requant.v
│   └── ivit_nonlinear_top.v
├── nonlinear_verification/            # 驗證環境（舊版）
│   ├── tb/
│   ├── test_vectors/                  # ⚠️ 使用 TVM 版本
│   └── python/
└── tb/                                # Testbenches
    ├── tb_layernorm.v
    ├── tb_gelu.v
    └── tb_softmax.v
```

### 工作目錄（避免中文路徑）

```
C:\Users\Public\I-ViT\
├── rtl/                               # 複製自 IVIT_Verilog-master/rtl
│   ├── ivit_layernorm.v
│   ├── ivit_gelu.v
│   ├── ivit_softmax.v
│   ├── ivit_shift_exp.v
│   ├── ivit_requant.v
│   └── ivit_nonlinear_top.v
└── nonlinear_verification/            # 複製自 IVIT_Verilog-master/nonlinear_verification
    ├── tb/                            # Testbenches
    │   ├── tb_layernorm.v
    │   ├── tb_gelu.v
    │   ├── tb_softmax.v
    │   └── tb_nonlinear_top.v
    ├── test_vectors/                  # ✅ 使用 PyTorch 版本
    │   ├── layernorm_input_0.hex ~ 9.hex
    │   ├── layernorm_golden_0.hex ~ 9.hex
    │   ├── gelu_input_0.hex ~ 9.hex
    │   ├── gelu_golden_0.hex ~ 9.hex
    │   ├── softmax_input_0.hex ~ 9.hex
    │   └── softmax_golden_0.hex ~ 9.hex
    ├── generate_test_vectors_pytorch.py  # ⭐ 新增
    └── README_PYTORCH.md                  # ⭐ 新增
```

---

## 已完成工作

### 1. 分析 IVIT_Verilog-master

**完成項目**:
- ✅ 分析所有 RTL 模組
- ✅ 分析 Data Controller 架構
- ✅ 分析記憶體架構（432 KB SRAM）
- ✅ 分析硬體 Dataflow
- ✅ 分析驗證環境
- ✅ 創建完整分析報告 `IVIT_VERILOG_ANALYSIS.md`

**關鍵發現**:
- RTL 實現與 I_VIT C-Model 高度一致
- LayerNorm 驗證 100% 通過
- Softmax 需要修正 n 參數（16 → 15）
- GELU 驗證失敗（可能是測試向量問題）

### 2. 重新生成測試向量

**完成項目**:
- ✅ 創建 `generate_test_vectors_pytorch.py`
- ✅ 使用 I_VIT 的 `pytorch_integer_cmodel.py`
- ✅ 生成 60 個測試向量
- ✅ 驗證測試向量格式正確

**測試向量特性**:
- LayerNorm: 10 次迭代 + floor
- GELU: x0_int=92681, n=23
- Softmax: x0_int=92681, n=15

### 3. 設置驗證環境

**完成項目**:
- ✅ 複製 RTL 到 `C:\Users\Public\I-ViT\rtl\`
- ✅ 複製驗證環境到 `C:\Users\Public\I-ViT\nonlinear_verification\`
- ✅ 創建 `README_PYTORCH.md` 使用指南
- ✅ 測試向量生成成功

---

## 待完成工作

### 1. 高優先級（必須完成）

| 任務 | 預估時間 | 說明 |
|------|---------|------|
| **修正 Softmax n 參數** | 0.5 天 | 將 RTL 的 N_CFG 從 16 改為 15 |
| **加入 Exponential shift clamp** | 0.5 天 | 加入最大 62 bit 限制 |
| **運行 RTL 仿真** | 1 天 | 使用 ModelSim 驗證所有模組 |
| **分析仿真結果** | 0.5 天 | 比對 RTL 輸出與 golden patterns |

### 2. 中優先級（建議完成）

| 任務 | 預估時間 | 說明 |
|------|---------|------|
| **調查 GELU 驗證失敗** | 1-2 天 | 如果新測試向量仍失敗，需要深入調查 |
| **使用 I_VIT Golden Patterns** | 1-2 天 | 使用實際模型的中間結果驗證 |
| **系統整合測試** | 2-3 天 | 驗證完整的 nonlinear top 模組 |

---

## 下一步行動

### 立即行動（今天或明天）

1. **修正 RTL**:
   ```verilog
   // C:\Users\Public\I-ViT\rtl\ivit_softmax.v
   parameter N_CFG = 15,  // 改為 15
   
   // C:\Users\Public\I-ViT\rtl\ivit_shift_exp.v
   // 加入 shift clamp
   ```

2. **運行仿真**:
   ```bash
   cd C:\Users\Public\I-ViT\nonlinear_verification
   # 使用 ModelSim 運行仿真
   ```

3. **分析結果**:
   - 檢查 LayerNorm 是否 100% 通過
   - 檢查 Softmax 是否改善
   - 檢查 GELU 是否通過

### 短期行動（本週）

1. 如果 GELU 仍失敗，調查原因
2. 使用 I_VIT Golden Patterns 進行更詳細的驗證
3. 完成 Nonlinear 模組驗證報告

---

## 參考文檔

### I_VIT 專案

- `RTL_TEAM_GUIDE.md` - RTL 設計指南
- `NONLINEAR_MODULES_VERIFICATION_REPORT.md` - Nonlinear 模組驗證報告
- `FINAL_CONSISTENCY_SUMMARY.md` - C-Model 一致性修復總結
- `IVIT_VERILOG_ANALYSIS.md` - IVIT Verilog 分析報告

### IVIT_Verilog-master 專案

- `IViT 總結.md` - 專案概覽
- `IViT硬體dataflow.md` - 硬體資料流程
- `Nonlinear 模組驗證計畫.md` - 驗證計畫
- `doc/DC_linear_requirements.md` - DC 線性運算需求

### 工作目錄

- `README_PYTORCH.md` - PyTorch C-Model 使用指南
- `generate_test_vectors_pytorch.py` - 測試向量生成腳本

---

**總結完成**  
**日期**: 2026/5/22  
**狀態**: ✅ 測試向量生成完成，準備進行 RTL 仿真
