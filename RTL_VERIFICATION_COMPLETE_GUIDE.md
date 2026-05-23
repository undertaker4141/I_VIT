# RTL 驗證完整指南

**專案**: I-ViT Integer Vision Transformer  
**日期**: 2026/5/22  
**狀態**: ✅ 準備就緒，可以開始 RTL 仿真

---

## 📋 目錄

1. [專案概覽](#專案概覽)
2. [工作總結](#工作總結)
3. [快速開始](#快速開始)
4. [詳細步驟](#詳細步驟)
5. [檔案結構](#檔案結構)
6. [參考文檔](#參考文檔)
7. [常見問題](#常見問題)

---

## 專案概覽

### 目標

驗證 IVIT_Verilog-master 專案的 RTL nonlinear 模組（LayerNorm, GELU, Softmax）與 PyTorch C-Model 的一致性。

### 背景

- **原問題**: 舊的驗證環境使用 TVM 版本的 C-Model，但 TVM 推論準確率為 0%
- **解決方案**: 使用 PyTorch 一致的 C-Model 重新生成測試向量
- **關鍵改進**: 
  - LayerNorm: 10 次迭代 + floor
  - GELU: x0_int=92681, n=23, shift clamp 62 bit
  - Softmax: x0_int=92681, n=15（不是 16）, shift clamp 62 bit

### 工作目錄

```
C:\Users\Public\I-ViT\
├── rtl\                        # RTL 模組（已修正）
├── nonlinear_verification\     # 驗證環境
│   ├── tb\                     # Testbenches
│   ├── test_vectors\           # 測試向量（PyTorch 版本）
│   ├── generate_test_vectors_pytorch.py
│   ├── run_modelsim.bat
│   ├── README_PYTORCH.md
│   ├── SIMULATION_GUIDE.md
│   └── RTL_MODIFICATIONS_SUMMARY.md
```

**注意**: 使用 `C:\Users\Public\I-ViT` 而非中文路徑，避免 ModelSim 模擬錯誤。

---

## 工作總結

### 已完成工作

#### 1. C-Model 模組化（Task 1）

✅ **完成日期**: 2026/5/22

- 將 C-Model 拆分為 13 個獨立模組
- 目錄結構: `cmodel_modules/` 包含 5 個子目錄
- 所有模組都有單元測試和文檔
- 報告: `CMODEL_MODULES_COMPLETION_REPORT.md`

#### 2. Golden Patterns 重新生成（Task 2）

✅ **完成日期**: 2026/5/22

- 選擇新測試圖片: `ImageNet/val/n01496331/ILSVRC2012_val_00000921.JPEG`
- Ground Truth: 5
- PyTorch 推論正確（預測 5）
- 重新生成 Golden Patterns: `golden_patterns/golden_patterns.npz`
- 報告: `golden_patterns/golden_patterns_report.txt`

#### 3. C-Model 一致性修復（Task 3）

✅ **完成日期**: 2026/5/22

- LayerNorm: 10 次迭代 + floor
- GELU/Softmax: 明確的 int64
- Softmax: n=15（原本 16）
- 加入 shift clamp（62 bit）
- 驗證: 100 張圖片，Top-1: 85%, 一致率: 95%
- 報告: `FINAL_CONSISTENCY_SUMMARY.md`

#### 4. Nonlinear 模組驗證（Task 4）

✅ **完成日期**: 2026/5/22

- 驗證 I_VIT 專案的 nonlinear 模組
- 使用 Golden Patterns 驗證 LayerNorm
- 使用隨機數據驗證 GELU 和 Softmax
- 所有模組通過驗證
- 報告: `NONLINEAR_MODULES_VERIFICATION_REPORT.md`

#### 5. RTL Team 使用指南（Task 5）

✅ **完成日期**: 2026/5/22

- 創建完整的 RTL 設計參考指南
- 包含所有模組的詳細說明
- 提供使用範例和驗證流程
- 文檔: `RTL_TEAM_GUIDE.md`

#### 6. IVIT_Verilog-master 分析（Task 6）

✅ **完成日期**: 2026/5/22

- 詳細分析所有 RTL 模組
- 分析硬體架構和記憶體配置
- 識別需要修正的問題
- 報告: `IVIT_VERILOG_ANALYSIS.md`

#### 7. RTL 驗證環境設置（Task 7）

✅ **完成日期**: 2026/5/22

**7.1 測試向量生成**:
- 創建 `generate_test_vectors_pytorch.py`
- 使用 PyTorch C-Model 生成 60 個測試向量
- LayerNorm, GELU, Softmax 各 10 組

**7.2 RTL 修正**:
- ✅ Softmax n 參數: 16 → 15
- ✅ Exponential shift clamp: 加入 62 bit 限制
- 檔案: `ivit_softmax.v`, `ivit_shift_exp.v`

**7.3 驗證環境**:
- ✅ 複製 RTL 到 `C:\Users\Public\I-ViT\rtl\`
- ✅ 複製驗證環境到 `C:\Users\Public\I-ViT\nonlinear_verification\`
- ✅ 創建仿真腳本 `run_modelsim.bat`
- ✅ 創建使用指南和模板

**文檔**:
- `README_PYTORCH.md` - PyTorch C-Model 使用指南
- `RTL_MODIFICATIONS_SUMMARY.md` - RTL 修正總結
- `SIMULATION_GUIDE.md` - 仿真執行指南
- `SIMULATION_RESULTS_TEMPLATE.md` - 結果記錄模板
- `RTL_VERIFICATION_SETUP_SUMMARY.md` - 完整設置總結

### 待完成工作

#### 8. RTL 仿真執行（Task 8）

⏳ **預估時間**: 0.5-1 天

**步驟**:
1. 運行 ModelSim 仿真
2. 驗證 LayerNorm, GELU, Softmax
3. 記錄仿真結果
4. 分析錯誤（如果有）

**執行命令**:
```bash
cd C:\Users\Public\I-ViT\nonlinear_verification
run_modelsim.bat
```

#### 9. 結果分析與報告（Task 9）

⏳ **預估時間**: 0.5 天

**步驟**:
1. 分析仿真結果
2. 填寫 `SIMULATION_RESULTS_TEMPLATE.md`
3. 創建最終驗證報告
4. 通知 RTL team

---

## 快速開始

### 前置條件

1. ✅ ModelSim 已安裝並在 PATH 中
2. ✅ 測試向量已生成（60 個 .hex 檔案）
3. ✅ RTL 已修正（Softmax n=15, shift clamp）
4. ✅ 工作目錄: `C:\Users\Public\I-ViT\nonlinear_verification`

### 執行仿真（3 步驟）

```bash
# 步驟 1: 進入工作目錄
cd C:\Users\Public\I-ViT\nonlinear_verification

# 步驟 2: 運行仿真
run_modelsim.bat

# 步驟 3: 檢查結果
# 查看 transcript 輸出，確認是否通過
```

### 預期結果

| 模組 | 預期通過率 | 允許誤差 |
|------|-----------|---------|
| LayerNorm | 100% | ±5 |
| GELU | 95%+ | ±10 |
| Softmax | 95%+ | ±2 |

---

## 詳細步驟

### 步驟 1: 環境檢查

```bash
# 檢查 ModelSim
where vlog
where vsim

# 檢查檔案
dir ..\rtl\*.v
dir tb\*.v
dir test_vectors\*.hex

# 檢查 RTL 修正
findstr /C:"N_CFG = 15" ..\rtl\ivit_softmax.v
findstr /C:"shift_amt_clamped" ..\rtl\ivit_shift_exp.v
```

### 步驟 2: 編譯 RTL

```bash
# 建立工作目錄
vlib work

# 編譯 RTL 模組
vlog -work work ..\rtl\ivit_layernorm.v
vlog -work work ..\rtl\ivit_shift_exp.v
vlog -work work ..\rtl\ivit_gelu.v
vlog -work work ..\rtl\ivit_softmax.v
vlog -work work ..\rtl\ivit_requant.v
vlog -work work ..\rtl\ivit_nonlinear_top.v

# 編譯 Testbench
vlog -work work tb\tb_layernorm.v
vlog -work work tb\tb_gelu.v
vlog -work work tb\tb_softmax.v
vlog -work work tb\tb_nonlinear_top.v
```

### 步驟 3: 運行仿真

```bash
# LayerNorm (預計 5-10 分鐘)
vsim -c -do "run -all; quit" work.tb_layernorm

# GELU (預計 2-5 分鐘)
vsim -c -do "run -all; quit" work.tb_gelu

# Softmax (預計 2-5 分鐘)
vsim -c -do "run -all; quit" work.tb_softmax
```

### 步驟 4: 分析結果

查看 transcript 輸出，尋找：

**成功標誌**:
```
✅✅✅ LayerNorm 測試全部通過！✅✅✅
✅✅✅ GELU 測試全部通過！✅✅✅
✅✅✅ Softmax 測試全部通過！✅✅✅
```

**失敗標誌**:
```
❌ FAIL - X/Y 個錯誤 (錯誤率: Z%)
```

### 步驟 5: 記錄結果

使用 `SIMULATION_RESULTS_TEMPLATE.md` 記錄結果：

```bash
# 複製模板
copy SIMULATION_RESULTS_TEMPLATE.md SIMULATION_RESULTS.md

# 編輯 SIMULATION_RESULTS.md，填入實際結果
```

---

## 檔案結構

### I_VIT 專案（C-Model）

```
C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\
├── cmodel_modules\                    # 模組化 C-Model
│   ├── normalization\
│   │   └── int_layer_norm.py
│   ├── activation\
│   │   ├── int_gelu.py
│   │   ├── int_softmax.py
│   │   └── int_exp_shift.py
│   ├── quantization\
│   │   └── requantize_integer.py
│   └── utils\
├── cmodel_rtl_reference\              # RTL 參考實現
│   ├── pytorch_integer_cmodel.py     # ⭐ 用於生成測試向量
│   ├── pure_numpy_cmodel.py
│   └── nonlinear_cmodel_reference.py # ⚠️ TVM 版本（已棄用）
├── golden_patterns\                   # Golden Patterns
│   ├── golden_patterns.npz
│   └── golden_patterns_report.txt
├── RTL_TEAM_GUIDE.md                 # RTL 設計指南
├── IVIT_VERILOG_ANALYSIS.md          # RTL 分析報告
├── NONLINEAR_MODULES_VERIFICATION_REPORT.md
├── FINAL_CONSISTENCY_SUMMARY.md
└── RTL_VERIFICATION_COMPLETE_GUIDE.md # ⭐ 本文檔
```

### IVIT_Verilog-master 專案（RTL 原始碼）

```
C:\桌面\冠泓\大學\專題\I-ViT\IVIT_Verilog-master\
├── rtl\                               # RTL 模組（原始版本）
│   ├── ivit_layernorm.v
│   ├── ivit_gelu.v
│   ├── ivit_softmax.v               # ⚠️ N_CFG = 16（需修正）
│   ├── ivit_shift_exp.v             # ⚠️ 缺少 shift clamp
│   ├── ivit_requant.v
│   └── ivit_nonlinear_top.v
└── nonlinear_verification\            # 驗證環境（舊版）
    └── test_vectors\                  # ⚠️ 使用 TVM 版本
```

### 工作目錄（驗證環境）

```
C:\Users\Public\I-ViT\
├── rtl\                               # RTL 模組（已修正）
│   ├── ivit_layernorm.v
│   ├── ivit_gelu.v
│   ├── ivit_softmax.v               # ✅ N_CFG = 15
│   ├── ivit_shift_exp.v             # ✅ 已加入 shift clamp
│   ├── ivit_requant.v
│   └── ivit_nonlinear_top.v
└── nonlinear_verification\
    ├── tb\                            # Testbenches
    │   ├── tb_layernorm.v
    │   ├── tb_gelu.v
    │   ├── tb_softmax.v
    │   └── tb_nonlinear_top.v
    ├── test_vectors\                  # ✅ 使用 PyTorch 版本
    │   ├── layernorm_input_0.hex ~ 9.hex
    │   ├── layernorm_golden_0.hex ~ 9.hex
    │   ├── gelu_input_0.hex ~ 9.hex
    │   ├── gelu_golden_0.hex ~ 9.hex
    │   ├── softmax_input_0.hex ~ 9.hex
    │   └── softmax_golden_0.hex ~ 9.hex
    ├── generate_test_vectors_pytorch.py  # 測試向量生成腳本
    ├── run_modelsim.bat                  # 仿真執行腳本
    ├── README_PYTORCH.md                 # 使用指南
    ├── SIMULATION_GUIDE.md               # 仿真指南
    ├── RTL_MODIFICATIONS_SUMMARY.md      # RTL 修正總結
    └── SIMULATION_RESULTS_TEMPLATE.md    # 結果模板
```

---

## 參考文檔

### 核心文檔（必讀）

1. **`README_PYTORCH.md`**
   - PyTorch C-Model 使用指南
   - 測試向量生成方法
   - 位置: `C:\Users\Public\I-ViT\nonlinear_verification\`

2. **`SIMULATION_GUIDE.md`**
   - 仿真執行詳細指南
   - 故障排除方法
   - 位置: `C:\Users\Public\I-ViT\nonlinear_verification\`

3. **`RTL_MODIFICATIONS_SUMMARY.md`**
   - RTL 修正詳細說明
   - 修正前後對比
   - 位置: `C:\Users\Public\I-ViT\nonlinear_verification\`

### 背景文檔（參考）

4. **`RTL_TEAM_GUIDE.md`**
   - RTL 設計參考指南
   - 所有模組的詳細說明
   - 位置: `C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\`

5. **`IVIT_VERILOG_ANALYSIS.md`**
   - IVIT_Verilog-master 完整分析
   - RTL 架構說明
   - 位置: `C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\`

6. **`FINAL_CONSISTENCY_SUMMARY.md`**
   - C-Model 一致性修復總結
   - PyTorch vs TVM 對比
   - 位置: `C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\`

7. **`NONLINEAR_MODULES_VERIFICATION_REPORT.md`**
   - I_VIT Nonlinear 模組驗證報告
   - 位置: `C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\`

8. **`RTL_VERIFICATION_SETUP_SUMMARY.md`**
   - 驗證環境設置完整總結
   - 位置: `C:\桌面\冠泓\大學\專題\I-ViT\I_VIT\`

---

## 常見問題

### Q1: 為什麼要使用 `C:\Users\Public\I-ViT` 而不是原始路徑？

**A**: 原始路徑包含中文字元（`C:\桌面\冠泓\...`），ModelSim 可能無法正確處理，導致模擬錯誤。使用純英文路徑可以避免這個問題。

### Q2: 為什麼不使用 TVM 版本的 C-Model？

**A**: TVM 推論的準確率是 0%，生成的測試向量不可靠。PyTorch 版本的準確率是 85%+，與原始模型一致。

### Q3: Softmax 的 n 參數為什麼是 15 而不是 16？

**A**: PyTorch 原始實現使用 n=15，TVM 版本錯誤使用 n=16。必須與訓練時的參數一致才能通過驗證。

### Q4: 什麼是 shift clamp？為什麼需要？

**A**: Shift clamp 限制 shift 操作的最大位數為 62 bit，防止 Verilog 的 shift 溢位。PyTorch C-Model 也有相同的限制，必須保持一致。

### Q5: 如果仿真失敗怎麼辦？

**A**: 
1. 檢查錯誤率和最大誤差
2. 如果誤差小（≤5），可能是量化誤差，可以接受
3. 如果誤差大或錯誤率高，檢查 RTL 修正是否正確套用
4. 參考 `SIMULATION_GUIDE.md` 的故障排除章節

### Q6: 測試向量如何生成？

**A**: 使用 `generate_test_vectors_pytorch.py` 腳本，基於 I_VIT 的 `pytorch_integer_cmodel.py`。詳見 `README_PYTORCH.md`。

### Q7: 如何確認 RTL 修正已正確套用？

**A**: 
```bash
# 檢查 Softmax N_CFG
findstr /C:"N_CFG = 15" C:\Users\Public\I-ViT\rtl\ivit_softmax.v

# 檢查 shift clamp
findstr /C:"shift_amt_clamped" C:\Users\Public\I-ViT\rtl\ivit_shift_exp.v
findstr /C:"shift_amt_clamped" C:\Users\Public\I-ViT\rtl\ivit_softmax.v
```

### Q8: 仿真需要多久？

**A**: 
- LayerNorm: 5-10 分鐘
- GELU: 2-5 分鐘
- Softmax: 2-5 分鐘
- 總計: 約 10-20 分鐘

### Q9: 如何使用 GUI 模式除錯？

**A**: 
```bash
vsim -gui work.tb_layernorm
# 在 GUI 中: add wave -r /*, run 1us
```

### Q10: 下一步是什麼？

**A**: 
1. 運行 RTL 仿真
2. 記錄結果
3. 如果通過，通知 RTL team
4. 如果失敗，分析錯誤並修正
5. 考慮使用 I_VIT Golden Patterns 進行更詳細驗證

---

## 總結

### 關鍵成就

✅ **C-Model 模組化** - 13 個獨立模組，易於維護和測試  
✅ **Golden Patterns 重新生成** - 使用正確的測試圖片  
✅ **C-Model 一致性修復** - 與 PyTorch 原始碼完全一致  
✅ **Nonlinear 模組驗證** - 所有模組通過驗證  
✅ **RTL Team 指南** - 完整的設計參考文檔  
✅ **RTL 分析** - 詳細分析 IVIT_Verilog-master  
✅ **驗證環境設置** - 測試向量、RTL 修正、仿真腳本全部就緒

### 當前狀態

🎯 **準備就緒** - 可以開始 RTL 仿真

### 下一步行動

1. ⏳ 運行 ModelSim 仿真
2. ⏳ 記錄仿真結果
3. ⏳ 分析結果並創建報告
4. ⏳ 通知 RTL team

---

**文檔完成**  
**日期**: 2026/5/22  
**版本**: 1.0  
**作者**: Kiro AI Assistant
