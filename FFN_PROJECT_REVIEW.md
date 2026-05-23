# FFN 專案完整查看報告

**日期**: 2026-05-23  
**專案位置**: `C:\桌面\冠泓\大學\專題\I-ViT\FFN`

---

## 📋 專案概述

這是一個基於 **Xilinx Zynq FPGA** 的 **Feed-Forward Neural Network (FFN) 硬體加速器**專案，專門用於加速兩層前饋神經網路的推理計算。

### 核心特點
- ✅ 使用 **8-bit 量化整數運算**（INT8）提高效能
- ✅ 實現 **64 個 MAC（乘加累積）單元**並行計算（8 個 PE，每個 PE 8 個 MAC）
- ✅ 透過 **AXI4-Stream** 介面與 ARM 處理器通訊
- ✅ 支援 **可配置的量化參數**（Post-Training Quantization）
- ✅ 完整的 C 程式碼驅動（ARM 端）
- ✅ SystemVerilog Testbench 驗證

---

## 📁 專案結構

```
FFN/
├── src/                          # Verilog 原始碼
│   ├── FFN_top.v                # 頂層模組
│   ├── main_ctrl/               # 控制模組
│   │   ├── fsm64.v             # 主狀態機 (IDLE→FSLD→BASE→DONE)
│   │   ├── schedule_ctrl.v     # 排程控制器
│   │   ├── get_ins.v           # 指令解析器
│   │   ├── d_empn_rd_mux.v     # 資料多工器
│   │   └── yolo_rst_if.v       # 重置介面
│   ├── compute_engine/          # 計算引擎
│   │   ├── pe_top.v            # PE 頂層
│   │   ├── pe_array.v          # PE 陣列控制器
│   │   ├── pe_8e.v             # 單個 PE（8 MAC，7 級流水線）
│   │   ├── getpe_result.v      # 結果聚合器
│   │   ├── quan2uint8.v        # 量化模組（32-bit → 8-bit）
│   │   └── pe_qz_pkg.v         # 封裝模組（8×8-bit → 64-bit）
│   ├── input_module/            # 輸入 SRAM 模組
│   │   ├── if_top.v
│   │   ├── ifsram_r.v
│   │   └── ifsram_w.v
│   ├── kernel_module/           # 權重 SRAM 模組（8 banks）
│   │   ├── ker_top.v
│   │   ├── kersram_r.v
│   │   └── kersram_w.v
│   ├── bias_module/             # 偏置 SRAM 模組
│   │   ├── bias_top.v
│   │   ├── biassram_r.v
│   │   └── biassram_w.v
│   ├── output_module/           # 輸出 SRAM 模組
│   │   ├── ot_top.v
│   │   ├── otsram_r.v
│   │   ├── otsram_w.v
│   │   └── ot_fifo.v
│   └── other_module/            # 其他模組
│       ├── fifo/
│       │   ├── INPUT_STREAM_if.v   # AXI 輸入介面
│       │   └── OUTPUT_STREAM_if.v  # AXI 輸出介面
│       └── counter/
│           ├── count_yi_v3.v
│           ├── count_yi_v4.v
│           └── count_yi_v5.v

├── tb/                          # 測試平台
│   └── FFN_tb.sv               # SystemVerilog 測試檔
├── Ccode/                       # C 程式碼（ARM 端）
│   ├── main.c                  # 主程式
│   ├── axi_dma.c/h             # DMA 驅動
│   ├── cs_ip.c/h               # FFN 層執行函式
│   ├── v3_data.c/h             # 資料載入函式
│   └── platform.c/h            # 平台配置
├── *.dat                        # 測試資料檔案
│   ├── input_token.dat         # 輸入特徵
│   ├── FFN_W1.dat              # FFN Layer 1 權重
│   ├── FFN_W2.dat              # FFN Layer 2 權重
│   ├── FFN1_out_original.dat   # FFN Layer 1 Golden Reference
│   ├── FFN2_out_original.dat   # FFN Layer 2 Golden Reference
│   ├── FFN_layer1_*.dat        # FPGA 測試資料
│   └── FFN_layer2_*.dat        # FPGA 測試資料
├── *.md                         # 分析文檔
│   ├── FFN_專案分析.md         # 完整專案分析
│   ├── AXI_介面詳細分析.md     # AXI4-Stream 介面分析
│   ├── PE_處理元件詳細分析.md  # PE 處理元件詳細分析
│   └── SRAM_IP_使用指南.md     # SRAM IP 使用指南
└── SOC_FFN_lab.pdf             # 專案文件
```

---

## 🏗️ 系統架構

### 整體資料流

```
ARM CPU (PS端)
    ↓ (AXI DMA)
AXI Input Stream → 指令解析器 → 控制 FSM
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
Input SRAM        Kernel SRAM         Bias SRAM
(輸入特徵)         (權重矩陣)          (偏置值)
512×64-bit        8×512×64-bit       512×32-bit
    ↓                   ↓                   ↓
    └───────────────────┼───────────────────┘
                        ↓
                PE Array (8×8 MACs = 64 MACs)
                  7 級流水線設計
                        ↓
                量化模組 (32-bit → 8-bit)
                  7 級流水線（FPGA）
                        ↓
                封裝模組 (8×8-bit → 64-bit)
                        ↓
                Output SRAM
                512×64-bit
                        ↓
                AXI Output Stream
                        ↓
                ARM CPU (結果驗證)
```

### 主狀態機（fsm64.v）

```
狀態轉換：IDLE → FSLD → BASE → DONE → IDLE

- IDLE：等待啟動信號
- FSLD（First Load）：首次載入權重和偏置
- BASE：執行基礎計算（可能多次迭代）
- DONE：完成並返回 IDLE
```

---

## 🔧 核心模組詳解

### 1. PE (Processing Element) - 計算核心

**架構**：8 個 PE，每個 PE 執行 8 次乘法累加（總共 64 MAC）

**pe_8e.v - 7 級流水線設計**：
```
Stage 0: 輸入暫存
Stage 1: 8 次乘法 + 激活加法樹（第一層）
Stage 2: 4 次加法 + 激活加法樹（第二層）
Stage 3: 2 次加法 + 激活加法樹（第三層）
Stage 4: 最終加法 + 激活累加器
Stage 5: MAC 累加器
Stage 6: 加偏置
Stage 7: 輸出暫存
```

**關鍵特性**：
- ✅ 樹狀加法器減少關鍵路徑延遲
- ✅ 雙路徑計算（MAC 結果 + 激活總和）
- ✅ 累加器支援可變長度累加
- ✅ 偏置延遲鏈確保正確對齊

**資料型態**：
- 輸入：8-bit unsigned
- 權重：8-bit unsigned
- 偏置：32-bit signed
- MAC 結果：32-bit signed
- 輸出（量化後）：8-bit unsigned

### 2. 量化模組（quan2uint8.v）

**量化公式**：
```
output = saturate(round((MAC_result - act_sum × z_weight) × m0_scale >> (31 + index)))
```

**流水線**：7 級（FPGA）/ 6 級（ASIC）

**參數**：
- `m0_scale`：縮放因子（32-bit）
- `index`：右移位數（8-bit）
- `z_of_weight`：權重零點（16-bit）
- `z3`：激活零點（8-bit）

### 3. AXI4-Stream 介面

**輸入通道（S_AXIS_MM2S）**：
- `TVALID`：有效信號
- `TREADY`：就緒信號
- `TDATA`：64-bit 資料
- `TKEEP`：8-bit 位元組有效
- `TLAST`：最後信號

**輸出通道（M_AXIS_S2MM）**：
- 相同信號定義
- 透過 FIFO 和 Register Slice 緩衝

**握手協議**：
- TVALID 和 TREADY 必須同時為 1 才能完成傳輸
- 支援背壓（Backpressure）處理

### 4. 指令解析器（get_ins.v）

**指令頭識別**：
- `INST_HEAD = 0xefef123abbeeff22`：指令模式
- `DATA_HEAD = 0xefef6543dadaff11`：資料模式

**配置參數（16 個 64-bit 字）**：
- CFG_0：啟動信號
- CFG_1：量化參數（m0_scale, index, z_of_weight, z3）
- CFG_2：Token 數和輸入大小
- CFG_3：偏置載入大小
- CFG_4：權重長度和 Tile 配置
- CFG_5~13：保留
- CFG_14~15：輸出格式參數

---

## 📊 FFN Layer 配置

### FFN Layer 1
- **輸入**：8 tokens × 64 features = 512 個 64-bit 值
- **權重**：64 × 2048 = 131,072 個 64-bit 值
- **偏置**：2048 個 32-bit 值
- **輸出**：8 tokens × 256 features = 2,048 個 64-bit 值

**Testbench 配置**：
```systemverilog
cfg_if_token_nums_sub1       = 3'd7    // 8 tokens
cfg_if_totalsize_sub1        = 9'd511  // 512 words
cfg_bias_once_load_size_sub1 = 9'd511  // 512 words
cfg_ker_length_sub1          = 9'd63   // 64 words per column
cfg_ker_tile_readnums_sub1   = 2'd3    // 4 tiles
cfg_ker_tile_size_sub1       = 6'd63   // 64 words per tile
```

### FFN Layer 2
- **輸入**：2 tokens × 256 features = 512 個 64-bit 值
- **權重**：256 × 512 = 131,072 個 64-bit 值
- **偏置**：512 個 32-bit 值
- **輸出**：2 tokens × 64 features = 128 個 64-bit 值

**Testbench 配置**：
```systemverilog
cfg_if_token_nums_sub1       = 3'd1    // 2 tokens
cfg_if_totalsize_sub1        = 9'd511  // 512 words
cfg_bias_once_load_size_sub1 = 9'd511  // 512 words
cfg_ker_length_sub1          = 9'd255  // 256 words per column
cfg_ker_tile_readnums_sub1   = 2'd0    // 1 tile
cfg_ker_tile_size_sub1       = 6'd63   // 64 words per tile
```

---

## 💻 軟體介面（C 程式碼）

### 主要函式

**1. AXI_DMA_Init()**
- 初始化 AXI DMA 控制器
- 設定中斷處理器
- 配置 DMA 通道

**2. FFN_first_layer() / FFN_second_layer()**
- 執行 FFN 層計算
- 資料傳輸流程：
  1. 傳送指令頭（INST_HEAD × 2）
  2. 傳送配置參數（CFG[16]）
  3. 傳送資料頭（DATA_HEAD × 2）
  4. 傳送輸入特徵
  5. 傳送偏置值
  6. 傳送權重矩陣（分批傳送）
  7. 接收輸出結果

**3. COMP_ARRAY_DATA()**
- 比對 FPGA 輸出與 Golden Reference
- 逐個元素檢查
- 輸出錯誤統計

### 配置參數範例（FFN Layer 1）
```c
u64 CFG[16] = {
    0xffff000000000000,  // CFG_0: 啟動信號
    0x1484121114001b00,  // CFG_1: 量化參數
    0x0000000000000fff,  // CFG_2: Token 數和輸入大小
    0x00000000000001ff,  // CFG_3: 偏置載入大小
    0x000000000001fe3f,  // CFG_4: 權重長度和 Tile 配置
    // ... CFG_5 ~ CFG_13 保留
    0xff8000e000000000,  // CFG_14: 輸出格式參數
    0x1f80010000000000   // CFG_15: 輸出格式參數
};
```

---

## 🧪 測試與驗證

### Testbench (FFN_tb.sv)

**支援模式**：
- ✅ FFN1 配置（8 tokens × 64 → 8 tokens × 2048）
- ✅ FFN2 配置（2 tokens × 256 → 2 tokens × 512）
- ✅ RTL 模擬（200 MHz）
- ✅ Gate-level 模擬（需 SDF 檔案）
- ✅ Vivado 模擬（100 MHz）

**測試流程**：
1. 從 `.dat` 檔案載入測試資料
2. 生成 AXI4-Stream 交易
3. 傳送指令和資料
4. 接收輸出結果
5. 自動比對與 Golden Reference
6. 輸出波形（FSDB）

**關鍵測試點**：
- ✅ AXI 握手協議
- ✅ 背壓處理
- ✅ FIFO 溢出測試
- ✅ 資料正確性驗證

---

## ⚡ 效能特性

### 計算能力
- **吞吐量**：64 MACs/cycle
- **理論效能**：
  - 200 MHz：12.8 GMAC/s
  - 100 MHz：6.4 GMAC/s
- **實際效能**：約理論值的 70-90%（考慮資料載入開銷）

### 延遲分析
- **PE 流水線**：7 cycles
- **量化流水線**：7 cycles（FPGA）
- **單個輸出神經元**：約 38 cycles
- **FFN Layer 1 完整計算**：約 140,000 cycles ≈ 0.7 ms @ 200MHz

### 記憶體頻寬
- **輸入**：64 bits/cycle
- **權重**：512 bits/cycle（8 × 64-bit 並行讀取）
- **偏置**：256 bits/cycle（8 × 32-bit）
- **輸出**：64 bits/cycle

### 資源使用（估計 - Xilinx Zynq）
- **DSP48E2**：約 20 個
  - PE 乘法器：16 個（SIMD 模式）
  - 量化乘法器：3 個
- **BRAM**：約 12 個 BRAM36（42 KB）
  - Input SRAM：4 KB
  - Kernel SRAM：32 KB
  - Bias SRAM：2 KB
  - Output SRAM：4 KB
- **LUT/FF**：
  - 控制邏輯：約 5,000 LUT
  - 流水線暫存器：約 3,000 FF

---

## 🎯 設計特色與優化

### 1. 流水線設計
- ✅ 7 級 PE 流水線提高吞吐量
- ✅ AXI 介面使用 register slice 改善時序
- ✅ 樹狀加法器減少關鍵路徑

### 2. 量化策略
- ✅ INT8 運算降低功耗和面積
- ✅ 可配置量化參數支援不同模型
- ✅ 分離追蹤 MAC 結果和激活總和

### 3. 記憶體優化
- ✅ 雙埠 SRAM 支援同時讀寫
- ✅ 8 個並行讀取埠匹配 PE 陣列
- ✅ 使用 FPGA BRAM 原語

### 4. 資料流優化
- ✅ AXI4-Stream 串流介面
- ✅ 內部 FIFO 緩衝和流量控制
- ✅ 指令/資料區分機制
- ✅ 級聯設計節省輸入匯流排

---

## 📝 關鍵發現與觀察

### 優點
1. ✅ **完整的硬體加速器設計**：從 AXI 介面到 PE 陣列，架構完整
2. ✅ **良好的模組化設計**：各模組職責清晰，易於維護
3. ✅ **高效的流水線設計**：7 級流水線平衡延遲和吞吐量
4. ✅ **完整的軟體支援**：C 程式碼驅動和測試完整
5. ✅ **詳細的文檔**：三份詳細的分析文檔（專案、AXI、PE）
6. ✅ **可配置性**：支援不同的 FFN 層配置
7. ✅ **驗證完整**：SystemVerilog testbench 和 Golden Reference

### 設計亮點
1. **雙路徑計算**：MAC 結果和激活總和並行計算，支援量化公式
2. **樹狀加法器**：減少關鍵路徑延遲，提高時脈頻率
3. **級聯設計**：8 個 PE 共享輸入特徵，節省匯流排
4. **偏置延遲鏈**：確保偏置正確對齊到對應的 MAC 結果
5. **背壓處理**：完整的 AXI 流量控制機制

### 潛在改進方向
1. **增加並行度**：可以增加 PE 數量（8 → 16）或 PE 列數（1 → 2）
2. **提高時脈頻率**：增加流水線級數（7 → 10）
3. **權重壓縮**：支援 4-bit 量化或稀疏性
4. **功耗優化**：時脈門控、DVFS、零跳過
5. **記憶體頻寬優化**：增加片上緩衝或使用更高效的記憶體架構

---

## 🔗 與 I-ViT 專案的關聯

### 相似之處
1. **量化策略**：都使用 INT8 量化
2. **流水線設計**：都採用多級流水線提高吞吐量
3. **模組化設計**：都有清晰的模組劃分

### 差異之處
1. **應用場景**：
   - FFN：Feed-Forward Network（全連接層）
   - I-ViT：Vision Transformer（包含 Attention、LayerNorm、GELU、Softmax）
2. **計算模式**：
   - FFN：矩陣乘法（MAC 密集）
   - I-ViT：多種非線性運算（Softmax、GELU、LayerNorm）
3. **介面**：
   - FFN：AXI4-Stream（Zynq FPGA）
   - I-ViT：可能需要不同的介面設計

### 可借鑒之處
1. ✅ **AXI4-Stream 介面設計**：可用於 I-ViT 的資料傳輸
2. ✅ **量化模組設計**：quan2uint8.v 可參考用於 I-ViT
3. ✅ **流水線設計經驗**：7 級流水線的平衡策略
4. ✅ **Testbench 架構**：SystemVerilog testbench 的組織方式
5. ✅ **C 程式碼驅動**：ARM 端的驅動程式架構

---

## 📚 文檔完整性

### 已有文檔
1. ✅ **FFN_專案分析.md**：完整的專案分析（架構、模組、資料格式）
2. ✅ **AXI_介面詳細分析.md**：AXI4-Stream 介面完整分析
3. ✅ **PE_處理元件詳細分析.md**：PE 處理元件詳細分析（7 級流水線）
4. ✅ **SRAM_IP_使用指南.md**：SRAM IP 使用指南
5. ✅ **SOC_FFN_lab.pdf**：專案文件

### 文檔品質
- ✅ 詳細的架構圖
- ✅ 完整的信號說明
- ✅ 清晰的時序圖
- ✅ 實用的範例程式碼
- ✅ 常見問題與解決方案

---

## 🎓 總結

FFN 專案是一個**成熟、完整的 FPGA 神經網路加速器設計**，展現了以下特點：

1. **專業的硬體設計**：從 AXI 介面到 PE 陣列，每個模組都經過精心設計
2. **高效的計算架構**：64 MAC 並行計算，7 級流水線設計
3. **完整的驗證環境**：SystemVerilog testbench 和 C 程式碼驅動
4. **詳細的文檔**：三份詳細的分析文檔，易於理解和維護
5. **良好的可擴展性**：模組化設計，易於修改和擴展

這個專案可以作為 **I-ViT 專案的重要參考**，特別是在 AXI 介面設計、量化模組、流水線設計和驗證方法學方面。

---

**查看完成日期**：2026-05-23  
**查看人員**：Kiro AI Assistant  
**專案狀態**：✅ 完整查看完成
