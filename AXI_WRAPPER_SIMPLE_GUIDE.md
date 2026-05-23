# AXI Wrapper 簡易說明 - 資料拆解與 Testbench

**日期**: 2026-05-23  
**狀態**: ✅ 已完成並可測試

---

## 📊 Nonlinear 模組介面資訊

### 輸入介面
```verilog
input in_valid,                      // 輸入有效信號
input signed [15:0] in_data,         // 輸入資料（16-bit）
input in_last,                       // 最後一筆資料標記
output in_ready,                     // 準備接收信號
```

### 輸出介面
```verilog
output out_valid,                    // 輸出有效信號
output signed [7:0] out_data,        // 輸出資料（8-bit）
output out_last,                     // 最後一筆資料標記
```

### 關鍵特性
- **輸入寬度**：16-bit（有符號整數）
- **輸出寬度**：8-bit（有符號整數）
- **處理方式**：串行處理（一次處理 1 個資料）
- **並行通道數**：1

---

## 🔄 Step 2: 資料拆解（Bank Splitting）

### AXI 64-bit → Nonlinear 16-bit

**問題**：AXI DMA 傳輸是 64-bit，但 Nonlinear 模組只需要 16-bit

**解決方案**：在 Wrapper 中拆解資料

```verilog
// 檔案：ivit_nonlinear_axi_wrapper.v（第 280 行左右）

// 64-bit AXI TDATA 的結構
// ┌─────────┬─────────┬─────────┬─────────┐
// │ [63:48] │ [47:32] │ [31:16] │ [15:0]  │  64-bit
// └─────────┴─────────┴─────────┴────┬────┘
//    未使用    未使用    未使用       │
//                                     ↓
//                              core_in_data (16-bit)

// 資料拆解邏輯
assign core_in_data = (IN_WIDTH == 16) ? isif_data_dout[15:0] : 
                      (IN_WIDTH == 8)  ? isif_data_dout[7:0] : 
                      isif_data_dout[IN_WIDTH-1:0];
```

### 目前的設計

**方式**：只使用 64-bit 中的低 16-bit（[15:0]）

**優點**：
- ✅ 簡單直接
- ✅ 容易除錯
- ✅ 已經可以工作

**缺點**：
- ⚠️ 頻寬利用率只有 25%（64-bit 中只用了 16-bit）
- ⚠️ 需要 4 倍的 AXI 傳輸次數

### 資料傳輸示例

假設要傳送 4 個 16-bit 資料：`[0x1234, 0x5678, 0x9ABC, 0xDEF0]`

**目前方式**（需要 4 次 AXI 傳輸）：
```
AXI 傳輸 1: 0x0000_0000_0000_1234  ← 只用 [15:0]
AXI 傳輸 2: 0x0000_0000_0000_5678  ← 只用 [15:0]
AXI 傳輸 3: 0x0000_0000_0000_9ABC  ← 只用 [15:0]
AXI 傳輸 4: 0x0000_0000_0000_DEF0  ← 只用 [15:0]
```

**未來優化方式**（只需 1 次 AXI 傳輸）：
```
AXI 傳輸 1: 0xDEF0_9ABC_5678_1234  ← 用全部 64-bit
             └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘
               4     3     2     1
```
但這需要在 Wrapper 中添加狀態機來依序送入 4 個資料。

---

## 📦 輸出資料封裝（8-bit → 64-bit）

### Nonlinear 8-bit → AXI 64-bit

**問題**：Nonlinear 模組輸出是 8-bit，但 AXI DMA 需要 64-bit

**解決方案**：在 Wrapper 中封裝資料

```verilog
// 檔案：ivit_nonlinear_axi_wrapper.v（第 310-340 行）

// 8 個 8-bit 輸出封裝成 1 個 64-bit
reg [7:0] out_buffer [0:7];  // 緩衝 8 個輸出
reg [2:0] out_cnt;           // 計數器（0~7）

// 收集 8 個輸出
always @(posedge clk) begin
    if (core_out_valid && osif_full_n) begin
        out_buffer[out_cnt] <= core_out_data;
        if (out_cnt == 3'd7 || core_out_last) begin
            out_cnt <= 3'd0;
            out_packing <= 1'b1;  // 觸發封裝
        end else begin
            out_cnt <= out_cnt + 1'b1;
        end
    end
end

// 封裝成 64-bit
assign osif_data_din = {out_buffer[7], out_buffer[6], out_buffer[5], out_buffer[4],
                        out_buffer[3], out_buffer[2], out_buffer[1], out_buffer[0]};
```

### 輸出封裝示例

假設 Nonlinear 輸出 8 個 8-bit 資料：`[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]`

**封裝過程**：
```
收集 8 個輸出：
  out_buffer[0] = 0x12
  out_buffer[1] = 0x34
  out_buffer[2] = 0x56
  out_buffer[3] = 0x78
  out_buffer[4] = 0x9A
  out_buffer[5] = 0xBC
  out_buffer[6] = 0xDE
  out_buffer[7] = 0xF0

封裝成 64-bit：
  AXI TDATA = 0xF0DE_BC9A_7856_3412
              └─┬─┘└─┬─┘└─┬─┘└─┬─┘
                7   6   5   4   3   2   1   0
```

---

## 🧪 Step 3: Testbench 模擬 DMA

### Testbench 架構

**檔案位置**：`C:\Users\Public\I-ViT\nonlinear_verification\tb\tb_nonlinear_axi_wrapper.sv`

### 核心功能

#### 1. 測試資料陣列

```systemverilog
// 輸入資料陣列（16-bit）
reg [15:0] input_data [0:TEST_DATA_COUNT-1];

// 輸出資料陣列（8-bit）
reg [7:0] output_data [0:TEST_DATA_COUNT-1];

// 初始化測試資料
initial begin
    for (i = 0; i < TEST_DATA_COUNT; i = i + 1) begin
        input_data[i] = $signed(i - 128);  // 簡單的測試模式
    end
end
```

#### 2. AXI 握手邏輯

```systemverilog
// Task: 送出一筆 AXI 資料
task axi_send_data;
    input [63:0] data;
    input        last;
    begin
        @(posedge clk);
        S_AXIS_MM2S_TVALID <= 1'b1;      // 拉高 TVALID
        S_AXIS_MM2S_TDATA <= data;       // 送出資料
        S_AXIS_MM2S_TKEEP <= 8'hFF;      // 所有位元組有效
        S_AXIS_MM2S_TLAST <= last;       // 最後一筆標記
        
        // 等待握手（TREADY = 1）
        wait(S_AXIS_MM2S_TREADY);
        @(posedge clk);
        
        // 拉低 TVALID
        S_AXIS_MM2S_TVALID <= 1'b0;
        S_AXIS_MM2S_TLAST <= 1'b0;
    end
endtask
```

**關鍵點**：
- ✅ 當 `TREADY = 1` 時，才拉高 `TVALID` 並送資料
- ✅ 握手完成後（`TVALID && TREADY`），資料傳輸成功
- ✅ 使用 `TLAST` 標記最後一筆資料

#### 3. 送資料迴圈

```systemverilog
// Task: 送出所有輸入資料
task send_input_data;
    input integer count;
    integer m;
    reg [63:0] packed_data;
    begin
        // 將 4 個 16-bit 輸入封裝成 64-bit
        for (m = 0; m < count; m = m + 4) begin
            packed_data = {input_data[m+3], input_data[m+2], 
                          input_data[m+1], input_data[m]};
            
            if (m + 4 >= count)
                axi_send_data(packed_data, 1'b1);  // 最後一筆
            else
                axi_send_data(packed_data, 1'b0);  // 非最後一筆
        end
    end
endtask
```

#### 4. 接收輸出資料

```systemverilog
// 自動收集輸出資料
always @(posedge clk) begin
    if (M_AXIS_S2MM_TVALID && M_AXIS_S2MM_TREADY) begin
        // 解包 64-bit 成 8 個 8-bit
        output_data[output_count+0] <= M_AXIS_S2MM_TDATA[7:0];
        output_data[output_count+1] <= M_AXIS_S2MM_TDATA[15:8];
        output_data[output_count+2] <= M_AXIS_S2MM_TDATA[23:16];
        output_data[output_count+3] <= M_AXIS_S2MM_TDATA[31:24];
        output_data[output_count+4] <= M_AXIS_S2MM_TDATA[39:32];
        output_data[output_count+5] <= M_AXIS_S2MM_TDATA[47:40];
        output_data[output_count+6] <= M_AXIS_S2MM_TDATA[55:48];
        output_data[output_count+7] <= M_AXIS_S2MM_TDATA[63:56];
        output_count <= output_count + 8;
    end
end
```

---

## 🎯 測試流程

### 完整的測試序列

```systemverilog
initial begin
    // 1. Reset
    resetn = 0;
    #100;
    resetn = 1;
    #50;
    
    // 2. 送指令頭（INST_HEAD × 2）
    send_instruction_header();
    
    // 3. 送配置參數（CFG_0 ~ CFG_15）
    send_configuration(TEST_MODE, TEST_DATA_COUNT);
    
    // 4. 送資料頭（DATA_HEAD × 2）
    send_data_header();
    
    // 5. 送輸入資料
    send_input_data(TEST_DATA_COUNT);
    
    // 6. 等待處理完成
    #10000;
    
    // 7. 檢查結果
    if (output_count == TEST_DATA_COUNT) begin
        $display("PASS: Output count matches");
    end else begin
        $display("FAIL: Output count mismatch");
    end
    
    $finish;
end
```

---

## 🚀 如何執行測試

### 方法 1：使用 ModelSim（推薦）

```bash
# 1. 進入驗證目錄
cd C:\Users\Public\I-ViT\nonlinear_verification

# 2. 啟動 ModelSim 仿真
vsim -do run_axi_wrapper_sim.tcl

# 3. 查看波形
# ModelSim 會自動添加所有關鍵信號到波形視窗
```

### 方法 2：修改測試參數

編輯 `tb_nonlinear_axi_wrapper.sv`：

```systemverilog
// 測試 GELU 模式
parameter TEST_MODE = TEST_MODE_GELU;
parameter TEST_DATA_COUNT = 768;

// 測試 LayerNorm 模式
parameter TEST_MODE = TEST_MODE_LAYERNORM;
parameter TEST_DATA_COUNT = 192;

// 測試 Softmax 模式
parameter TEST_MODE = TEST_MODE_SOFTMAX;
parameter TEST_DATA_COUNT = 197;
```

---

## 📊 波形分析重點

### 關鍵信號

#### AXI 輸入握手
```
S_AXIS_MM2S_TVALID  ─┐     ┌─────┐     ┌─
S_AXIS_MM2S_TREADY  ─────┐ │     │ ┌───┘
S_AXIS_MM2S_TDATA   ──X──┤ DATA1 ├─┤ DATA2
                          └───┬───┘
                              ↓
                          握手成功
```

#### 核心模組資料流
```
core_in_valid   ─┐     ┌─────┐     ┌─
core_in_ready   ─────┐ │     │ ┌───┘
core_in_data    ──X──┤ 16bit ├─┤ 16bit
                      └───┬───┘
                          ↓
                      資料進入核心
```

#### AXI 輸出握手
```
M_AXIS_S2MM_TVALID  ─┐     ┌─────┐     ┌─
M_AXIS_S2MM_TREADY  ─────┐ │     │ ┌───┘
M_AXIS_S2MM_TDATA   ──X──┤ DATA1 ├─┤ DATA2
                          └───┬───┘
                              ↓
                          握手成功
```

---

## 💡 常見問題

### Q1: 為什麼 64-bit 只用 16-bit？

**答**：因為 Nonlinear 模組是串行處理，一次只能接收 1 個 16-bit 資料。

**未來優化**：可以添加狀態機，依序送入 4 個 16-bit 資料，提升頻寬利用率。

### Q2: 如何提升頻寬利用率？

**方案 1**：在 Wrapper 中添加狀態機
```verilog
// 狀態機依序送入 4 個 16-bit
case (bank_cnt)
    2'd0: core_in_data <= isif_data_dout[15:0];
    2'd1: core_in_data <= isif_data_dout[31:16];
    2'd2: core_in_data <= isif_data_dout[47:32];
    2'd3: core_in_data <= isif_data_dout[63:48];
endcase
```

**方案 2**：修改 Nonlinear 模組支援並行輸入
```verilog
// 修改為 4 個並行輸入
input signed [15:0] in_data_0,
input signed [15:0] in_data_1,
input signed [15:0] in_data_2,
input signed [15:0] in_data_3,
```

### Q3: Testbench 如何模擬 DMA？

**答**：Testbench 使用 `task` 來模擬 DMA 的行為：
1. 宣告測試資料陣列
2. 用迴圈依序送出資料
3. 檢查 `TREADY` 信號，確保握手成功
4. 自動收集輸出資料

---

## ✅ 總結

### 已完成的工作

✅ **Step 1: AXI Wrapper**
- 標準 AXI4-Stream 介面
- 包裝 Nonlinear 模組

✅ **Step 2: 資料拆解**
- 64-bit → 16-bit（輸入）
- 8-bit → 64-bit（輸出）

✅ **Step 3: Testbench**
- 測試資料陣列
- AXI 握手邏輯
- 自動送資料迴圈
- 輸出資料收集

### 可以開始測試了！

```bash
cd C:\Users\Public\I-ViT\nonlinear_verification
vsim -do run_axi_wrapper_sim.tcl
```

---

**文檔版本**: 1.0  
**最後更新**: 2026-05-23  
**作者**: Kiro AI Assistant  
**狀態**: ✅ 已完成並可測試
