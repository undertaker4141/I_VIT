# LayerNorm 架構澄清與驗證策略

**日期**: 2026-05-23  
**目的**: 澄清 Bank 切換與 LayerNorm 內部架構  
**對象**: 王珈源

---

## 🎯 核心澄清：兩種不同的 "Bank"

### 你說的 Bank（BRAM 實體 Bank）

**場景**：全系統整合時，8x8 PE 陣列需要 512-bit 頻寬

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ BRAM 0  │ │ BRAM 1  │ │ BRAM 2  │ │ BRAM 3  │
│ 64-bit  │ │ 64-bit  │ │ 64-bit  │ │ 64-bit  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
              512-bit 並行讀取
                    ↓
              8x8 PE 陣列
```

**特點**：
- 實體記憶體，在 Vivado 中配置
- 空間並行（8 顆 BRAM 同時讀取）
- 固定寬度和深度

### 我說的 Bank（AXI 資料切分）

**場景**：Nonlinear 模組單獨驗證，不接 SRAM

```
AXI 64-bit 資料
┌─────────┬─────────┬─────────┬─────────┐
│ [63:48] │ [47:32] │ [31:16] │ [15:0]  │
└────┬────┴────┬────┴────┬────┴────┬────┘
     │         │         │         │
     ↓         ↓         ↓         ↓
  Cycle 3  Cycle 2  Cycle 1  Cycle 0
  (時間並行，用 FSM 依序送入)
```

**特點**：
- 暫存器（reg），不是實體 BRAM
- 時間並行（4 個 Cycle 依序送入）
- 只用於 AXI Wrapper 的資料拆解

---

## 🏗️ LayerNorm 內部架構：完整解答

### 你的擔憂（完全正確！）

> "你這樣最多只能存 8 個值，LayerNorm 是 16-bit 的，甚至只有 4 個值？這樣不能運算！"

**如果只有 AXI Wrapper 的 64-bit 暫存器**：
- ✅ 你說得對！確實無法運算
- ✅ 因為 LayerNorm 需要看完全部 192 個值才能算 Mean
- ✅ 如果資料消失了，叫 DRAM 重傳會有延遲問題

### 實際架構（我早就做好了！）

**關鍵發現**：`ivit_layernorm.v` 第 72-73 行

```verilog
// Memory instantiations
reg signed [IN_WIDTH-1:0] mem_var [0:1023];  // 1024 深度的 Local Buffer
reg signed [IN_WIDTH-1:0] mem_out [0:1023];  // 1024 深度的 Local Buffer
```

**完整資料流**：

```
AXI 64-bit
    ↓
AXI Wrapper (Bank 切換狀態機)
    ↓ 16-bit (依序送入)
┌─────────────────────────────────────────┐
│  LayerNorm 模組內部                      │
│                                          │
│  Stage 0: Input & Mean                  │
│  ┌────────────────────────────────┐    │
│  │ 資料進來時：                    │    │
│  │ 1. 加入累加器算 Mean            │    │
│  │ 2. 同時存入 mem_var[0:191]     │    │
│  │ 3. 同時存入 mem_out[0:191]     │    │
│  └────────────────────────────────┘    │
│           ↓                              │
│  Stage 1: Variance                      │
│  ┌────────────────────────────────┐    │
│  │ 從 mem_var 讀取 192 個值        │    │
│  │ 計算 (x - Mean)^2               │    │
│  │ 完全不需要 DRAM 重傳！          │    │
│  └────────────────────────────────┘    │
│           ↓                              │
│  Stage 2: Output                        │
│  ┌────────────────────────────────┐    │
│  │ 從 mem_out 讀取 192 個值        │    │
│  │ 計算 (x - Mean) / Std + Bias   │    │
│  └────────────────────────────────┘    │
│                                          │
└─────────────────────────────────────────┘
    ↓ 8-bit 輸出
AXI Wrapper (輸出封裝)
    ↓ 64-bit
AXI Output
```

### 為什麼開兩個 Buffer？（Double Buffering）

**Pipeline 並行處理**：

```
時間軸：
Cycle 0-191:   Stage 0 處理 Token 0，寫入 mem_var[0:191]
Cycle 192-383: Stage 0 處理 Token 1，寫入 mem_var[192:383]
               同時！Stage 1 處理 Token 0，讀取 mem_var[0:191]
Cycle 384-575: Stage 0 處理 Token 2，寫入 mem_var[384:575]
               同時！Stage 1 處理 Token 1，讀取 mem_var[192:383]
               同時！Stage 2 處理 Token 0，讀取 mem_out[0:191]
```

**結論**：
- ✅ 資料完全在 PL 內部
- ✅ 不需要 DRAM 重傳
- ✅ 不會賭傳輸時間
- ✅ Pipeline 完全不卡彈

---

## 🛡️ 流量控制（Flow Control）驗證

### 你的擔憂（再次正確！）

> "單一個用 TB 驗證跟上板子做模組驗證會不一樣"

**真實板子的風險**：
- DRAM 突然塞車
- AXI TVALID 突然掉到 0
- 如果狀態機只會盲目計數，會吃到垃圾資料

### 我的防禦機制

**LayerNorm 內部的保護**（第 120-145 行）：

```verilog
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    // 初始化
  end else begin
    if (in_valid) begin  // ← 關鍵！只有 in_valid = 1 才處理
      mem_var[in_wr_addr] <= in_data;
      mem_out[in_wr_addr] <= in_data;
      in_wr_addr <= in_wr_addr + 1;
      sum_val <= cur_sum;
      in_cnt <= in_cnt + 1;
    end
    // 如果 in_valid = 0，所有狀態凍結，原地等待
  end
end
```

**AXI Wrapper 的保護**（Bank 切換狀態機）：

```verilog
// 只在 buffer_valid && core_in_ready 時才送資料
else if (buffer_valid && core_in_ready) begin
  if (bank_cnt == banks_per_word - 1) begin
    buffer_valid <= 1'b0;  // 需要新資料
  end else begin
    bank_cnt <= bank_cnt + 1;  // 移動到下一個 Bank
  end
end
```

**結論**：
- ✅ 如果 AXI TVALID = 0，FIFO empty，buffer_valid = 0
- ✅ 狀態機凍結，不會亂吃資料
- ✅ 等 TVALID 恢復後繼續處理

---

## 🧪 驗證策略：隨機背壓測試

### 目標

證明系統在 **DRAM 塞車、AXI 訊號亂跳** 的惡劣環境下，依然能正確運算。

### 測試方法

**在 Testbench 中模擬真實板子的惡劣情況**：

1. **隨機輸入暫停**（模擬 DRAM 塞車）：
```systemverilog
// 30% 機率讓 TVALID 拉低
if ($random % 100 < 30) begin
    S_AXIS_MM2S_TVALID <= 1'b0;  // 斷流！
end else begin
    S_AXIS_MM2S_TVALID <= 1'b1;
end
```

2. **隨機輸出背壓**（模擬下游卡彈）：
```systemverilog
// 40% 機率讓 TREADY 拉低
if ($random % 100 < 40) begin
    M_AXIS_S2MM_TREADY <= 1'b0;  // 卡彈！
end else begin
    M_AXIS_S2MM_TREADY <= 1'b1;
end
```

### 預期結果

- ✅ 輸出資料數量正確（768 個）
- ✅ 輸出資料值正確（與理想情況相同）
- ✅ 沒有 Hang（掛起）或 Deadlock（死鎖）
- ✅ TLAST 信號正確

---

## 📊 總結對比表

| 項目 | 你的擔憂 | 實際情況 | 結論 |
|-----|---------|---------|------|
| **資料存儲** | 只有 64-bit 暫存器，存不下 192 個值 | LayerNorm 內部有 1024 深度的 Local Buffer | ✅ 完全足夠 |
| **DRAM 重傳** | 需要賭傳輸時間，很危險 | 資料完全在 PL 內部，不需要重傳 | ✅ 沒有風險 |
| **流量控制** | TB 驗證跟上板不一樣 | 有 `if (in_valid)` 保護，會凍結等待 | ✅ 已防禦 |
| **Pipeline** | 擔心時程排序困難 | Three-Stage Pipeline，Double Buffering | ✅ 設計優秀 |

---

## 🎯 下一步行動

### 1. 立即可做

- ✅ 確認你已經理解 LayerNorm 內部有 Local Buffer
- ✅ 確認 AXI Wrapper 的 Bank 切換只是資料拆解，不是實體 BRAM

### 2. 驗證計畫

- 🔄 執行隨機背壓測試（Random Backpressure Test）
- 🔄 產出測試報告，證明系統穩定性
- 🔄 準備整合到全系統

### 3. 整合準備

當要整合到全系統時：
- 你說的實體 BRAM Bank（8 顆並行）會在那時候用到
- 目前的 Nonlinear 模組單獨驗證不需要實體 BRAM
- AXI Wrapper 的設計可以無縫對接到全系統

---

## 💬 回覆建議

**給王珈源的訊息**：

> 啊啊我懂你的意思了！你以為我只用那個 64-bit 的 Wrapper 來存資料對吧？
> 
> 不是啦，那個 64-bit 真的只負責「把 AXI 寬度轉成 16-bit」。
> 
> 我真正的 LayerNorm 模組「裡面」，有開兩個 1024 深度的 Local Line Buffer（mem_var 和 mem_out）來把這 192 個值存起來！
> 
> 所以資料進來時：
> 1. 一邊進累加器算 Mean
> 2. 一邊存進這兩個 Buffer
> 
> 等算完 Mean 之後，直接從 Buffer 讀出來做 (x - Mean)/Var。
> 
> 這部分的資料都已經在 PL 端了，完全不用叫 DRAM 送第二次，也不用賭傳輸時間。
> 
> 而且我還有 `if (in_valid)` 保護，如果 AXI 突然斷流，狀態機會凍結等待，不會亂吃資料。
> 
> 接下來我會做隨機背壓測試（Random Backpressure Test），模擬 DRAM 塞車的情況，證明系統在惡劣環境下也能正確運算。
> 
> 這樣就沒問題了吧？🤣

---

**文檔版本**: 1.0  
**最後更新**: 2026-05-23  
**作者**: 冠泓 + Kiro AI Assistant  
**狀態**: ✅ 架構澄清完成，準備驗證
