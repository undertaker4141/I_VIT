# I-ViT 全整數化硬體加速器 RTL 專題分工規劃

本文件依據「基於 I-ViT 演算法之高效能 DeiT-Tiny 全整數化硬體加速器設計與實現」以及專案之 C-Model 參考與硬體資料流 (Dataflow) 規劃，將硬體實作 (RTL) 切割為四個核心角色的工作，供組內成員分工參考。

---

## 角色一：運算核心與非線性設計 (PE Array & Computation End)
**負責重點**：運算引擎心臟、處理器陣列 (MAC) 與 C-Model 非線性演算法硬體化。

**具體工作內容**：
1. **PE MAC 陣列設計**：
   * 實現 32-way (或其他指定大小) 單一指令多重數據 (SIMD) 陣列。
   * 確保 8-bit 有號整數 (INT8) 相乘後，累加器 (Accumulator) 能正確擴展至 32-bit (INT32) 避免溢位。
2. **重量化單元 (Requant Unit)**：
   * 實作將 32-bit Partial Sum 轉回 8-bit 的縮放與位移邏輯 (Dyadic Scaling & Barrel Shifter)。
3. **非線性模組 (Non-linear Engine)**：
   * **強烈依賴** `nonlinear_cmodel_reference.py` 的邏輯進行撰寫。
   * **LayerNorm**: 實作精準的 `int16` 截斷特性的 Mean 計算去均值、10次牛頓疊代求平分根 (STD)、以及最終的正規化與加上 Bias。
   * **GELU & Softmax**: 實作 `int_exp_shift_kernel` (最大值防溢位、泰勒多項式平移估算、整數長除法求 Sigmoid / 機率分佈)。
4. **模組級驗證**：確保這些運算模組送入單一測試向量時，輸出結果與 C-Model 達到 100% Bit-Exact (完全一致)。

---

## 角色二：片內記憶體與資料搬運 (SRAM & Data-path Architecture)
**負責重點**：BRAM 規劃、記憶體讀寫控制器、位址映射 (Address Mapping) 與緩衝策略。

**具體工作內容**：
1. **SRAM Bank 實體化與介面封裝**：
   * 建立並封裝 432 KB 的 BRAM 架構，切割為：IFM (40KB)、Weight (152KB, 需分 Bank)、OFM (80KB, 切 Bank 0/1 供 Ping-Pong)、Param (8KB)、Residual (152KB)。
2. **Ping-Pong Buffer 連鎖設計**：
   * 設計 Weight SRAM 與 OFM SRAM 的雙緩衝區 (Ping-Pong) 切換邏輯，確保計算與外部載入的 Pipeline 不會斷掉。
3. **資料對齊與 Tiling (切塊) 處理**：
   * 負責計算硬體 trace 中的位址邏輯，特別是 **FFN 的 Sequence Tiling (序列切塊)**，解決 768 通道膨脹的記憶體存取策略。
   * 負責 Classification Head 階段的 **Output Channel Tiling**。
4. **總線與介面整合**：
   * 為運算單元準備好讀取/寫入介面。接收來自 FSM 的控制指令並轉譯為實際的 BRAM Read/Write Addresses。

---

## 角色三：頂層系統控制與狀態機 (Global System FSM)
**負責重點**：大腦中樞，控制系統在 Patch Embedding、MHA、FFN 與 Classification 層之間的任務轉換與管線調度。

**具體工作內容**：
1. **全域狀態機 (Global Finite State Machine)**：
   * 根據 `硬體trace.md` 裡的 Dataflow，實作主控制器，指揮目前系統處於網路的哪一階段 (Phase 1 QKV $\rightarrow$ Phase 2 Attention $\rightarrow$ Phase 3 FFN ...等)。
2. **控制訊號分發 (Control Signals Generation)**：
   * 生成 PE 的 Enable / Reset 訊號 (決定何時開始內積、何時清空重置累加器)。
   * 向 Requant Unit 送出「模式切換訊號」(指示現在是 Linear/Conv 模式還是 Attention Score 模式，並載入對應的量化參數)。
3. **管線延遲管理 (Pipeline Bubble & Latency)**：
   * 精確計算並處理各個模組之間的 Clock Latency (例如 Softmax 和 LayerNorm 等非線性運算所需要的等待週期)，避免資料流衝突。
4. **硬體中斷與握手 (Interrupt & Handshake)**：
   * 紀錄層數 (如 12 層 Transformer block)，在每層結束或整張圖片運算完成時，送出中斷訊號給 CPU / 外界。

---

## 角色四：系統整合與 FPGA 驗證 (System Integration & FPGA Prototyping)
**負責重點**：統籌拼裝、AXI 通訊、合成 (Synthesis)、佈局佈線與上板驗證。

**具體工作內容**：
1. **Top-Level Integration (頂層接線)**：
   * 將角色一的 PE/Non-linear、角色二的 BRAM 與角色三的 FSM 在頂層完整實體化並正確接線。
2. **外部通訊介面 (Interfacing)**：
   * 開發對外通訊介面 (通常為 AXI4 / AXI-Stream Protocol)，負責與 FPGA PS 端 (如 Zynq CPU) 或 Host 端進行 DRAM 資料交握與 DMA 傳輸。
3. **Testbench 與 Bit-Exact 全系統驗證**：
   * 撰寫完整的 Top-Level 測試台 (Testbench)。
   * 從 Python/TVM 生出完整的 Input Feature & Weight 測試檔案 (Test Vectors)，餵入 ModelSim/Vivado 模擬，確保全系統輸出與 TVM/C-Model **一模一樣**。
4. **FPGA 燒錄與除錯 (Implementation & On-board Debugging)**：
   * 負責 Vivado 合成 (Synthesis) 與實作 (Implementation)，解鎖 Timing Violation (時序違規)。
   * 配置 FPGA 開發板上的 ILA (Integrated Logic Analyzer)，在上板遭遇非預期行為時，能夠抓取內部波形揪出 Bug。
