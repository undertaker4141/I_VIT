* IFM SRAM 40 KB
* Weight SRAM 152(76,76) KB
* OFM SRAM 80(40,40) KB
* Param SRAM 8(4,4) KB
* Residual SRAM： 152 KB
* 總面積 : 432 KB

# Patch Embedding 與 預處理
* 準備
  * 輸入圖片數據 : 224 x 224 x 3，分五次讀取(224//48 = 4, 224-48x4 = 32,(42,42,42,42,28))，48 x 224 x 3 = 33 K
  * 權重讀取 : 16 x 16 x 3 x (32 x 3) = 72 K (96, 96)，有 192 個 filter，可以完整 load
  * position emdedding 權重讀取 42 組存入 OFM SRAM Bank 1 (INT32，42 x 192 x 4 = 33 K) 
* dataflow
  * IFM 從左側輸入PE陣列 load 32組 patch。Weight 從上方輸入，一次32個filter。每組 patch 總共要 repeat load 6 次。這樣就得到一組 token。一次可以得到 32 組 token。(但當次的第二輪會變成只有10組token做計算，有22個PE row浪費)
  * 計算完成是 INT32，所以一組 token 的大小是 1 x 192 x 4，每組的 IFMap 分別是 32、10
  * IFMap 分五次讀取 (可能會卡時間的地方)。
  * PE 算完的 token 存儲到 OFM SRAM，42組 token 蒐集完丟到 Adder 與 OFM SRAM Bank 1 做加法，分成兩份，一份傳入 Residual SRAM 存儲(INT32)，一份送到norm後回傳(INT8)。
  * 載入下一組 embedding 到 OFM SRAM Bank 1
* 頻寬分析
  1. 輸入 (IFM)
     * 設計： 影像總大小為 224 x 224 X 3 = 150.5 KB。切成 5 次讀取。
     * m : 沒有重複讀取， m = 1
     * IFM 頻寬消耗： 150.5 x 1 = 150.5 KB
  2. 權重
     * 設計: 總共 147.5 KB，因為容量足夠，只需載入 1 次。
     * n : 權重常駐，n = 1
     * 頻寬消耗 : 147.5 x 1 = 147.5 KB
  3. Position Embedding
     * 頻寬消耗 : 196 x 192 x 4 = 150.5 KB
  4. 輸出回傳 (OFM)
     * 設計： 為了保留殘差，寫回一份 INT32；為了下一層運算，寫回一份 INT8。
     * INT8 : 196 x 192 x 1 = 37.6 KB
     * 總共 : 37.6 KB
  * 共 486.1 KB/Frame，帶入 220 FPS ，486.1 x 220 = 106,942 KB/s ≈ 104.5 MB/s

# MHA (Multi-Head Self-Attention) 與 注意力機制

* 準備
  * 輸入特徵數據 (IFMap) : 198 x 192。大小為 198 x 192 x 1 Byte = 37.1 KB。(塞入 IFM SRAM 40 KB )。
  * 權重讀取 (Wq, Wk, Wv) : 192 x (192 x 2) = 76 KB。可以分成兩份(2+1)分別放在 bank0 bank1
  * 注意力矩陣極限值 : 單一 Head 的 Attention Map 大小為 198 x 198 x 1 Byte = 38.3 KB。(剛好塞入 OFM SRAM Bank 0 的 40 KB ，不用切塊)。


* dataflow (分四個子階段)
  * **階段 1：Q, K, V 矩陣生成**
    * IFM (38.1 KB) 載入(此階段不洗掉)。
    * Weight 從上方 Ping-Pong 分別載入 QKV 權重。PE 陣列算完得到 INT32，丟到 requant unit 轉成 INT8，得到 Q, K, V 三個 198 x 192 的矩陣 (各 38.1 KB)，放在OFM SRAM。
    * Q 寫入 IFM SRAM(最後動作) 、 K與V 分別寫入Weight SRAM的Bank 0、Bank 1。


  * **階段 2：注意力分數 與 Shiftmax (分 3 個 Head 循序計算)**
    * IFM 載入 Q 時重新安排位置， IFM 從左側載入 Head 1 的 Q (198 x 64 = 12.3 KB)。
    * Weight 從上方載入 Head 1 的 K 轉置 (64 x 198 = 12.3 KB)。
    * PE 陣列進行計算。因為 198 不是 32 的整數倍 (198 = 32 x 6 + 6)，所以 IFM 總共 load 7 次。前 6 次 PE 滿載，第 7 次只有 6 組 token 做計算，(有 26 個 PE row 浪費)。
    * PE 算完 198 x 198 的 INT32 分數，先丟給 requant 執行量化 (得到 198 x 198 x 1 = 39.3 K)，再丟到 softmax 模組計算。暫存 OFM SRAM Bank1。


  * **階段 3：注意力輸出**
    * 從 OFM Bank1 載入剛算好的 Attention Map (198 x 198 x 1 = 39.3 KB)。
    * Weight 從上方載入 Head 1 的 V (198 x 64 x 1 = 12.7 KB)。
    * PE 算完得到 198 x 64 的 INT32 結果，丟給 requant 執行量化，暫存 OFM SRAM (198 x 64 x 1 = 12.6 KB) ，等到所有head計算完後傳入 IFM SRAM。
   
  (階段 2 與 3 會重複執行 3 次，把 3 個 Head 算完)。


  * **階段 4：Output Projection 與 殘差加法**
    * IFM 載入 3 個 Head 拼接(一樣透過放置方式拼接)的完整特徵 198 x 192 (38.1 KB)。
    * Weight 載入 Out Proj 權重 192 x 192 (36.8 KB)。
    * PE 算完後 32 x 192 x 4 = 25 KB，所以每 32 個就換一次ping-pong，丟到 Adder 與 residual SRAM 的對應值 做加法(殘差)，分成兩份，一份存回 residual SRAM (INT32)，一份送到norm後回傳(INT8)。


* 頻寬分析 (單次 Frame) 

1. 輸入讀取 (IFM Read from DRAM)
   * **階段 1：** 僅需讀取最原始的輸入特徵 。
   * `198 x 192 x 1 Byte = 37.1 KB`
   * **階段 2 & 3：** Q 已經在 IFM SRAM，Attn Map 在 OFM Bank 1。**全部片上內循環 (0 KB)**。
   * **階段 4：** V_out 已經被搬進 IFM SRAM，直接使用。**片上讀取 (0 KB)**。
   * **IFM DRAM 總讀取： 37.1 KB**

2. 權重讀取 (Weight Read from DRAM)
   * **階段 1：** 讀取 Wq, Wk, Wv 權重。
   * `192 x 192 x 3 Bytes = 110.6 KB`
   * **階段 2 & 3：** K 存在 Weight Bank 0，V 存在 Weight Bank 1。**完全不需從 DRAM 讀取 (0 KB)**。
   * **階段 4：** 讀取 Output Projection 的權重 W_out。
   * `192 x 192 x 1 Byte = 36.9 KB`
   * **Weight DRAM 總讀取： 110.6 + 36.9 = 147.5 KB**

3. 輸出寫回 (OFM Write to DRAM)

   * **階段 1、2、3：** Q, K, V 以及 Attention Map 這些巨大的中間產物，全都被死死鎖在 SRAM 裡。**完全不寫回 DRAM (0 KB)**。
   * **階段 4：** 寫回最終保留的 INT32 殘差基準，以及給下一層運算的 INT8 正規化結果。
   * INT8 寫回：`198 x 192 x 1 Byte = 37.1 KB`
   * **OFM DRAM 總寫回： 37.1 KB**

* **單次推論總資料傳輸量 (Total Data Transfer / Frame)：**
`IFM (37.1) + Weight (147.5) + OFM (37.1) = 221.7 KB / Frame`
* **系統目標吞吐量 (@ 220 FPS)：**
`221.7 KB/Frame × 220 FPS = 48,774 KB/s ≈ 47.63 MB/s`


# FFN (Feed-Forward Network) 與 前饋網路

* 準備
  * 輸入特徵數據 (IFMap) : 198 x 192。大小為 198 x 192 x 1 Byte = 37.1 KB。(塞入 IFM SRAM 40 KB)。
  * 權重讀取 (W1, W2) : 單層權重高達 192 x 768 = 147.5 KB。可以直接存於 Weight SRAM
  * 中間特徵極限值 : Linear 1 算完後，維度會膨脹到 198 x 768 = 148.5 KB。(晶片內的所有 SRAM 都塞不下，必須寫回 DRAM，並在 Linear 2 進行「序列切塊 Sequence Tiling」)。

* dataflow (分兩個主要階段)
  * **階段 1：Linear 1 (維度擴展 192  768) 與 ShiftGELU**
    * IFM (37.1 KB) 完整載入，**Input Stationary** (此階段不洗掉)。
    * Weight 從上方 Ping-Pong 載入 W1 權重 (148 KB)。因為權重大於 Bank 容量，控制器將 768 個通道切成 4 份 (192 x 768/4 = 37 K) 交替載入 Bank 0 與 Bank 1。
    * PE 陣列算完得到 INT32，丟給 requant unit，轉成 INT8 放於 OFM SRAM (192 個一組)，期間 Bank0 Bank1 交換存儲，然後放到 Gelu 模組運算回傳。

  * **階段 2：Linear 2 (維度收縮 768  192) 與 殘差加法**
    * 因為 148.5 KB 中間特徵塞不進 IFM，且殘差的 INT32 Buffer 只有 30 KB。所以將 198 個 Token 以 32 為單位切塊 (符合 PE 陣列寬度)，切成 `(32, 32, 32, 32, 32, 32, 6)` 共 7 次循環。
    * IFM 從左側載入 32 組 Token 的中間特徵 (32 x 768 = 24 KB)。塞入 IFM SRAM。
    * Weight 從上方 Ping-Pong 載入完整的 W2 權重 (144 KB，四次載入，192 x 192 x 1 = 36 K)。
    * PE 算完得到 32 x 192 的 INT32 結果。
    * 丟到 Adder 做加法 (殘差)，分成兩份，一份存回 residual sram (INT32)，一份送到 norm 後回傳 DRAM (INT8) (如果已經做12次就不做norm)。
    * (重複執行 7 次，把 198 個 Token 算完)。

* 頻寬分析 (單次 Frame)


**1. 輸入讀取 (IFM Read from DRAM)**

  * 階段 1 (Linear 1)： 讀取最原始的 。完美塞入 IFM SRAM，只需讀取一次。
    * `198 x 192 x 1 Byte = 37.1 KB`


  * 階段 2 (Linear 2)：** 讀取 198 x 768 的中間特徵。因為 SRAM 放不下，所以切成 7 次讀取 (32, 32, ..., 6)。但每個 Token 都只從 DRAM 搬出來「一次」()。
    * `198 x 768 x 1 Byte = 148.5 KB`

  * IFM DRAM 總讀取： 37.1 + 148.5 = 185.6 KB

**2. 權重讀取 (Weight Read from DRAM)**

  * 階段 1 (W1) IFM 是固定在 SRAM 裡不動的，所以 W1 只要從 DRAM 讀進來一次 ()。分 4 次 Ping-Pong (每次 36 KB)。
    * `192 x 768 x 1 Byte = 144.0 KB`
  * 階段 2 (W2)：只需載入一次 :
    * `768 x 192 x 1 Byte = 144.0 KB`

  * Weight DRAM 總讀取： 144.0 + 144.0 = 288.0 KB

**3. 輸出寫回 (OFM Write to DRAM)**

  * 階段 1 (Linear 1 輸出)： GELU 算完後的 INT8 中間產物，必須寫回 DRAM 供階段 2 使用。
    * `198 x 768 x 1 Byte = 148.5 KB`

  * 階段 2 (Linear 2 輸出)： 寫回最終保留的 INT32 殘差基準，以及給下一層運算的 INT8 正規化結果（若為第 12 層則免去 INT8）。
    * INT8 寫回：`198 x 192 x 1 Byte = 37.1 KB`

  * OFM DRAM 總寫回： 148.5 + 37.1 = 185.6 KB


* **單次推論總資料傳輸量 (Total Data Transfer / Frame)：**
`IFM (185.6) + Weight (288.0) + OFM (185.6) =  659.2 KB / Frame`
* **系統目標吞吐量 (@ 220 FPS)：**
`659.2 KB/Frame × 220 FPS = 145,024 KB/s ≈  141.63 MB/s`

# Classification Head (分類頭) 與 最終預測

* 準備
  * **輸入特徵數據 (IFMap) :** 經過 12 層 Transformer 後，198 個 Token 中只需要提取第 1 個 (CLS) 與第 2 個 (DIST) Token。大小為 2 x 192 x 4。
  * **權重讀取 (W_cls, W_dist) :** 兩個 Token 各自對應一個 192 x 1000 的分類器權重。總大小為 192 x 1000 x 2 = 375 K。
  (大於 Weight SRAM 的 128 KB，必須使用 **輸出通道切塊 Output Channel Tiling** 配合 Ping-Pong 載入)。
  * **輸出矩陣極限值 :** 輸出為兩個 Token 各自的 1000 類預測分數 (Logits)。2 x 1000 x 4 = 7.8 K


* dataflow (分兩個子階段：處理 CLS 與 處理 DIST)
  * **階段 1：Class Token 分類 (CLS  1000 類)**
    * IFM (0.4 KB) 載入 2 個 Token。
    * Weight SRAM 無法一次吃下單一分類器的 187.5 KB (192 x 1000)。控制器將 1000 個類別切成 4 份(每份250類，大小192 x 250 x 1 = 46.9K)。
    * 這 4 份權重完美交替載入 Weight Bank 0 與 Bank 1 (Ping-Pong 4 次)。
    * PE 陣列進行計算。(因為 IFM 只有 2 個 Token，所以 PE 陣列的 32 個 Row 中，只有前 2 個 Row 在工作，底下 30 個 Row 處於閒置浪費狀態。)
    * PE 算完得到 1 x 250 x 4 的 INT32 分數，向下排空存入 OFM SRAM。(重複 4 次，湊滿 1000 類)。


* **階段 2：Distillation Token 分類 (DIST  1000 類)**
  * IFM 依然 Stationary (不需重新讀取，因為 CLS 和 DIST 一開始就一起讀進來了)。
  * 載入 DIST 專屬的分類器權重  (一樣 375 KB，切 4 份 Ping-Pong 載入)。
  * PE 算完後，得到 DIST 的 1000 類 INT32 分數，存入 OFM SRAM。
  * 最終寫回： 將 OFM SRAM 中總共 8 KB 的 INT32 分數 (包含 CLS 與 DIST) 寫回外部 DRAM，觸發中斷 (Interrupt) 通知主機板的 CPU 收尾。


* 頻寬分析 (單次 Frame)
1. 輸入 (IFM)
   * 設計： 只需要讀取 2 個 Token，且只需讀取 1 次 (m=1)。
   * IFM 頻寬消耗： 2 x 192 x 1 = 0.4 KB

2. 權重
   * 設計： 所有的分類器權重都只需要從 DRAM 讀進來 1 次 ( n = 1 )。
   * 頻寬消耗： 187.5 + 187.5 = 375 KB

3. 輸出回傳 (OFM)
   * 設計： 寫回 INT32 的預測分數 (Logits) 給 CPU。
   * OFM 頻寬消耗： 2 x 1000 x 4 = 7.8 KB

* 總頻寬消耗： 0.4 + 375.0 + 0 + 7.8 = 383.2 KB / Frame
* 帶入 220 FPS： 383.2 x 220 = 82.3 MB/s

# 最終總頻寬估算
Total Data = Patch + 12 x (MHA + FFN) + Class Head

**1. 單次推論總資料量 (KB / Frame)**

  * **Patch Embedding (1 層):** 486.1 KB
  * **MHA (12 層):** 221.7 KB x 12 = 2660.4 KB
  * **FFN (12 層):** 659.2 × 12 = 7,910.4 KB
  * **Classification Head (1 層):** 383.2 KB
  * **單張圖片總傳輸量：** 486.1 + 2660.4 + 7910.4 + 383.2 = 11,440.1 KB ≈ 11.17 MB / Frame

**2. 系統目標吞吐量 (@ 220 FPS)**

  * **總頻寬需求：** 11,440.1 KB/Frame x 220 FPS = 2,516,822 KB/s
  * 換算為 GB/s：2,516,822 / 1024 / 1024 = 2.4 GB/s