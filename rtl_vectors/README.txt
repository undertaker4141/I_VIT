RTL 測試向量
================================================================================

此目錄包含從 PyTorch 量化模型提取的 Golden Patterns，
轉換為 RTL 模擬器可讀的格式。

文件說明：
--------------------------------------------------------------------------------
input_image.*        - 輸入圖片（歸一化並量化後）
patch_embed_output.* - Patch Embedding 層輸出
block0_norm1_output.*- Block 0 LayerNorm1 輸出（單元測試用）
final_output.*       - 最終輸出 logits
prediction.txt       - 預測結果

格式說明：
--------------------------------------------------------------------------------
.hex - 十六進制格式（每行一個值）
.mem - 記憶體初始化格式（@地址 數據）
.txt - 十進制文本格式

使用方法：
--------------------------------------------------------------------------------
在 SystemVerilog testbench 中：
  $readmemh("input_image.hex", input_mem);
  $readmemh("final_output.hex", golden_mem);

在 Verilog testbench 中：
  $readmemh("input_image.hex", input_mem);
  $readmemh("final_output.hex", golden_mem);
