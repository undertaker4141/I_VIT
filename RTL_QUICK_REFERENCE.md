# RTL 實作快速參考卡

## 🎯 目標
實作 DeiT-Tiny 整數量化 Vision Transformer 的 RTL 設計

## 📊 模型規格
```
架構:         DeiT-Tiny
Embedding:    192 維
深度:         12 個 Transformer Blocks
注意力頭:     3 個
MLP 比例:     4 (hidden = 768)
類別數:       1000
輸入大小:     224×224×3
Patch 大小:   16×16
參數量:       ~5.7M
量化大小:     ~1.4MB (int8)
準確率:       73.54% Top-1
```

## 🔧 數據類型

| 層級 | 輸入 | 權重 | 輸出 | 累加器 |
|------|------|------|------|--------|
| Dense/Conv | int8 | int8 | int32 | int32 |
| MatMul | int8/int32 | int8/int32 | int32 | int32 |
| LayerNorm | int8/int32 | - | int32 | int64 |
| GELU | int8/int32 | - | int32 | int32 |
| Softmax | int32 | - | int8 | uint32 |

## 📁 關鍵文件位置

### C-Model 算法參考
```
cmodel_rtl_reference/
├── linear_cmodel_reference.py      # 線性層算法
└── nonlinear_cmodel_reference.py   # 非線性層算法
```

### RTL 測試向量
```
rtl_vectors/
├── input_image.hex                 # 輸入 (3 種格式)
├── patch_embed_output.hex          # Patch Embedding 輸出
├── block0_norm1_output.hex         # Block 0 測試
├── final_output.hex                # 最終輸出
└── prediction.txt                  # 預期結果: Class 1, Prob 0.3996
```

### Golden Patterns
```
golden_patterns/
├── golden_patterns.npz             # 378 個中間層數據
└── model_weights.npz               # 模型權重
```

## 🧮 核心算法

### 1. 線性層 (Dense)
```verilog
// 輸入: x[7:0] (int8), w[7:0] (int8), b[31:0] (int32)
// 輸出: y[31:0] (int32)

// Step 1: MAC 陣列
for (i = 0; i < OUT_FEATURES; i++) begin
    acc[i] = b[i];  // 初始化為 bias
    for (j = 0; j < IN_FEATURES; j++) begin
        acc[i] = acc[i] + x[j] * w[i][j];  // 32-bit 累加
    end
    y[i] = acc[i];
end
```

### 2. LayerNorm
```verilog
// 輸入: x[7:0] (int8), b[31:0] (int32)
// 輸出: y[31:0] (int32)

// Step 1: 計算 mean (int16 溢位模擬)
sum = 0;
for (i = 0; i < N; i++) sum = sum + x[i];
sum_wrapped = sum[15:0];  // 模擬 int16 溢位
mean = sum_wrapped / N;

// Step 2: Centering (int16 溢位模擬)
for (i = 0; i < N; i++) begin
    y[i] = x[i] - mean;
    y[i] = y[i][15:0];  // 模擬 int16 溢位
end

// Step 3: Variance (uint32)
var = 0;
for (i = 0; i < N; i++) var = var + y[i] * y[i];

// Step 4: Newton 迭代求 sqrt (10 次)
std = 2^16;
for (iter = 0; iter < 10; iter++) begin
    std = (std + var / std) / 2;
end

// Step 5: Normalize
factor = 2^31 - 1;
for (i = 0; i < N; i++) begin
    y_norm[i] = (y[i] * (factor / std)) / 2;
    y[i] = y_norm[i] + b[i];
end
```

### 3. GELU
```verilog
// 輸入: x[7:0] (int8)
// 輸出: y[31:0] (int32)

// Step 1: Stability shift
x_max = max(x);
for (i = 0; i < N; i++) x_algo[i] = x[i] - x_max;

// Step 2: Exponential (使用 Shift-based 近似)
for (i = 0; i < N; i++) begin
    exp1[i] = shift_exp(x_algo[i]);
end
exp_max = shift_exp(-x_max);

// Step 3: Sigmoid
for (i = 0; i < N; i++) begin
    sum_exp = exp1[i] + exp_max;
    factor = (2^31 - 1) / sum_exp;
    sigmoid[i] = (exp1[i] * factor) >> 24;  // 右移到 8-bit
end

// Step 4: 輸出
for (i = 0; i < N; i++) y[i] = x[i] * sigmoid[i];
```

### 4. Softmax
```verilog
// 輸入: x[31:0] (int32)
// 輸出: y[7:0] (int8)

// Step 1: Stability shift
x_max = max(x);
for (i = 0; i < N; i++) x_algo[i] = x[i] - x_max;

// Step 2: Exponential
for (i = 0; i < N; i++) exp[i] = shift_exp(x_algo[i]);

// Step 3: Normalize
sum_exp = 0;
for (i = 0; i < N; i++) sum_exp = sum_exp + exp[i];
factor = (2^31 - 1) / sum_exp;

// Step 4: 輸出
for (i = 0; i < N; i++) y[i] = (exp[i] * factor) >> 24;
```

## 🔍 驗證方法

### SystemVerilog Testbench 模板
```systemverilog
module tb_vit_layer;
    // 宣告記憶體
    reg [31:0] input_mem [0:150527];
    reg [31:0] golden_mem [0:999];
    reg [31:0] output_mem [0:999];
    
    // 載入測試向量
    initial begin
        $readmemh("rtl_vectors/input_image.hex", input_mem);
        $readmemh("rtl_vectors/final_output.hex", golden_mem);
    end
    
    // 實例化 DUT
    vit_model dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(input_mem),
        .output_data(output_mem),
        .valid(valid)
    );
    
    // 驗證
    integer i, errors;
    initial begin
        wait(valid);
        errors = 0;
        
        for (i = 0; i < 1000; i = i + 1) begin
            if (output_mem[i] !== golden_mem[i]) begin
                $error("Mismatch at %d: got %h, expected %h",
                       i, output_mem[i], golden_mem[i]);
                errors = errors + 1;
            end
        end
        
        if (errors == 0) begin
            $display("✓ Verification PASSED!");
            $display("  Predicted class: 1");
            $display("  Expected class: 1");
        end else begin
            $display("✗ Verification FAILED with %d errors", errors);
        end
        
        $finish;
    end
endmodule
```

## 📈 實作順序建議

### Phase 1: 基礎模組 (Week 1-2)
- [ ] `int_dense_kernel` - 線性層
- [ ] `int_matmul_kernel` - 矩陣乘法
- [ ] 單元測試與驗證

### Phase 2: 非線性模組 (Week 3-4)
- [ ] `int_layer_norm` - LayerNorm
- [ ] `int_gelu` - GELU 激活
- [ ] `int_softmax` - Softmax
- [ ] 單元測試與驗證

### Phase 3: Attention 模組 (Week 5-6)
- [ ] QKV Projection
- [ ] Multi-Head Attention
- [ ] Attention Output Projection
- [ ] 集成測試

### Phase 4: Transformer Block (Week 7-8)
- [ ] MLP (Feed-Forward)
- [ ] Residual Connections
- [ ] 完整 Block 驗證

### Phase 5: 完整模型 (Week 9-10)
- [ ] Patch Embedding
- [ ] Position Embedding
- [ ] 12 個 Transformer Blocks
- [ ] Classification Head
- [ ] 端到端驗證

## 🎯 驗證檢查點

### Checkpoint 1: 線性層
```
輸入: 隨機 int8 [2, 192]
權重: 隨機 int8 [768, 192]
預期: 與 C-Model 100% 匹配
```

### Checkpoint 2: LayerNorm
```
輸入: 隨機 int8 [2, 197, 192]
預期: 輸出範圍合理，無溢位
```

### Checkpoint 3: Transformer Block
```
輸入: block0_norm1_output.hex
預期: 與 Golden Patterns 匹配
```

### Checkpoint 4: 完整模型
```
輸入: input_image.hex
預期輸出: final_output.hex
預期類別: 1 (概率 0.3996)
```

## ⚡ 性能優化建議

### 並行化
- 使用多個 MAC 單元並行計算
- Attention 的多頭可以並行處理
- MLP 的 FC1/FC2 可以流水線化

### 記憶體優化
- 使用雙緩衝技術
- 權重可以預載入到 BRAM
- 中間結果可以重用 buffer

### 時序優化
- 將長路徑分割成多個流水線階段
- LayerNorm 的 Newton 迭代可以展開
- 除法器可以用 LUT 替代

## 🐛 常見問題

### Q: LayerNorm 輸出不匹配？
A: 檢查是否正確模擬了 int16 溢位行為

### Q: GELU/Softmax 輸出全為 0？
A: 檢查 Shift-based exponential 的實作，特別是動態移位器

### Q: 最終準確率低於預期？
A: 逐層檢查，使用 Golden Patterns 定位問題層

### Q: 資源使用過高？
A: 考慮時分複用 MAC 單元，或使用更小的並行度

## 📞 獲取幫助

### 文檔
- `RTL_SIMULATION_READY.md` - 完整指南
- `docs/cmodel/RTL_SIMULATION_GUIDE.md` - 詳細說明
- `docs/cmodel/HARDWARE_CMODEL_GUIDE.md` - 硬體建議

### 工具
- `verify_rtl_setup.py` - 驗證環境設置
- `I-ViT/verify_cmodel.py` - 驗證 C-Model

### C-Model 參考
- `cmodel_rtl_reference/linear_cmodel_reference.py`
- `cmodel_rtl_reference/nonlinear_cmodel_reference.py`

---

**祝 RTL 實作順利！** 🚀
