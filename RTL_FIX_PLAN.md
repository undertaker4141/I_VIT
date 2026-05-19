# C-Model RTL 落地修復計劃

**創建日期**: 2026-05-19  
**目標**: 將 C-Model 從 PoC 級別提升到 RTL 落地標準

---

## 修復優先級

```
P0 (最優先) → P1 (次優先) → P2 (RTL 面) → P3 (驗證面)
     ↓              ↓              ↓              ↓
  整數化         修復 GELU/    修復 RTL      Block 精度
 Requantize      Softmax      Templates      驗證 ≥98%
```

---

## P0: 整數化 Requantize 路徑（最優先）

### 目標
將 `requantize()` 和 `quant_act_residual()` 從浮點運算改為純整數運算

### 當前問題
```python
# ❌ 當前實現（浮點）
scale = input_sf / output_sf                        # 浮點除法
output = np.round(x_int.astype(np.float32) * scale) # 浮點乘法
```

### 修復方案

#### Step 1: 預計算 M 和 S 值

創建新函數 `precompute_requant_params()`:

```python
def precompute_requant_params(input_sf, output_sf):
    """
    預計算 requantization 參數
    
    scale = input_sf / output_sf = M * 2^(-S)
    
    返回:
        M: 整數乘數 (int32)
        S: 右移位數 (int)
    """
    scale = input_sf / output_sf
    
    # 找到合適的 S，使得 M 在 int32 範圍內
    # 目標: M = round(scale * 2^S)，且 M < 2^31
    
    if scale >= 1.0:
        # scale >= 1: 需要放大
        S = 0
        M = int(np.round(scale))
    else:
        # scale < 1: 需要縮小
        # 找到最大的 S，使得 M = scale * 2^S < 2^31
        S = 0
        while S < 31:
            M_candidate = scale * (2 ** (S + 1))
            if M_candidate >= (2**31 - 1):
                break
            S += 1
        
        M = int(np.round(scale * (2 ** S)))
    
    return M, S
```

#### Step 2: 純整數 Requantize

```python
def requantize_integer(x_int, M, S, output_bits=8):
    """
    純整數 Requantization
    
    參數:
        x_int: 整數輸入 (int8/int16/int32)
        M: 整數乘數 (int32)
        S: 右移位數 (int)
        output_bits: 輸出位元數 (8/16/32)
    
    返回:
        output_int: 重新量化的整數輸出
    
    算法:
        output = round((x * M) >> S)
               = ((x * M) + (1 << (S-1))) >> S
    """
    # 使用 int64 避免乘法溢出
    x_int64 = x_int.astype(np.int64)
    M_int64 = np.int64(M)
    
    # 乘法
    scaled = x_int64 * M_int64
    
    # 右移前加上 rounding bias (相當於 round)
    if S > 0:
        rounding_bias = np.int64(1) << (S - 1)
        output = (scaled + rounding_bias) >> S
    else:
        output = scaled
    
    # Clip to output range
    if output_bits == 8:
        return np.clip(output, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(output, -32768, 32767).astype(np.int16)
    else:
        return np.clip(output, -2147483648, 2147483647).astype(np.int32)
```

#### Step 3: 純整數殘差連接

```python
def quant_act_residual_integer(x1_int, M1, S1, x2_int, M2, S2, M_out, S_out, output_bits=16):
    """
    純整數 QuantAct with Residual
    
    參數:
        x1_int: 第一個輸入（主路徑）
        M1, S1: x1 的 requantization 參數
        x2_int: 第二個輸入（殘差路徑）
        M2, S2: x2 的 requantization 參數
        M_out, S_out: 輸出的 requantization 參數
        output_bits: 輸出位元數
    
    返回:
        output_int: 重新量化的整數輸出
    
    算法:
        1. x1_scaled = (x1 * M1) >> S1
        2. x2_scaled = (x2 * M2) >> S2
        3. y = x1_scaled + x2_scaled
        4. output = (y * M_out) >> S_out
    """
    # Step 1: Requantize x1
    x1_int64 = x1_int.astype(np.int64)
    x1_scaled = (x1_int64 * M1 + (1 << (S1 - 1))) >> S1 if S1 > 0 else x1_int64 * M1
    
    # Step 2: Requantize x2
    x2_int64 = x2_int.astype(np.int64)
    x2_scaled = (x2_int64 * M2 + (1 << (S2 - 1))) >> S2 if S2 > 0 else x2_int64 * M2
    
    # Step 3: Add
    y = x1_scaled + x2_scaled
    
    # Step 4: Requantize output
    output = (y * M_out + (1 << (S_out - 1))) >> S_out if S_out > 0 else y * M_out
    
    # Clip to output range
    if output_bits == 8:
        return np.clip(output, -128, 127).astype(np.int8)
    elif output_bits == 16:
        return np.clip(output, -32768, 32767).astype(np.int16)
    else:
        return np.clip(output, -2147483648, 2147483647).astype(np.int32)
```

#### Step 4: 更新所有調用點

需要更新的函數:
1. `int_linear()` - 輸出 requantize
2. `int_matmul()` - 輸出 requantize
3. `int_layer_norm()` - 輸出 requantize
4. `int_gelu()` - 輸出 requantize
5. `int_softmax()` - 輸出 requantize
6. `int_vit_block()` - 殘差連接

### 驗證方法

1. **單元測試**: 測試 `requantize_integer()` 與浮點版本的誤差
2. **Block 測試**: 重跑 12 個 block，比較精度
3. **端到端測試**: 重跑 100 張圖片，確認準確率不下降

### 預期結果

- ✅ 所有 requantize 操作使用純整數
- ✅ 誤差 < 1 LSB（最低有效位）
- ✅ Block 精度 ≥ 當前水平（不下降）

---

## P1: 修復 GELU/Softmax 的 object 類型（次優先）

### 目標
將 `int_gelu()` 和 `int_softmax()` 中的 `object` 類型改為明確的 `int64`

### 當前問題
```python
# ❌ 當前實現（Python 任意精度）
term = exp_int.astype(object) * factor.astype(object)  # 不是 int64！
sigmoid_int = (term >> shift_amt).astype(np.int64)
```

### 修復方案

#### Step 1: 分析溢出範圍

```python
# GELU 中的乘法
# exp_int: 最大值 ~2^31 (int32 max)
# factor: 最大值 ~2^31 (int32 max)
# term = exp_int * factor: 最大值 ~2^62

# Softmax 中的乘法
# exp_int: 最大值 ~2^31 (int32 max)
# factor: 最大值 ~2^31 (int32 max)
# term = exp_int * factor: 最大值 ~2^62
```

結論: **需要 64-bit 整數**，但不會溢出 int64 (2^63)

#### Step 2: 使用明確的 int64 乘法

```python
def int_gelu(x_int, scaling_factor, output_bit=8, n=20):
    """
    純整數 GELU（修復版）
    """
    # ... (前面的代碼不變)
    
    # Step 6: Scale and shift（修復：使用明確的 int64）
    shift_amt = 31 - output_bit + 1  # = 24 for output_bit=8
    
    # 明確使用 int64 乘法
    exp_int64 = exp_int.astype(np.int64)
    factor_int64 = factor.astype(np.int64)
    
    # 64-bit 乘法（不會溢出）
    term = exp_int64 * factor_int64
    
    # 右移
    sigmoid_int = (term >> shift_amt).astype(np.int64)
    
    # Step 7: Multiply with input
    output = pre_x_int * sigmoid_int
    
    return output.astype(np.int32)
```

#### Step 3: 添加溢出檢查（可選，用於驗證）

```python
def int64_multiply_checked(a, b, shift_amt):
    """
    64-bit 整數乘法，帶溢出檢查
    """
    a_int64 = a.astype(np.int64)
    b_int64 = b.astype(np.int64)
    
    # 乘法
    result = a_int64 * b_int64
    
    # 檢查溢出（可選）
    max_val = np.max(np.abs(result))
    if max_val > 2**62:
        print(f"Warning: Large multiplication result: {max_val}")
    
    # 右移
    output = result >> shift_amt
    
    return output
```

### 驗證方法

1. **數值測試**: 比較 `object` 版本和 `int64` 版本的輸出差異
2. **溢出測試**: 確認最大值不超過 2^62
3. **Block 測試**: 重跑 GELU 和 Softmax 相關的 blocks

### 預期結果

- ✅ 所有乘法使用明確的 `int64`
- ✅ 無溢出（最大值 < 2^62）
- ✅ 數值誤差 = 0（與 object 版本完全相同）

---

## P2: 修復 RTL Templates 合成錯誤

### 目標
修復 LayerNorm 和 Dense Kernel 的 RTL 合成錯誤

### 問題 1: LayerNorm Multiple Driver

**當前問題**:
```systemverilog
// ❌ cnt 被多個 always_ff 驅動
always_ff @(posedge clk) begin
    if (state == COMPUTE_MEAN) cnt <= cnt + 1;
end

always_ff @(posedge clk) begin
    if (state == COMPUTE_VAR) cnt <= cnt + 1;
end

// ... 還有 2 個 always_ff block 也驅動 cnt
```

**修復方案**:
```systemverilog
// ✅ 合併到一個 always_ff block
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cnt <= 0;
    end else begin
        case (state)
            COMPUTE_MEAN, COMPUTE_VAR, COMPUTE_NORM, NEWTON_ITER: begin
                cnt <= cnt + 1;
            end
            default: begin
                cnt <= 0;
            end
        endcase
    end
end
```

### 問題 2: LayerNorm RTL 除法

**當前問題**:
```systemverilog
// ❌ RTL 除法無法合成
mean <= $signed(sum_wrapped) / $signed(N);
```

**修復方案**:
```systemverilog
// ✅ 使用預計算的倒數
// 假設 N = 768，預計算 1/N * 2^20
parameter int INV_N = (1 << 20) / N;  // 預計算

// 乘法 + 右移取代除法
mean <= ($signed(sum_wrapped) * INV_N) >>> 20;
```

### 問題 3: Dense Kernel 並行度

**當前問題**:
```systemverilog
// ❌ 768 個並行乘法器
for (int i = 0; i < 768; i++) begin
    acc[i] <= acc[i] + (x_buf[in_cnt] * weight_int[i][in_cnt]);
end
```

**修復方案**:
```systemverilog
// ✅ 使用 PE (Processing Element) 架構
// 假設 PE_NUM = 16（可配置）

parameter int PE_NUM = 16;
parameter int OUT_PER_PE = OUT_FEATURES / PE_NUM;  // 768 / 16 = 48

// 每個 PE 處理 48 個輸出
genvar pe_idx;
generate
    for (pe_idx = 0; pe_idx < PE_NUM; pe_idx++) begin : pe_array
        int_dense_pe #(
            .IN_FEATURES(IN_FEATURES),
            .OUT_FEATURES(OUT_PER_PE)
        ) pe_inst (
            .clk(clk),
            .rst_n(rst_n),
            .x_int(x_int),
            .weight_int(weight_int[pe_idx*OUT_PER_PE +: OUT_PER_PE]),
            .bias_int(bias_int[pe_idx*OUT_PER_PE +: OUT_PER_PE]),
            .y_int(y_int[pe_idx*OUT_PER_PE +: OUT_PER_PE])
        );
    end
endgenerate
```

### 驗證方法

1. **語法檢查**: `verilator --lint-only`
2. **合成測試**: Vivado/Quartus 合成
3. **功能驗證**: RTL 仿真 vs C-Model

---

## P3: Block 精度驗證（≥98%）

### 目標
確認所有 12 個 blocks 的精度 ≥98%

### 當前狀態

| Block | 相關係數 | 狀態 |
|-------|---------|------|
| 0 | 98.7% | ✅ |
| 1 | 94.4% | ⚠️ |
| 2 | 87.5% | ❌ |
| 3 | 91.7% | ⚠️ |
| 4 | 84.1% | ❌ |
| 5-11 | ? | ? |

### 驗證流程

1. **完成 P0 和 P1 修復**
2. **重跑 Block 測試**:
   ```bash
   python I-ViT/test_all_blocks.py
   ```
3. **分析低精度 Blocks**:
   - 找出誤差最大的層
   - 檢查 SF 是否合理
   - 檢查 requantize 參數
4. **修復並重測**

### 預期結果

- ✅ 所有 12 個 blocks 精度 ≥98%
- ✅ 端到端準確率 ≥87%（不下降）

---

## 實施時間表

| 週 | 任務 | 交付物 |
|----|------|--------|
| Week 1 | P0: 整數化 Requantize | `requantize_integer()`, `quant_act_residual_integer()` |
| Week 1 | P1: 修復 GELU/Softmax | 明確 int64 乘法 |
| Week 2 | P2: 修復 RTL Templates | 可合成的 LayerNorm, Dense Kernel |
| Week 2 | P3: Block 精度驗證 | 所有 blocks ≥98% 報告 |

---

## 成功標準

### C-Model 標準
- ✅ 所有運算使用純整數（int8/int16/int32/int64）
- ✅ 無浮點運算（除了預計算 M, S）
- ✅ Block 精度 ≥98%
- ✅ 端到端準確率 ≥87%

### RTL 標準
- ✅ 所有 templates 可合成
- ✅ 無 multiple driver 錯誤
- ✅ 無 RTL 除法
- ✅ 明確的並行度設計

### 文檔標準
- ✅ 更新 `HARDWARE_CMODEL_GUIDE.md`
- ✅ 創建 `REQUANTIZE_SPEC.md`（M, S 計算規格）
- ✅ 創建 `RTL_IMPLEMENTATION_GUIDE.md`

---

## 風險與緩解

| 風險 | 影響 | 緩解措施 |
|-----|------|---------|
| 整數化後精度下降 | 高 | 使用更高精度的 M, S（增加位寬） |
| int64 乘法溢出 | 中 | 添加飽和邏輯 |
| RTL 面積過大 | 中 | 調整 PE 數量（並行度） |
| Block 精度無法達標 | 高 | 需要重新訓練模型 |

---

**創建人**: Kiro AI  
**審核人**: [待填寫]  
**批准人**: [待填寫]
