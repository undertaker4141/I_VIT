/**
 * Integer LayerNorm
 * =================
 * 
 * 全整數 LayerNorm RTL 實作
 * 
 * 參考：cmodel_rtl_reference/nonlinear_cmodel_reference.py
 * 
 * 算法：
 * 1. 計算 mean (int16 溢位模擬)
 * 2. Centering: y = x - mean
 * 3. 計算 variance (uint32)
 * 4. Newton 迭代求 sqrt (10 次)
 * 5. Normalize: y_norm = y * factor / std / 2
 * 6. 加 bias
 * 
 * 注意：
 * - Mean 計算有 int16 溢位（模擬硬體行為）
 * - Variance 使用 uint32（避免溢位）
 * - Newton 迭代需要 10 個 clock cycles
 * - 中間計算需要 64-bit 暫存器
 */

module int_layer_norm #(
    parameter N = 192  // Feature dimension
) (
    input  logic clk,
    input  logic rst_n,
    
    // 控制信號
    input  logic start,
    output logic done,
    
    // 輸入數據
    input  logic signed [31:0] x_int [N],
    input  logic x_valid,
    
    // Bias（預載入）
    input  logic signed [31:0] bias_int [N],
    
    // 輸出數據
    output logic signed [31:0] out_int [N],
    output logic out_valid
);

    // ========================================================================
    // 內部信號
    // ========================================================================
    
    // 狀態機
    typedef enum logic [3:0] {
        IDLE,
        CALC_MEAN,
        CENTER,
        CALC_VAR,
        NEWTON_INIT,
        NEWTON_ITER,
        NORMALIZE,
        ADD_BIAS,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    
    // 計數器
    logic [$clog2(N)-1:0] cnt;
    logic [3:0] newton_cnt;  // Newton 迭代計數器（0-9）
    
    // 中間結果
    logic signed [31:0] sum_val;
    logic signed [15:0] sum_wrapped;
    logic signed [31:0] mean_int;
    logic signed [31:0] y_int [N];
    logic [31:0] data_sq [N];
    logic [31:0] var;
    logic [31:0] std;
    logic [31:0] std_safe;
    logic signed [31:0] factor;
    logic signed [31:0] factor_div_std;
    logic signed [63:0] term [N];
    logic signed [31:0] y_norm [N];
    
    // 常數
    localparam signed [31:0] FACTOR_CONST = 32'h7FFFFFFF;  // 2^31 - 1
    localparam [31:0] STD_INIT = 32'h00010000;  // 2^16 = 65536
    
    // ========================================================================
    // 狀態機
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start && x_valid) begin
                    next_state = CALC_MEAN;
                end
            end
            
            CALC_MEAN: begin
                if (cnt == N - 1) begin
                    next_state = CENTER;
                end
            end
            
            CENTER: begin
                if (cnt == N - 1) begin
                    next_state = CALC_VAR;
                end
            end
            
            CALC_VAR: begin
                if (cnt == N - 1) begin
                    next_state = NEWTON_INIT;
                end
            end
            
            NEWTON_INIT: begin
                next_state = NEWTON_ITER;
            end
            
            NEWTON_ITER: begin
                if (newton_cnt == 9) begin  // 10 次迭代（0-9）
                    next_state = NORMALIZE;
                end
            end
            
            NORMALIZE: begin
                if (cnt == N - 1) begin
                    next_state = ADD_BIAS;
                end
            end
            
            ADD_BIAS: begin
                if (cnt == N - 1) begin
                    next_state = DONE_STATE;
                end
            end
            
            DONE_STATE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // ========================================================================
    // Step 1: 計算 Mean (int16 溢位模擬)
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_val <= 0;
            cnt <= 0;
        end else if (state == IDLE) begin
            sum_val <= 0;
            cnt <= 0;
        end else if (state == CALC_MEAN) begin
            sum_val <= sum_val + x_int[cnt];
            cnt <= cnt + 1;
            
            if (cnt == N - 1) begin
                // 模擬 int16 溢位
                sum_wrapped <= sum_val[15:0];
                // 計算 mean (truncated division towards zero)
                mean_int <= $signed(sum_wrapped) / $signed(N);
            end
        end
    end
    
    // ========================================================================
    // Step 2: Centering (y = x - mean)
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0;
        end else if (state == CENTER) begin
            // y_int = x_int - mean_int
            y_int[cnt] <= x_int[cnt] - mean_int;
            
            // 模擬 int16 溢位
            y_int[cnt] <= $signed(y_int[cnt][15:0]);
            
            cnt <= cnt + 1;
        end else if (state == CENTER && cnt == N - 1) begin
            cnt <= 0;
        end
    end
    
    // ========================================================================
    // Step 3: 計算 Variance (uint32)
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            var <= 0;
            cnt <= 0;
        end else if (state == CALC_VAR) begin
            // data_sq = y_int * y_int (轉為 uint32)
            data_sq[cnt] <= $unsigned(y_int[cnt] * y_int[cnt]);
            
            // 累加 variance
            var <= var + data_sq[cnt];
            
            cnt <= cnt + 1;
        end else if (state == CALC_VAR && cnt == N - 1) begin
            cnt <= 0;
        end
    end
    
    // ========================================================================
    // Step 4: Newton 迭代求 sqrt (10 次)
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            std <= 0;
            newton_cnt <= 0;
        end else if (state == NEWTON_INIT) begin
            std <= STD_INIT;  // 初始值 2^16
            newton_cnt <= 0;
        end else if (state == NEWTON_ITER) begin
            // std = (std + var / std) / 2
            std_safe <= (std == 0) ? 1 : std;
            std <= (std + (var / std_safe)) / 2;
            newton_cnt <= newton_cnt + 1;
        end
    end
    
    // ========================================================================
    // Step 5: Normalize
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            factor_div_std <= 0;
            cnt <= 0;
        end else if (state == NORMALIZE) begin
            if (cnt == 0) begin
                // 計算 factor / std
                std_safe <= (std == 0) ? 1 : std;
                factor_div_std <= FACTOR_CONST / $signed(std_safe);
            end
            
            // term = factor_div_std * y_int (64-bit)
            term[cnt] <= $signed(factor_div_std) * $signed(y_int[cnt]);
            
            // y_norm = term / 2 (truncated division towards zero)
            if (term[cnt] >= 0) begin
                y_norm[cnt] <= term[cnt] / 2;
            end else begin
                y_norm[cnt] <= -((-term[cnt]) / 2);
            end
            
            cnt <= cnt + 1;
        end else if (state == NORMALIZE && cnt == N - 1) begin
            cnt <= 0;
        end
    end
    
    // ========================================================================
    // Step 6: 加 Bias
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0;
            out_valid <= 0;
        end else if (state == ADD_BIAS) begin
            out_int[cnt] <= y_norm[cnt] + bias_int[cnt];
            cnt <= cnt + 1;
            
            if (cnt == N - 1) begin
                out_valid <= 1;
            end
        end else if (state == DONE_STATE) begin
            cnt <= 0;
            out_valid <= 0;
        end else begin
            out_valid <= 0;
        end
    end
    
    // ========================================================================
    // 控制信號
    // ========================================================================
    
    assign done = (state == DONE_STATE);

endmodule


/**
 * 使用示例
 * ========
 * 
 * // 實例化
 * int_layer_norm #(
 *     .N(192)
 * ) layer_norm (
 *     .clk(clk),
 *     .rst_n(rst_n),
 *     .start(start),
 *     .done(done),
 *     .x_int(x_int),
 *     .x_valid(x_valid),
 *     .bias_int(bias_int),
 *     .out_int(out_int),
 *     .out_valid(out_valid)
 * );
 * 
 * // 載入 bias
 * initial begin
 *     $readmemh("bias.hex", bias_mem);
 *     for (int i = 0; i < N; i++) begin
 *         bias_int[i] = bias_mem[i];
 *     end
 * end
 * 
 * // 餵入輸入數據
 * initial begin
 *     wait(rst_n);
 *     
 *     // 載入輸入向量
 *     for (int i = 0; i < N; i++) begin
 *         x_int[i] = input_data[i];
 *     end
 *     x_valid = 1;
 *     
 *     // 啟動計算
 *     start = 1;
 *     @(posedge clk);
 *     start = 0;
 *     x_valid = 0;
 *     
 *     // 等待完成
 *     wait(done);
 * end
 * 
 * // 收集輸出
 * always @(posedge clk) begin
 *     if (out_valid) begin
 *         for (int i = 0; i < N; i++) begin
 *             output_data[i] = out_int[i];
 *         end
 *     end
 * end
 * 
 * 性能分析：
 * - 延遲：約 N + 15 cycles (N 個元素 + 10 次 Newton 迭代 + 其他)
 * - 面積：需要 N 個 32-bit 暫存器 + 64-bit 乘法器 + 除法器
 * - 優化建議：
 *   1. Pipeline Newton 迭代
 *   2. 使用查找表（LUT）加速除法
 *   3. 並行處理多個元素
 */
