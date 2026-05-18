/**
 * Integer Dense Kernel (Fully Connected Layer)
 * =============================================
 * 
 * 全整數全連接層 RTL 實作
 * 
 * 參考：cmodel_rtl_reference/linear_cmodel_reference.py
 * 
 * 算法：
 * 1. 輸入和權重都是 8-bit signed integer
 * 2. MAC (Multiply-Accumulate) 使用 32-bit accumulator
 * 3. 加上 32-bit bias
 * 4. 輸出 32-bit signed integer
 * 
 * 性能：
 * - 延遲：IN_FEATURES + 2 cycles
 * - 吞吐量：每 IN_FEATURES cycles 處理一個輸入向量
 */

module int_dense_kernel #(
    parameter IN_FEATURES = 192,
    parameter OUT_FEATURES = 768
) (
    input  logic clk,
    input  logic rst_n,
    
    // 控制信號
    input  logic start,
    output logic done,
    
    // 輸入數據
    input  logic signed [7:0] x_int,
    input  logic x_valid,
    
    // 權重和偏置（預載入）
    input  logic signed [7:0]  weight_int [OUT_FEATURES][IN_FEATURES],
    input  logic signed [31:0] bias_int [OUT_FEATURES],
    
    // 輸出數據
    output logic signed [31:0] out_int,
    output logic out_valid
);

    // ========================================================================
    // 內部信號
    // ========================================================================
    
    // 狀態機
    typedef enum logic [1:0] {
        IDLE,
        MAC,
        ADD_BIAS,
        DONE
    } state_t;
    
    state_t state, next_state;
    
    // 計數器
    logic [$clog2(IN_FEATURES)-1:0] in_cnt;
    logic [$clog2(OUT_FEATURES)-1:0] out_cnt;
    
    // MAC 累加器
    logic signed [31:0] acc [OUT_FEATURES];
    
    // 輸入緩衝
    logic signed [7:0] x_buf [IN_FEATURES];
    logic [$clog2(IN_FEATURES)-1:0] x_buf_cnt;
    
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
                if (start) begin
                    next_state = MAC;
                end
            end
            
            MAC: begin
                if (in_cnt == IN_FEATURES - 1) begin
                    next_state = ADD_BIAS;
                end
            end
            
            ADD_BIAS: begin
                if (out_cnt == OUT_FEATURES - 1) begin
                    next_state = DONE;
                end else begin
                    next_state = MAC;
                end
            end
            
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // ========================================================================
    // 輸入緩衝
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_buf_cnt <= 0;
            for (int i = 0; i < IN_FEATURES; i++) begin
                x_buf[i] <= 0;
            end
        end else if (state == IDLE && x_valid) begin
            x_buf[x_buf_cnt] <= x_int;
            x_buf_cnt <= x_buf_cnt + 1;
        end else if (state == DONE) begin
            x_buf_cnt <= 0;
        end
    end
    
    // ========================================================================
    // MAC 運算
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_cnt <= 0;
            for (int i = 0; i < OUT_FEATURES; i++) begin
                acc[i] <= 0;
            end
        end else if (state == IDLE) begin
            in_cnt <= 0;
            for (int i = 0; i < OUT_FEATURES; i++) begin
                acc[i] <= 0;
            end
        end else if (state == MAC) begin
            // MAC: acc[i] += x_buf[in_cnt] * weight_int[i][in_cnt]
            for (int i = 0; i < OUT_FEATURES; i++) begin
                acc[i] <= acc[i] + (x_buf[in_cnt] * weight_int[i][in_cnt]);
            end
            in_cnt <= in_cnt + 1;
        end
    end
    
    // ========================================================================
    // 加 Bias 和輸出
    // ========================================================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_cnt <= 0;
            out_int <= 0;
            out_valid <= 0;
        end else if (state == ADD_BIAS) begin
            // 加 bias 並輸出
            out_int <= acc[out_cnt] + bias_int[out_cnt];
            out_valid <= 1;
            out_cnt <= out_cnt + 1;
        end else if (state == DONE) begin
            out_cnt <= 0;
            out_valid <= 0;
        end else begin
            out_valid <= 0;
        end
    end
    
    // ========================================================================
    // 控制信號
    // ========================================================================
    
    assign done = (state == DONE);

endmodule


/**
 * 使用示例
 * ========
 * 
 * // 實例化
 * int_dense_kernel #(
 *     .IN_FEATURES(192),
 *     .OUT_FEATURES(768)
 * ) dense_layer (
 *     .clk(clk),
 *     .rst_n(rst_n),
 *     .start(start),
 *     .done(done),
 *     .x_int(x_int),
 *     .x_valid(x_valid),
 *     .weight_int(weight_int),
 *     .bias_int(bias_int),
 *     .out_int(out_int),
 *     .out_valid(out_valid)
 * );
 * 
 * // 載入權重
 * initial begin
 *     $readmemh("weights.hex", weight_mem);
 *     $readmemh("bias.hex", bias_mem);
 *     
 *     // 將權重載入到 weight_int
 *     for (int i = 0; i < OUT_FEATURES; i++) begin
 *         for (int j = 0; j < IN_FEATURES; j++) begin
 *             weight_int[i][j] = weight_mem[i * IN_FEATURES + j];
 *         end
 *     end
 *     
 *     // 將偏置載入到 bias_int
 *     for (int i = 0; i < OUT_FEATURES; i++) begin
 *         bias_int[i] = bias_mem[i];
 *     end
 * end
 * 
 * // 餵入輸入數據
 * initial begin
 *     wait(rst_n);
 *     
 *     // 餵入輸入向量
 *     for (int i = 0; i < IN_FEATURES; i++) begin
 *         @(posedge clk);
 *         x_int = input_data[i];
 *         x_valid = 1;
 *     end
 *     @(posedge clk);
 *     x_valid = 0;
 *     
 *     // 啟動計算
 *     start = 1;
 *     @(posedge clk);
 *     start = 0;
 *     
 *     // 等待完成
 *     wait(done);
 * end
 * 
 * // 收集輸出
 * always @(posedge clk) begin
 *     if (out_valid) begin
 *         output_data[output_cnt] = out_int;
 *         output_cnt = output_cnt + 1;
 *     end
 * end
 */
