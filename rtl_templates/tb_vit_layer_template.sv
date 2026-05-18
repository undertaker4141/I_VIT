/**
 * Vision Transformer Layer Testbench Template
 * ============================================
 * 
 * 此 testbench 模板展示如何使用 Golden Patterns 驗證 RTL 實作
 * 
 * 使用方法：
 * 1. 修改參數以匹配你的設計
 * 2. 實例化你的 RTL 模組
 * 3. 載入測試向量
 * 4. 執行模擬並比較結果
 */

`timescale 1ns / 1ps

module tb_vit_layer;

    // ========================================================================
    // 參數定義
    // ========================================================================
    
    // 時鐘參數
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz
    
    // 數據參數（根據實際設計修改）
    parameter INPUT_SIZE = 224 * 224 * 3;   // 輸入圖片大小
    parameter OUTPUT_SIZE = 1000;            // 輸出類別數
    parameter DATA_WIDTH = 8;                // 數據位寬
    
    // ========================================================================
    // 信號定義
    // ========================================================================
    
    // 時鐘和復位
    logic clk;
    logic rst_n;
    
    // 控制信號
    logic start;
    logic done;
    
    // 數據信號
    logic signed [DATA_WIDTH-1:0] input_data;
    logic signed [31:0] output_data;
    logic output_valid;
    
    // 記憶體
    logic [7:0] input_mem [0:INPUT_SIZE-1];
    logic [31:0] golden_mem [0:OUTPUT_SIZE-1];
    logic [31:0] output_mem [0:OUTPUT_SIZE-1];
    
    // 計數器
    integer input_cnt;
    integer output_cnt;
    integer error_cnt;
    
    // ========================================================================
    // 時鐘生成
    // ========================================================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ========================================================================
    // DUT 實例化
    // ========================================================================
    
    // TODO: 實例化你的 RTL 模組
    // 例如：
    // vit_layer #(
    //     .INPUT_SIZE(INPUT_SIZE),
    //     .OUTPUT_SIZE(OUTPUT_SIZE)
    // ) dut (
    //     .clk(clk),
    //     .rst_n(rst_n),
    //     .start(start),
    //     .input_data(input_data),
    //     .output_data(output_data),
    //     .output_valid(output_valid),
    //     .done(done)
    // );
    
    // ========================================================================
    // 測試流程
    // ========================================================================
    
    initial begin
        // 初始化
        rst_n = 0;
        start = 0;
        input_cnt = 0;
        output_cnt = 0;
        error_cnt = 0;
        
        // 載入測試向量
        $display("========================================");
        $display("Loading test vectors...");
        $display("========================================");
        
        // 載入輸入數據
        $readmemh("../rtl_vectors/input_image.hex", input_mem);
        $display("Loaded input data: %0d bytes", INPUT_SIZE);
        
        // 載入 Golden 輸出
        $readmemh("../rtl_vectors/final_output.hex", golden_mem);
        $display("Loaded golden output: %0d values", OUTPUT_SIZE);
        
        // 復位
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        // 開始測試
        $display("\n========================================");
        $display("Starting test...");
        $display("========================================\n");
        
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        // 等待完成
        wait(done);
        
        // 比較結果
        $display("\n========================================");
        $display("Comparing results...");
        $display("========================================\n");
        
        for (int i = 0; i < OUTPUT_SIZE; i++) begin
            if (output_mem[i] !== golden_mem[i]) begin
                $error("Mismatch at index %0d: got %h, expected %h", 
                       i, output_mem[i], golden_mem[i]);
                error_cnt++;
            end
        end
        
        // 顯示結果
        $display("\n========================================");
        $display("Test Results");
        $display("========================================");
        $display("Total outputs: %0d", OUTPUT_SIZE);
        $display("Errors: %0d", error_cnt);
        
        if (error_cnt == 0) begin
            $display("\n*** TEST PASSED ***\n");
        end else begin
            $display("\n*** TEST FAILED ***\n");
        end
        
        $display("========================================\n");
        
        // 結束模擬
        #(CLK_PERIOD * 10);
        $finish;
    end
    
    // ========================================================================
    // 輸入數據餵入
    // ========================================================================
    
    // TODO: 根據你的設計修改數據餵入邏輯
    // 例如：
    // always @(posedge clk) begin
    //     if (rst_n && start && input_cnt < INPUT_SIZE) begin
    //         input_data <= input_mem[input_cnt];
    //         input_cnt <= input_cnt + 1;
    //     end
    // end
    
    // ========================================================================
    // 輸出數據收集
    // ========================================================================
    
    // TODO: 根據你的設計修改數據收集邏輯
    // 例如：
    // always @(posedge clk) begin
    //     if (rst_n && output_valid && output_cnt < OUTPUT_SIZE) begin
    //         output_mem[output_cnt] <= output_data;
    //         output_cnt <= output_cnt + 1;
    //     end
    // end
    
    // ========================================================================
    // 波形記錄
    // ========================================================================
    
    initial begin
        $dumpfile("tb_vit_layer.vcd");
        $dumpvars(0, tb_vit_layer);
    end
    
    // ========================================================================
    // 超時保護
    // ========================================================================
    
    initial begin
        #(CLK_PERIOD * 1000000);  // 1M cycles timeout
        $error("Simulation timeout!");
        $finish;
    end

endmodule
