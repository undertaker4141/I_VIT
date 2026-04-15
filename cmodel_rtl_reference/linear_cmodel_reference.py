"""
IViT RTL C-Model 參考實作: 線性層 (Dense, MatMul)
==================================================================
此檔案包含了已通過 TVM 全整數推論 100% 驗證 (Bit-Exact) 的線性層演算法算式。
線性層在對稱量化 (Symmetric Quantization, Zero Point = 0) 的條件下，
運算邏輯極度簡化，退化為純粹的整數矩陣乘法加法器 (MAC) 陣列，
過程不需要參與任何的 Scale Shift 或 Offset 補償。

硬體設計團隊 (RTL Team) 可以參考下列極具體的資料型別轉換方式，
在設計 MAC (Multiply-Accumulate) 單元時，能確保 bit-width 正確且不會溢位。
"""

import numpy as np

def int_dense_kernel(x_int, weight_int, bias_int=None):
    """
    全整數 Dense / Linear / 全連接層
    
    硬體層面演算法流 (Algorithm Flow):
    
    [Step 1] 資料匯入與展寬 (Bit-width Extension):
        輸入 x_int: 型別為 int8 (或等效的 8-bit 整數訊號)
        權重 weight_int: 型別為 int8 
        ** 硬體在進入 MAC (DSP塊) 之前，應直接將這兩者作為 8-bit Signed Integer 
           送入 Multiplier。**
           
    [Step 2] 乘累加 (MAC Array):
        乘積結果會是 16-bit 有號整數。
        經過序列長度或 Channel 數目的累加後，必須使用至少 32-bit 的 Accumulator 
        (也就是轉換為 int32) 來儲存結果，否則極度容易引發溢位 (Overflow)。
        在 NumPy 的行為中等效於:
            x_val = int32(x_int)
            w_val = int32(weight_int)
            out_val = x_val @ w_val^T  (輸出為 int32)
            
    [Step 3] 加上 Bias (選項):
        如果有 Bias，Bias 原本即應為 32-bit 整數參數，直接與 MAC 的結果相加即可。
        out_val = out_val + bias_int32
        
    [Step 4] 輸出:
        結果為 int32 的特徵值，接下來可以直接進入下一層的 Add 或是 Requantize 塊。
    """
    x_val = x_int.astype(np.int32)
    w_val = weight_int.astype(np.int32)
    
    # 進行矩陣內積: MAC 累加
    out_val = np.matmul(x_val, w_val.T)
    
    if bias_int is not None:
        out_val = out_val + bias_int.astype(np.int32)
        
    return out_val.astype(np.int32)


def int_matmul_kernel(x_int, y_int):
    """
    全整數 Batch Matrix Multiplication (適用於 Attention Q @ K^T)
    
    硬體層面演算法流 (Algorithm Flow):
    
    [Step 1] 資料準備:
        x_int 代表 Query (Q)，y_int 代表 Key (K)。
        兩者皆為 8-bit / 32-bit 之整數 (視前面 Requantize 結果而定)。
        為安全起見，這裡也是同樣使用 int32 作為運算基準位元寬。
        
    [Step 2] 記憶體讀取與轉置 (Transpose):
        對於硬體的 Memory Controller 或 Buffer 而言，K 的最後兩個維度 
        (Sequences 與 Head Dimension) 是需要對調讀取順序的 (Transpose)。
        如果是 4D Data: (Batch, Heads, M, K) 和 (Batch, Heads, N, K)，
        硬體只需要把 y_int 視為 (Batch, Heads, K, N) 的順序依序餵入 MAC 陣列。
        
    [Step 3] 乘累加 (MAC Array):
        執行與 Dense 相同的矩陣乘累加運算，不需要其他 Offset：
        sum(x[..., m, k] * y[..., n, k]) 
        
    [Step 4] 輸出:
        一樣需確保輸出暫存器能容納 32-bit 的數值。
    """
    x_val = x_int.astype(np.int32)
    y_val = y_int.astype(np.int32)
    
    # y_val.transpose(0, 1, 3, 2) 即代表對倒數第二、第一的反轉(硬體轉置讀取)
    out_val = np.matmul(x_val, y_val.transpose(0, 1, 3, 2))
    
    return out_val.astype(np.int32)
