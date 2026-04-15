# I-ViT RTL C-Model 參考實作

此目錄 (`cmodel_rtl_reference`) 存放了已由 TVM Inference 模型 100% 驗證的 C-Model Python 原型碼。這些代碼的作用為**提供給硬體開發 (RTL) 工程師最直接的演算法實作參考**，確保 Verilog/SystemVerilog 的數值行為與高階軟體 (PyTorch / TVM) 的全整數流動完全 Bit-Exact 無誤差。

## 為什麼需要這些參考代碼？

因為 PyTorch 的 `round(x / scale)` 在模擬量化時是以 Float32 運算模擬，與真實底層硬體 (C++, LLVM Backend) 的定點運算在極限狀況下（如負數的截斷方式、Shift 的進位）是不相同的，會導致難以追蹤的精靈誤差 (Off-by-one errors)。

這裡面的腳本演算法，已經 100% 對齊了硬體指令集的行為（例如使用 `np.fix`, `astype(np.int16)` 模擬溢位），因此保證正確。

## 包含檔案

1. **`linear_cmodel_reference.py`**
   - 包含 `int_dense_kernel` 和 `int_matmul_kernel`。
   - **應用場景**: MAC Array 設計、QKV Projection、Fully Connected (FC1 / FC2 模組)、Attention Matrix Multiplication (Q @ K^T, Softmax @ V)。
   - **核心精神**: 對稱量化架構下，無需位移補償，完全退化為有號整數的矩陣乘積加法。

2. **`nonlinear_cmodel_reference.py`**
   - 包含 `int_exp_shift_kernel_standard`, `int_gelu_kernel_fixed`, `int_softmax_kernel_fixed`, `int_layer_norm_fixed`。
   - **應用場景**: Activation Functions 與正規化層設計。
   - **核心精神**: 解構出極度詳細的演算法流（Algorithm Flow）。如何避免 32-bit 溢位？何時需要轉為 uint32 甚至 64-bit register？如何做 Newton 迭代找 STD 平方根？皆有詳細註解。

## 硬體開發建議 (RTL Implementation Tips)

- **留意 Shift 與 除數 (Division)**
  Python 中的 `//` 運算符會向下取整 (Floor, e.g., `-3 // 2 = -2`)，而 C/C++/Verilog 中的整數除法通常截斷趨近於零 (Truncation towards zero, e.g., `-3 / 2 = -1`)。在我們的 C-Model 中皆已盡力寫為等效的 C-behavior。但在實作 Verilog 的位移指令 (Logical Shift Right vs. Arithmetic Shift Right) 時，請依照註解中指名的截斷方向規劃。
  
- **LayerNorm 暫存器位寬設計**
  LayerNorm 過程中的 Square 和 Variance 總和，由於都是正數，請放心的使用 **unsigned** (無號數) 加法器與暫存器，並提供至少 32-bit 的位寬。而在乘上 `factor` (2^31 - 1) 時，務必提升至 64-bit 暫存空間避免 Overflow，再透過右移或除法壓回 32-bit。
