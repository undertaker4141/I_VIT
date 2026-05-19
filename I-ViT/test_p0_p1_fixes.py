"""
P0 和 P1 修復驗收測試
==================================================================
測試目標：
1. P0: 驗證 requantize_integer 和 quant_act_residual_integer 的正確性
2. P1: 驗證 GELU 和 Softmax 不再使用 object 類型
3. 比較修復前後的數值差異

測試方法：
- 單元測試：測試新函數的正確性
- 對比測試：比較新舊實現的差異
- 類型檢查：確認不再使用 object 類型
"""

import sys
import os
import numpy as np

# 添加父目錄到路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'cmodel_rtl_reference'))

from pure_numpy_cmodel import (
    precompute_requant_params,
    requantize_integer,
    quant_act_residual_integer,
    requantize,
    quant_act_residual,
    int_gelu,
    int_softmax
)


def test_p0_precompute_requant_params():
    """測試 P0: precompute_requant_params"""
    print("\n" + "="*80)
    print("P0 測試 1: precompute_requant_params")
    print("="*80)
    
    # 測試案例 1: scale > 1 (放大)
    input_sf = 0.01
    output_sf = 0.001
    M, S = precompute_requant_params(input_sf, output_sf)
    
    scale_expected = input_sf / output_sf  # = 10.0
    scale_actual = M / (2 ** S)
    error = abs(scale_actual - scale_expected) / scale_expected
    
    print(f"\n案例 1: scale > 1 (放大)")
    print(f"  input_sf = {input_sf}, output_sf = {output_sf}")
    print(f"  期望 scale = {scale_expected}")
    print(f"  M = {M}, S = {S}")
    print(f"  實際 scale = M / 2^S = {scale_actual}")
    print(f"  相對誤差 = {error*100:.6f}%")
    
    assert error < 0.01, f"誤差過大: {error*100:.2f}%"
    print("  ✓ 通過")
    
    # 測試案例 2: scale < 1 (縮小)
    input_sf = 0.001
    output_sf = 0.01
    M, S = precompute_requant_params(input_sf, output_sf)
    
    scale_expected = input_sf / output_sf  # = 0.1
    scale_actual = M / (2 ** S)
    error = abs(scale_actual - scale_expected) / scale_expected
    
    print(f"\n案例 2: scale < 1 (縮小)")
    print(f"  input_sf = {input_sf}, output_sf = {output_sf}")
    print(f"  期望 scale = {scale_expected}")
    print(f"  M = {M}, S = {S}")
    print(f"  實際 scale = M / 2^S = {scale_actual}")
    print(f"  相對誤差 = {error*100:.6f}%")
    
    assert error < 0.01, f"誤差過大: {error*100:.2f}%"
    print("  ✓ 通過")
    
    # 測試案例 3: scale ≈ 1
    input_sf = 0.01
    output_sf = 0.0099
    M, S = precompute_requant_params(input_sf, output_sf)
    
    scale_expected = input_sf / output_sf
    scale_actual = M / (2 ** S)
    error = abs(scale_actual - scale_expected) / scale_expected
    
    print(f"\n案例 3: scale ≈ 1")
    print(f"  input_sf = {input_sf}, output_sf = {output_sf}")
    print(f"  期望 scale = {scale_expected}")
    print(f"  M = {M}, S = {S}")
    print(f"  實際 scale = M / 2^S = {scale_actual}")
    print(f"  相對誤差 = {error*100:.6f}%")
    
    assert error < 0.01, f"誤差過大: {error*100:.2f}%"
    print("  ✓ 通過")


def test_p0_requantize_integer():
    """測試 P0: requantize_integer vs requantize"""
    print("\n" + "="*80)
    print("P0 測試 2: requantize_integer vs requantize (浮點版本)")
    print("="*80)
    
    # 生成測試數據
    np.random.seed(42)
    x_int = np.random.randint(-32768, 32767, (2, 10, 192), dtype=np.int16)
    input_sf = 0.01
    output_sf = 0.005
    
    # 舊實現（浮點）
    output_float = requantize(x_int, input_sf, output_sf, output_bits=8)
    
    # 新實現（純整數）
    M, S = precompute_requant_params(input_sf, output_sf)
    output_int = requantize_integer(x_int, M, S, output_bits=8)
    
    # 比較差異
    diff = output_int.astype(np.int32) - output_float.astype(np.int32)
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    
    print(f"\n測試數據:")
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  input_sf = {input_sf}, output_sf = {output_sf}")
    print(f"  M = {M}, S = {S}")
    
    print(f"\n浮點版本輸出:")
    print(f"  shape={output_float.shape}, dtype={output_float.dtype}, range=[{output_float.min()}, {output_float.max()}]")
    
    print(f"\n整數版本輸出:")
    print(f"  shape={output_int.shape}, dtype={output_int.dtype}, range=[{output_int.min()}, {output_int.max()}]")
    
    print(f"\n差異分析:")
    print(f"  最大差異: {max_diff}")
    print(f"  平均差異: {mean_diff:.6f}")
    print(f"  差異 > 0 的比例: {np.sum(np.abs(diff) > 0) / diff.size * 100:.2f}%")
    
    # 驗證：差異應該 <= 1 LSB
    assert max_diff <= 1, f"最大差異過大: {max_diff}"
    print("  ✓ 通過（差異 <= 1 LSB）")


def test_p0_quant_act_residual_integer():
    """測試 P0: quant_act_residual_integer vs quant_act_residual"""
    print("\n" + "="*80)
    print("P0 測試 3: quant_act_residual_integer vs quant_act_residual (浮點版本)")
    print("="*80)
    
    # 生成測試數據
    np.random.seed(42)
    x1_int = np.random.randint(-32768, 32767, (2, 10, 192), dtype=np.int16)
    x1_sf = 0.01
    x2_int = np.random.randint(-32768, 32767, (2, 10, 192), dtype=np.int16)
    x2_sf = 0.015
    output_sf = 0.012
    
    # 舊實現（浮點）
    output_float = quant_act_residual(x1_int, x1_sf, x2_int, x2_sf, output_sf, output_bits=16)
    
    # 新實現（純整數）
    output_int = quant_act_residual_integer(x1_int, x1_sf, x2_int, x2_sf, output_sf, output_bits=16)
    
    # 比較差異
    diff = output_int.astype(np.int32) - output_float.astype(np.int32)
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    
    print(f"\n測試數據:")
    print(f"  輸入1: shape={x1_int.shape}, dtype={x1_int.dtype}, sf={x1_sf}")
    print(f"  輸入2: shape={x2_int.shape}, dtype={x2_int.dtype}, sf={x2_sf}")
    print(f"  output_sf = {output_sf}")
    
    print(f"\n浮點版本輸出:")
    print(f"  shape={output_float.shape}, dtype={output_float.dtype}, range=[{output_float.min()}, {output_float.max()}]")
    
    print(f"\n整數版本輸出:")
    print(f"  shape={output_int.shape}, dtype={output_int.dtype}, range=[{output_int.min()}, {output_int.max()}]")
    
    print(f"\n差異分析:")
    print(f"  最大差異: {max_diff}")
    print(f"  平均差異: {mean_diff:.6f}")
    print(f"  差異 > 0 的比例: {np.sum(np.abs(diff) > 0) / diff.size * 100:.2f}%")
    
    # 驗證：差異應該較小（可能 > 1 LSB，因為有兩次 requantize）
    assert max_diff <= 5, f"最大差異過大: {max_diff}"
    print(f"  ✓ 通過（差異 <= 5）")


def test_p1_gelu_no_object():
    """測試 P1: GELU 不再使用 object 類型"""
    print("\n" + "="*80)
    print("P1 測試 1: GELU 不使用 object 類型")
    print("="*80)
    
    # 生成測試數據
    np.random.seed(42)
    x_int = np.random.randint(-1000, 1000, (2, 10, 192), dtype=np.int32)
    scaling_factor = 0.01
    
    # 執行 GELU
    output = int_gelu(x_int, scaling_factor, output_bit=8, n=23)
    
    print(f"\n測試數據:")
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  scaling_factor = {scaling_factor}")
    
    print(f"\n輸出:")
    print(f"  shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 驗證：輸出應該是 int32
    assert output.dtype == np.int32, f"輸出類型錯誤: {output.dtype}"
    print("  ✓ 通過（輸出類型為 int32）")
    
    # 驗證：沒有 NaN 或 Inf
    assert not np.any(np.isnan(output)), "輸出包含 NaN"
    assert not np.any(np.isinf(output)), "輸出包含 Inf"
    print("  ✓ 通過（無 NaN 或 Inf）")


def test_p1_softmax_no_object():
    """測試 P1: Softmax 不再使用 object 類型"""
    print("\n" + "="*80)
    print("P1 測試 2: Softmax 不使用 object 類型")
    print("="*80)
    
    # 生成測試數據
    np.random.seed(42)
    x_int = np.random.randint(-1000, 1000, (2, 10, 64), dtype=np.int32)
    scaling_factor = 0.01
    
    # 執行 Softmax
    output = int_softmax(x_int, scaling_factor, output_bit=16, n=15)
    
    print(f"\n測試數據:")
    print(f"  輸入: shape={x_int.shape}, dtype={x_int.dtype}, range=[{x_int.min()}, {x_int.max()}]")
    print(f"  scaling_factor = {scaling_factor}")
    
    print(f"\n輸出:")
    print(f"  shape={output.shape}, dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 驗證：輸出應該是 int32
    assert output.dtype == np.int32, f"輸出類型錯誤: {output.dtype}"
    print("  ✓ 通過（輸出類型為 int32）")
    
    # 驗證：沒有 NaN 或 Inf
    assert not np.any(np.isnan(output)), "輸出包含 NaN"
    assert not np.any(np.isinf(output)), "輸出包含 Inf"
    print("  ✓ 通過（無 NaN 或 Inf）")


def test_p0_p1_integration():
    """整合測試：P0 和 P1 一起工作"""
    print("\n" + "="*80)
    print("整合測試: P0 和 P1 一起工作")
    print("="*80)
    
    # 模擬一個簡單的 Transformer block 流程
    np.random.seed(42)
    
    # 1. Linear 層輸出（int32）
    x_int32 = np.random.randint(-10000, 10000, (1, 197, 768), dtype=np.int32)
    linear_sf = 0.005
    
    # 2. Requantize 到 int8
    print("\n步驟 1: Requantize (int32 -> int8)")
    M, S = precompute_requant_params(linear_sf, 0.01)
    x_int8 = requantize_integer(x_int32, M, S, output_bits=8)
    print(f"  輸入: dtype={x_int32.dtype}, range=[{x_int32.min()}, {x_int32.max()}]")
    print(f"  輸出: dtype={x_int8.dtype}, range=[{x_int8.min()}, {x_int8.max()}]")
    print(f"  M={M}, S={S}")
    
    # 3. GELU
    print("\n步驟 2: GELU")
    gelu_input = x_int32  # GELU 輸入是 int32
    gelu_output = int_gelu(gelu_input, linear_sf, output_bit=8, n=23)
    print(f"  輸入: dtype={gelu_input.dtype}, range=[{gelu_input.min()}, {gelu_input.max()}]")
    print(f"  輸出: dtype={gelu_output.dtype}, range=[{gelu_output.min()}, {gelu_output.max()}]")
    
    # 4. Residual 連接
    print("\n步驟 3: Residual 連接")
    residual = np.random.randint(-32768, 32767, (1, 197, 768), dtype=np.int16)
    residual_sf = 0.012
    output_sf = 0.01
    
    # 將 gelu_output 轉換為 int16
    M_gelu, S_gelu = precompute_requant_params(linear_sf * (1/128), output_sf)
    gelu_int16 = requantize_integer(gelu_output, M_gelu, S_gelu, output_bits=16)
    
    # Residual 連接
    output = quant_act_residual_integer(gelu_int16, output_sf, residual, residual_sf, output_sf, output_bits=16)
    print(f"  主路徑: dtype={gelu_int16.dtype}, range=[{gelu_int16.min()}, {gelu_int16.max()}]")
    print(f"  殘差: dtype={residual.dtype}, range=[{residual.min()}, {residual.max()}]")
    print(f"  輸出: dtype={output.dtype}, range=[{output.min()}, {output.max()}]")
    
    # 驗證
    assert output.dtype == np.int16, f"輸出類型錯誤: {output.dtype}"
    assert not np.any(np.isnan(output)), "輸出包含 NaN"
    assert not np.any(np.isinf(output)), "輸出包含 Inf"
    print("\n  ✓ 整合測試通過")


def main():
    """主測試函數"""
    print("="*80)
    print("P0 和 P1 修復驗收測試")
    print("="*80)
    print("\nP0 修復: 整數化 Requantize 路徑")
    print("  - precompute_requant_params()")
    print("  - requantize_integer()")
    print("  - quant_act_residual_integer()")
    print("\nP1 修復: GELU/Softmax 使用明確的 int64")
    print("  - int_gelu() 不再使用 object")
    print("  - int_softmax() 不再使用 object")
    
    try:
        # P0 測試
        test_p0_precompute_requant_params()
        test_p0_requantize_integer()
        test_p0_quant_act_residual_integer()
        
        # P1 測試
        test_p1_gelu_no_object()
        test_p1_softmax_no_object()
        
        # 整合測試
        test_p0_p1_integration()
        
        # 總結
        print("\n" + "="*80)
        print("✓ 所有測試通過！")
        print("="*80)
        print("\nP0 修復驗收結果:")
        print("  ✓ precompute_requant_params() 正確計算 M 和 S")
        print("  ✓ requantize_integer() 與浮點版本差異 <= 1 LSB")
        print("  ✓ quant_act_residual_integer() 與浮點版本差異 <= 5")
        print("\nP1 修復驗收結果:")
        print("  ✓ int_gelu() 不再使用 object 類型")
        print("  ✓ int_softmax() 不再使用 object 類型")
        print("  ✓ 輸出類型為 int32，無 NaN 或 Inf")
        print("\n整合測試:")
        print("  ✓ P0 和 P1 修復可以一起工作")
        print("\n" + "="*80)
        print("結論: P0 和 P1 修復已完成並通過驗收")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
