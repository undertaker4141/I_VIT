#!/usr/bin/env python3
"""
C-Model Float 驗證腳本
使用 TVM golden patterns 和 Float 驗證方法
"""
import numpy as np
import os
import sys
from pathlib import Path

# 添加專案根目錄到路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cmodel_rtl_reference.nonlinear_cmodel_reference import (
    int_layer_norm_fixed,
    int_gelu_kernel_fixed,
    int_softmax_kernel_fixed
)

# TVM patterns 目錄
PATTERNS_DIR = PROJECT_ROOT / "patterns_tvm"
NUM_BLOCKS = 12


def load_npy(path):
    """載入 .npy 檔案"""
    if not os.path.exists(path):
        return None
    return np.load(path)


def verify_float(cmodel_output, golden_output, scale, layer_name, tolerance=0.001):
    """
    使用 Float 驗證方法
    
    Args:
        cmodel_output: C-model 整數輸出
        golden_output: TVM golden pattern 整數輸出
        scale: scaling factor (可以是 scalar 或 per-channel array)
        layer_name: 層名稱
        tolerance: 容許的最大浮點誤差
    
    Returns:
        bool: 是否通過驗證
    """
    # 轉換為浮點
    if isinstance(scale, np.ndarray) and scale.size > 1:
        # Per-channel scale
        scale_bc = scale.reshape(1, 1, -1) if len(cmodel_output.shape) == 3 else scale
        cmodel_float = cmodel_output.astype(np.float32) * scale_bc
        golden_float = golden_output.astype(np.float32) * scale_bc
    else:
        # Scalar scale
        scale_val = float(scale) if isinstance(scale, np.ndarray) else scale
        cmodel_float = cmodel_output.astype(np.float32) * scale_val
        golden_float = golden_output.astype(np.float32) * scale_val
    
    # 計算誤差
    diff = np.abs(cmodel_float - golden_float)
    max_error = diff.max()
    mean_error = diff.mean()
    
    # 判定
    passed = max_error < tolerance
    
    # 輸出結果
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {layer_name}")
    print(f"    Max error:  {max_error:.8f} (tolerance: {tolerance})")
    print(f"    Mean error: {mean_error:.8f}")
    
    if not passed:
        print(f"    ⚠️  超過容許誤差！")
    
    return passed


def verify_integer_tolerance(cmodel_output, golden_output, layer_name, tolerance=100):
    """
    使用整數容差驗證方法（備選）
    
    Args:
        cmodel_output: C-model 整數輸出
        golden_output: TVM golden pattern 整數輸出
        layer_name: 層名稱
        tolerance: 容許的整數差異
    
    Returns:
        bool: 是否通過驗證
    """
    diff = np.abs(cmodel_output.astype(np.int64) - golden_output.astype(np.int64))
    max_diff = diff.max()
    pass_rate = (diff <= tolerance).mean() * 100
    
    passed = pass_rate > 99.0
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {layer_name}")
    print(f"    Max diff:   {max_diff} (tolerance: ±{tolerance})")
    print(f"    Pass rate:  {pass_rate:.2f}%")
    
    if not passed:
        print(f"    ⚠️  通過率低於 99%！")
    
    return passed


def test_layernorm_float(block_id):
    """測試 LayerNorm 層（使用 Float 驗證）"""
    # 載入輸入
    if block_id == 0:
        input_path = PATTERNS_DIR / "golden" / "embed_add_pos.npy"
    else:
        input_path = PATTERNS_DIR / "golden" / f"block_{block_id - 1}_add2.npy"
    
    # 載入 golden output 和參數
    golden_path = PATTERNS_DIR / "golden" / f"block_{block_id}_norm1_int.npy"
    bias_path = PATTERNS_DIR / "weights" / f"block_{block_id}_norm1_bias.npy"
    scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_norm1_output_scale_float.npy"
    
    x_int = load_npy(input_path)
    golden = load_npy(golden_path)
    bias = load_npy(bias_path)
    scale = load_npy(scale_path)
    
    if any(v is None for v in [x_int, golden, bias, scale]):
        return None, "SKIP"
    
    # 執行 C-model
    cmodel_output = int_layer_norm_fixed(x_int.astype(np.int32), bias)
    
    # Float 驗證
    passed = verify_float(cmodel_output, golden, scale, 
                         f"Block {block_id} LayerNorm", tolerance=0.001)
    
    return passed, None


def test_gelu_float(block_id):
    """測試 GELU 層（使用 Float 驗證）"""
    # 載入輸入和參數
    input_path = PATTERNS_DIR / "golden" / f"block_{block_id}_fc1_int.npy"
    golden_path = PATTERNS_DIR / "golden" / f"block_{block_id}_gelu_int.npy"
    gelu_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_gelu_input_scale_float.npy"
    fc1_out_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_fc1_output_scale_float.npy"
    gelu_out_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_gelu_output_scale_float.npy"
    
    x_int = load_npy(input_path)
    golden = load_npy(golden_path)
    gelu_scale_arr = load_npy(gelu_scale_path)
    fc1_out_scale = load_npy(fc1_out_scale_path)
    gelu_out_scale = load_npy(gelu_out_scale_path)
    
    if any(v is None for v in [x_int, golden, gelu_scale_arr, fc1_out_scale]):
        return None, "SKIP"
    
    input_scale = float(gelu_scale_arr.flatten()[0])
    
    # Per-channel requantization
    ratio = fc1_out_scale.flatten() / input_scale
    x_float = x_int.astype(np.float64)
    ratio_bc = ratio.reshape(1, 1, -1)
    x_req = np.round(x_float * ratio_bc).astype(np.int32)
    x_req = np.clip(x_req, -128, 127).astype(np.int32)
    
    # 計算 x0_int
    x0_int = int(-1.0 / (input_scale * 1.702) - 1)
    x0_int = np.int64(x0_int)
    
    # 執行 C-model
    cmodel_output = int_gelu_kernel_fixed(x_req, x0_int, output_bit=8, n=23)
    
    # Float 驗證
    if gelu_out_scale is not None:
        scale = float(gelu_out_scale.flatten()[0])
    else:
        scale = input_scale  # fallback
    
    passed = verify_float(cmodel_output, golden, scale,
                         f"Block {block_id} GELU", tolerance=0.001)
    
    return passed, None


def test_softmax_float(block_id):
    """測試 Softmax 層（使用 Float 驗證）"""
    # 載入輸入和參數
    input_path = PATTERNS_DIR / "golden" / f"block_{block_id}_attn_matmul1_int.npy"
    golden_path = PATTERNS_DIR / "golden" / f"block_{block_id}_softmax_int.npy"
    m1_out_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_matmul_1_output_scale_float.npy"
    sm_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_softmax_input_scale_float.npy"
    sm_out_scale_path = PATTERNS_DIR / "scales" / f"block_{block_id}_qconfig_softmax_output_scale_float.npy"
    
    x_int = load_npy(input_path)
    golden = load_npy(golden_path)
    m1_out_scale = load_npy(m1_out_scale_path)
    sm_scale_arr = load_npy(sm_scale_path)
    sm_out_scale = load_npy(sm_out_scale_path)
    
    if any(v is None for v in [x_int, golden, m1_out_scale, sm_scale_arr]):
        return None, "SKIP"
    
    input_scale = float(sm_scale_arr.flatten()[0])
    m1_out_scale_val = float(m1_out_scale.flatten()[0])
    qk_scale = (192 // 3) ** -0.5  # = 0.125
    
    # Requantization
    ratio = (m1_out_scale_val * qk_scale) / input_scale
    x_req = np.round(x_int.astype(np.float64) * ratio).astype(np.int32)
    x_req = np.clip(x_req, -128, 127).astype(np.int32)
    
    # 計算 x0_int
    x0_int = int(-1.0 / input_scale - 1)
    x0_int = np.int64(x0_int)
    
    # 執行 C-model
    cmodel_output = int_softmax_kernel_fixed(x_req, x0_int, output_bit=8, n=16)
    
    # Float 驗證
    if sm_out_scale is not None:
        scale = float(sm_out_scale.flatten()[0])
    else:
        scale = 1.0 / 128  # 8-bit output scale
    
    passed = verify_float(cmodel_output, golden.astype(np.int32), scale,
                         f"Block {block_id} Softmax", tolerance=0.001)
    
    return passed, None


def main():
    """主程式"""
    print("=" * 70)
    print("C-Model Float 驗證 (使用 TVM Golden Patterns)")
    print("=" * 70)
    print(f"Patterns 目錄: {PATTERNS_DIR}")
    print(f"驗證方法: Float 驗證 (max error < 0.001)")
    print("=" * 70)
    
    all_results = {
        'layernorm': [],
        'gelu': [],
        'softmax': []
    }
    
    # 測試 LayerNorm
    print(f"\n{'─'*70}")
    print("測試 LayerNorm (Float 驗證)")
    print(f"{'─'*70}")
    for block_id in range(NUM_BLOCKS):
        passed, skip = test_layernorm_float(block_id)
        if skip:
            print(f"  ⏭️  Block {block_id}: {skip}")
            continue
        all_results['layernorm'].append(passed)
    
    # 測試 GELU
    print(f"\n{'─'*70}")
    print("測試 GELU (Float 驗證)")
    print(f"{'─'*70}")
    for block_id in range(NUM_BLOCKS):
        passed, skip = test_gelu_float(block_id)
        if skip:
            print(f"  ⏭️  Block {block_id}: {skip}")
            continue
        all_results['gelu'].append(passed)
    
    # 測試 Softmax
    print(f"\n{'─'*70}")
    print("測試 Softmax (Float 驗證)")
    print(f"{'─'*70}")
    for block_id in range(NUM_BLOCKS):
        passed, skip = test_softmax_float(block_id)
        if skip:
            print(f"  ⏭️  Block {block_id}: {skip}")
            continue
        all_results['softmax'].append(passed)
    
    # 總結
    print(f"\n{'='*70}")
    print("驗證總結")
    print(f"{'='*70}")
    
    for layer_type, results in all_results.items():
        if not results:
            print(f"  {layer_type.upper()}: 無測試結果")
            continue
        
        passed_count = sum(results)
        total_count = len(results)
        pass_rate = passed_count / total_count * 100
        
        status = "✅" if pass_rate == 100 else "⚠️" if pass_rate >= 90 else "❌"
        print(f"  {status} {layer_type.upper()}: {passed_count}/{total_count} PASS ({pass_rate:.1f}%)")
    
    print(f"{'='*70}")
    
    # 判定整體結果
    all_passed = all(all(results) for results in all_results.values() if results)
    
    if all_passed:
        print("\n🎉 所有測試通過！C-model 與 TVM 完全對齊。")
        return 0
    else:
        print("\n⚠️  部分測試未通過，請檢查上述錯誤訊息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
