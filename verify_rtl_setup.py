"""
驗證 RTL 模擬環境設置
==================================================================
此腳本檢查所有必要的文件是否已正確生成，並提供設置狀態報告。
"""

import os
import sys

def check_file(filepath, description):
    """檢查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    size = ""
    if exists and os.path.isfile(filepath):
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            size = f" ({size_bytes} B)"
        elif size_bytes < 1024*1024:
            size = f" ({size_bytes/1024:.1f} KB)"
        else:
            size = f" ({size_bytes/(1024*1024):.1f} MB)"
    
    print(f"  {status} {description}{size}")
    return exists

def check_directory(dirpath, description):
    """檢查目錄是否存在"""
    exists = os.path.exists(dirpath) and os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    count = ""
    if exists:
        files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        count = f" ({len(files)} 個文件)"
    
    print(f"  {status} {description}{count}")
    return exists

def main():
    """主函數"""
    print("="*80)
    print("RTL 模擬環境設置驗證")
    print("="*80)
    
    all_checks = []
    
    # 1. 檢查 PyTorch 模型
    print("\n1. PyTorch 量化模型")
    print("-"*80)
    all_checks.append(check_file("I-ViT/output_gpu/checkpoint_converted.pth", 
                                  "Checkpoint (轉換後)"))
    all_checks.append(check_file("I-ViT/test_pytorch_only.py", 
                                  "PyTorch 測試腳本"))
    
    # 2. 檢查 C-Model
    print("\n2. C-Model 算法單元")
    print("-"*80)
    all_checks.append(check_file("cmodel_rtl_reference/linear_cmodel_reference.py", 
                                  "線性層 C-Model"))
    all_checks.append(check_file("cmodel_rtl_reference/nonlinear_cmodel_reference.py", 
                                  "非線性層 C-Model"))
    all_checks.append(check_file("cmodel_rtl_reference/pytorch_cmodel.py", 
                                  "完整 C-Model 框架"))
    all_checks.append(check_file("I-ViT/verify_cmodel.py", 
                                  "C-Model 驗證腳本"))
    
    # 3. 檢查 Golden Patterns
    print("\n3. Golden Patterns")
    print("-"*80)
    all_checks.append(check_directory("golden_patterns", 
                                       "Golden Patterns 目錄"))
    all_checks.append(check_file("golden_patterns/golden_patterns.npz", 
                                  "Golden Patterns 數據"))
    all_checks.append(check_file("golden_patterns/golden_patterns_report.txt", 
                                  "Golden Patterns 報告"))
    all_checks.append(check_file("golden_patterns/model_weights.npz", 
                                  "模型權重"))
    all_checks.append(check_file("golden_patterns/model_weights_report.txt", 
                                  "模型權重報告"))
    
    # 4. 檢查 RTL 測試向量
    print("\n4. RTL 測試向量")
    print("-"*80)
    all_checks.append(check_directory("rtl_vectors", 
                                       "RTL 測試向量目錄"))
    all_checks.append(check_file("rtl_vectors/input_image.hex", 
                                  "輸入圖片 (.hex)"))
    all_checks.append(check_file("rtl_vectors/input_image.mem", 
                                  "輸入圖片 (.mem)"))
    all_checks.append(check_file("rtl_vectors/input_image.txt", 
                                  "輸入圖片 (.txt)"))
    all_checks.append(check_file("rtl_vectors/final_output.hex", 
                                  "最終輸出 (.hex)"))
    all_checks.append(check_file("rtl_vectors/prediction.txt", 
                                  "預測結果"))
    all_checks.append(check_file("rtl_vectors/README.txt", 
                                  "使用說明"))
    
    # 5. 檢查文檔
    print("\n5. 文檔")
    print("-"*80)
    all_checks.append(check_file("RTL_SIMULATION_READY.md", 
                                  "RTL 模擬準備完成報告"))
    all_checks.append(check_file("README_CMODEL.md", 
                                  "C-Model 快速開始指南"))
    all_checks.append(check_file("docs/cmodel/FINAL_SUMMARY.md", 
                                  "完整總結"))
    all_checks.append(check_file("docs/cmodel/RTL_SIMULATION_GUIDE.md", 
                                  "RTL 模擬指南"))
    all_checks.append(check_file("docs/cmodel/PYTORCH_CMODEL_GUIDE.md", 
                                  "C-Model 使用指南"))
    all_checks.append(check_file("docs/cmodel/HARDWARE_CMODEL_GUIDE.md", 
                                  "硬體實作指南"))
    
    # 6. 檢查工具腳本
    print("\n6. 工具腳本")
    print("-"*80)
    all_checks.append(check_file("I-ViT/extract_golden_patterns.py", 
                                  "Golden Patterns 提取"))
    all_checks.append(check_file("I-ViT/generate_rtl_vectors.py", 
                                  "RTL 測試向量生成"))
    all_checks.append(check_file("I-ViT/convert_checkpoint.py", 
                                  "Checkpoint 轉換"))
    
    # 總結
    print("\n" + "="*80)
    print("驗證總結")
    print("="*80)
    
    total = len(all_checks)
    passed = sum(all_checks)
    failed = total - passed
    
    print(f"\n總共檢查: {total} 項")
    print(f"通過: {passed} 項")
    print(f"失敗: {failed} 項")
    
    if failed == 0:
        print("\n✓ 所有文件已正確生成！")
        print("✓ RTL 模擬環境設置完成！")
        print("\n下一步：")
        print("  1. 閱讀 RTL_SIMULATION_READY.md")
        print("  2. 參考 cmodel_rtl_reference/ 中的算法開始 RTL 實作")
        print("  3. 使用 rtl_vectors/ 中的測試向量驗證 RTL 實作")
    else:
        print(f"\n✗ 有 {failed} 個文件缺失")
        print("請執行以下命令生成缺失的文件：")
        if not os.path.exists("golden_patterns/golden_patterns.npz"):
            print("  python I-ViT/extract_golden_patterns.py")
        if not os.path.exists("rtl_vectors/input_image.hex"):
            print("  python I-ViT/generate_rtl_vectors.py")
    
    print("="*80)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
