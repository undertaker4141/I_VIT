# Pattern Extraction Implementation Tasks

## 腳本開發
- [x] `quick_qat.py` - 1 Epoch QAT 校準腳本
- [x] `extract_patterns.py` - Pattern 抽取主程式
- [x] `verify_patterns.py` - 驗證腳本

## quick_qat.py 功能
- [x] 載入 DeiT-Tiny pretrained 模型
- [x] 關閉 Mixup/CutMix
- [x] 執行 1 Epoch 訓練
- [x] 儲存校準後的 checkpoint

## extract_patterns.py 功能
- [x] 載入 checkpoint
- [x] 註冊 forward hooks
- [x] 抽取 cls_token, pos_embed
- [x] 抽取 LayerNorm weight/bias_integer
- [x] 抽取 QuantLinear weight_integer/bias_integer
- [x] 計算 M/S 格式的 scale
- [x] 儲存 golden outputs (INT + Float)
- [x] 輸出 Hex TXT 格式

## 驗證
- [x] 測試 quick_qat.py 執行 (Acc@1: 73.35%)
- [x] 測試 extract_patterns.py 輸出 (1829 files)
- [x] 驗證輸出格式正確
