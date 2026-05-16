# Decoding-Regularized Encoding Experiment

本实验验证：在编码器训练中引入解码约束后，`pred_spike` 经 SID 解码的重建指标优于普通 CNN 编码器。

## 对照设置

- **Baseline**：普通 CNN 编码器（movie -> spike）
- **Regularized**：解码损失正则化编码器（冻结 SID，引入 decode loss）
- 两者使用同一数据、同一 split、同一 SID 解码器。

## 目录

- `configs/default.json`：实验配置
- `core/regularized_encoder.py`：正则化编码器训练
- `run_decoding_regularized_encoding.py`：总控脚本
- `outputs/`：实验产物

## 运行

```bash
python experiments\decoding_regularized_encoding\run_decoding_regularized_encoding.py --config experiments\decoding_regularized_encoding\configs\default.json --mode all
```

可选：

- `--mode train`：仅训练
- `--mode eval`：仅评估（需已有模型）
- `--skip-existing`：若模型已存在则跳过训练

## 输出

在 `outputs/<case>/` 下生成：

- `baseline_cnn/`：Baseline 编码器模型与预测
- `regularized_cnn/`：Regularized 编码器模型与预测
- `decode_compare/comparison_table.json`：核心对照结果
- `final_report.json`：完整报告与结论检查

