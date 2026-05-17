# Decoding-Regularized Encoding Experiment

[English](README.md)

## 实验动机

实验框架图见 `Picture/model/joint_training.pdf`

思路是在编码模型后加解码器，将解码损失也引入编码模型的训练中。希望达到的效果是能在提升预测脉冲的解码指标同时编码模型自身的相关系数不掉。

在 allensdk 单试次数据下测试，结果解码指标相比 baseline 略微提升，编码指标略微下降，整体效果不明显。

## 对照设置

- **Baseline**：普通 CNN 编码器（movie → spike）
- **Regularized**：解码损失正则化编码器（冻结 SID，引入 decode loss）
- 两者使用同一数据、同一 split、同一 SID 解码器。

## 目录结构

- `configs/default.json`：实验配置
- `core/regularized_encoder.py`：正则化编码器训练
- `run_decoding_regularized_encoding.py`：总控脚本
- `outputs/`：实验产物

## 运行方式

```bash
python experiments/decoding_regularized_encoding/run_decoding_regularized_encoding.py \
    --config experiments/decoding_regularized_encoding/configs/default.json \
    --mode all
```

可选：

- `--mode train`：仅训练
- `--mode eval`：仅评估（需已有模型）
- `--skip-existing`：若模型已存在则跳过训练

## 输出说明

在 `outputs/<case>/` 下生成：

- `baseline_cnn/`：Baseline 编码器模型与预测
- `regularized_cnn/`：Regularized 编码器模型与预测
- `decode_compare/comparison_table.json`：核心对照结果
- `final_report.json`：完整报告与结论检查
