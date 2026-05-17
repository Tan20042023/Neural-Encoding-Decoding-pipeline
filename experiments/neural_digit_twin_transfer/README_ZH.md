# Neural Digit Twin Transfer Experiment

[English](README.md)

## 实验动机

解码任务是从脉冲重建图像。显而易见，如果解码器预先没有见过与目标图像同分布或相似分布的图像，是无法解码出有效的图像的。因此在训练时希望解码器能够见过尽量多分布的数据。但是可用于训练的脉冲数据非常稀缺而且采集成本高昂，同时编码器训练所需的数据较少，且具有更好的跨域泛化性。

因此我们的思路是让在少量数据上训练的编码器合成"伪脉冲"数据，扩充解码器的训练集，提升解码器的跨域迁移解码性能。

实验框架见 `Picture/model/transfer.pdf`

## 实验设计

设计三组解码器，在 movie1 test 数据上比较：

| 组别 | 训练数据 | 策略 |
|------|----------|------|
| **Group 1** | 仅 movie3 | Baseline |
| **Group 2** | movie3 + movie1 real | 直接组合 |
| **Group 3** | movie3 + 伪脉冲 → movie1 real 微调 | 预训练后微调 |

预期结果：Group 3 > Group 2 > Group 1

## 目录结构

- `run_neural_digit_twin_transfer.py`：实验总控入口
- `core/data.py`：数据加载、resize、split 工具
- `core/models.py`：encoder / decoder 训练与评估
- `configs/default.json`：默认配置
- `outputs/`：实验产物

## 运行方式

```bash
python experiments/neural_digit_twin_transfer/run_neural_digit_twin_transfer.py \
    --config experiments/neural_digit_twin_transfer/configs/default.json \
    --mode all
```

可选：

- `--mode train`：仅训练
- `--mode eval`：仅评估（需已有模型）
- `--skip-existing`：若模型已存在则跳过训练

## 输出说明

默认输出目录：`experiments/neural_digit_twin_transfer/outputs/session_721123822/movie3_to_movie1_transfer`

关键文件：

- `split_indices.npz`：movie1 real / unlabeled / test 划分索引
- `split_info.json`：划分摘要
- `pseudo_spike.mat`：movie1 unlabeled 的伪 spike
- `group1_movie3_only/metrics.json`
- `group2_movie3_plus_movie1_real/metrics.json`
- `group3_finetune/metrics.json`
- `final_report.json`：三组汇总、排序与 `3>2>1` 检查
