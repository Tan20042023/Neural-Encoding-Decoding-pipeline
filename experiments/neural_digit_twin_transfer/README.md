# Neural Digit Twin Transfer Experiment

本实验对应论文中的解码迁移部分，目标是在 `movie3` 与少量 `movie1` 真实标注上训练编码器生成伪脉冲，再比较 3 组解码训练策略在 `movie1 test` 上的表现。

## 目录

- `run_neural_digit_twin_transfer.py`：实验总控入口
- `core/data.py`：数据加载、resize、split 复用
- `core/models.py`：encoder / decoder 训练与评估
- `configs/default.json`：默认配置
- `outputs/`：实验产物（建议不入库）

## 运行

```bash
python experiments\neural_digit_twin_transfer\run_neural_digit_twin_transfer.py --config experiments\neural_digit_twin_transfer\configs\default.json --mode all
```

可选：

- `--mode train`：仅训练
- `--mode eval`：仅评估（需已有模型）
- `--skip-existing`：若模型已存在则跳过训练

## 输出

默认输出目录：`experiments\neural_digit_twin_transfer\outputs\session_721123822\movie3_to_movie1_transfer`

关键文件：

- `split_indices.npz`：movie1 real / unlabeled / test 划分索引
- `split_info.json`：划分摘要
- `pseudo_spike.mat`：movie1 unlabeled 的伪 spike
- `group1_movie3_only\metrics.json`
- `group2_movie3_plus_movie1_real\metrics.json`
- `group3_finetune\metrics.json`
- `final_report.json`：三组汇总、排序与 `3>2>1` 检查

