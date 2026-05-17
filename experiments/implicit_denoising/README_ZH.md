# Implicit Denoising Experiment

[English](README.md)

## 实验设计

生物脉冲中包含由视觉信号直接驱动的活动以及无规律的随机活动。我们发现编码模型的预测脉冲即使在编码相关系数不高的情况下，其解码效果仍优于真实生物脉冲的解码效果。

我们猜测编码模型在训练中为了尽可能提升拟合效果，会尽力去拟合有规律的视觉信号直接驱动的活动，而忽略无规律的随机波动。因此编码模型的预测脉冲隐式地达到了一种"去噪"的效果。

为了进一步验证这一推论，我们使用了随机活动更强的单试次数据进行实验，发现编码脉冲的解码提升效果更明显，验证了上述推论。

## 实验结构

- `configs/default.json`：默认实验配置
- `run_implicit_denoising.py`：总控脚本
- `outputs/`：运行产物（模型权重、中间结果、指标报告）

## 实验分组

1. **多试次实验**
   - `multi_trial_movie`：`Dataset/movie/movie01.mat`
   - `multi_trial_allensdk`：`Dataset/allenSDK/session_721123822/movie1.mat`

2. **单试次验证**
   - `single_trial_allensdk`：`Dataset/allenSDK/session_721123822/single_trial/movie1_trial1.mat`

## 运行方式

```bash
python experiments/implicit_denoising/run_implicit_denoising.py \
    --config experiments/implicit_denoising/configs/default.json \
    --mode all
```

常用模式：

- `--mode all`：训练并评估（完整流程）
- `--mode train`：只训练
- `--mode eval`：只评估（需已有模型）
- `--skip-existing`：若已有 `*_best.keras` 则跳过训练阶段

## 输出说明

每个实验会在 `outputs/<experiment_name>/` 下生成：

- `split_indices.npz`：固定划分索引（保证对比一致）
- `cnn/`：CNN 编码器权重与预测结果
- `sid/`：SID 解码器权重与重建结果
- `comparison/`：编码-解码链路结果（CNN 预测脉冲 → SID）
- `metrics_summary.json`：该实验总指标汇总

总报告：`outputs/final_report.json`

其中 `comparison` 字段包含 `mse_improvement_abs/pct`、`psnr_improvement_abs`、`ssim_improvement_abs`，用于直接对应"编码-解码优于直接解码"的结论。
