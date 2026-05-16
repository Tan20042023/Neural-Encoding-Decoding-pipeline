# Implicit Denoising Experiment

本目录用于复现论文中“编码模型隐式降噪”相关实验，统一了多试次与单试次验证流程。

## 实验结构

- `configs/default.json`：默认实验配置
- `run_implicit_denoising.py`：总控脚本
- `outputs/`：运行产物（模型权重、中间结果、指标报告）

## 实验分组（按你的实际数据使用情况）

1. **多试次实验（跑 2 次）**
   - `multi_trial_movie`：`Dataset\movie\movie01.mat`
   - `multi_trial_allensdk`：`Dataset\allenSDK\session_721123822\movie1.mat`

2. **单试次验证（跑 1 次）**
   - `single_trial_allensdk`：`Dataset\allenSDK\session_721123822\single_trial\movie1_trial1.mat`

## 运行方式

在仓库根目录执行：

```bash
python experiments\implicit_denoising\run_implicit_denoising.py --config experiments\implicit_denoising\configs\default.json --mode all
```

常用模式：

- `--mode all`：训练并评估（完整流程）
- `--mode train`：只训练
- `--mode eval`：只评估（需已有模型）
- `--skip-existing`：若已有 `*_best.keras` 则跳过训练阶段

## 输出说明

每个实验会在 `outputs/<experiment_name>/` 下生成：

- `split_indices.npz`：固定划分索引（保证对比一致）
- `cnn/weights/`、`cnn/results/`：CNN 编码器产物
- `sid/weights/`、`sid/results/`：SID 解码器产物
- `comparison/`：编码-解码链路结果（CNN 预测脉冲 -> SID）
- `metrics_summary.json`：该实验总指标汇总

总报告：

- `outputs/final_report.json`

其中 `comparison` 字段包含：

- `mse_improvement_abs/pct`
- `psnr_improvement_abs`
- `ssim_improvement_abs`

用于直接对应论文中“编码-解码优于直接解码”的结论，以及单试次提升更明显的验证结果。

