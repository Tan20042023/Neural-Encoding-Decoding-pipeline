# Noisy Spike Robustness Experiment

本实验用于验证 CNN 编码得到的 `pred_spike` 在噪声扰动下的鲁棒性：

1. 对真实 `spike` 和 `pred_spike` 施加**相同强度**噪声（dropout / gaussian / poisson）。
2. 将两组扰动后的 spike 分别输入同一个 SID 解码器。
3. 比较 MSE / PSNR / SSIM 指标随噪声强度变化的趋势。

## 目录结构

- `configs/default.json`：实验配置（数据路径、噪声强度、输出目录）
- `run_noisy_spike_robustness.py`：总控脚本
- `outputs/`：运行产物（分噪声类型结果、汇总图、总报告）

## 运行

在仓库根目录执行：

```bash
python experiments\noisy_spike_robustness\run_noisy_spike_robustness.py --config experiments\noisy_spike_robustness\configs\default.json
```

## 输出

每个 case 在 `outputs/<case_name>/` 下生成：

- `dropout/`, `gaussian/`, `poisson/`
  - 各强度子目录（如 `dropout_0.1`）的 `*_decode_metrics.json`
  - `summary_plots_<noise>/` 汇总曲线图与 `aggregated_metrics.json`
- `case_report.json`

总报告：

- `outputs/final_report.json`

