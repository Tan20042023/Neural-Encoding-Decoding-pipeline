# Implicit Denoising Experiment

[中文](README_ZH.md)

## Design

Biological spikes contain both visually-driven activity and irregular spontaneous activity. We found that predicted spikes from encoding models outperform real biological spikes in decoding, even when encoding correlation is not high.

We hypothesize that during training, encoding models focus on fitting the structured visually-driven activity while ignoring irregular fluctuations. This implicitly achieves a "denoising" effect.

To verify this, we conducted experiments on single-trial data with stronger spontaneous activity. The decoding improvement from encoded spikes was more pronounced, confirming our hypothesis.

## Structure

- `configs/default.json`: Default experiment configuration
- `run_implicit_denoising.py`: Main script
- `outputs/`: Experiment outputs (model weights, intermediate results, metrics)

## Experiment Groups

1. **Multi-trial experiments**
   - `multi_trial_movie`: `Dataset/movie/movie01.mat`
   - `multi_trial_allensdk`: `Dataset/allenSDK/session_721123822/movie1.mat`

2. **Single-trial validation**
   - `single_trial_allensdk`: `Dataset/allenSDK/session_721123822/single_trial/movie1_trial1.mat`

## Usage

```bash
python experiments/implicit_denoising/run_implicit_denoising.py \
    --config experiments/implicit_denoising/configs/default.json \
    --mode all
```

Options:

- `--mode all`: Train and evaluate (full pipeline)
- `--mode train`: Train only
- `--mode eval`: Evaluate only (requires existing models)
- `--skip-existing`: Skip training if `*_best.keras` exists

## Outputs

Each experiment generates under `outputs/<experiment_name>/`:

- `split_indices.npz`: Fixed split indices (ensures consistent comparison)
- `cnn/`: CNN encoder weights and predictions
- `sid/`: SID decoder weights and reconstructions
- `comparison/`: Encoding-decoding pipeline results (CNN predicted spikes → SID)
- `metrics_summary.json`: Summary metrics for the experiment

Final report: `outputs/final_report.json`

The `comparison` field contains `mse_improvement_abs/pct`, `psnr_improvement_abs`, `ssim_improvement_abs`, corresponding to the conclusion that encoding-decoding outperforms direct decoding.
