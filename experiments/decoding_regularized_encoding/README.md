# Decoding-Regularized Encoding Experiment

[中文](README_ZH.md)

## Motivation

Framework diagram: `Picture/model/joint_training.pdf`

The idea is to add a decoder after the encoding model and introduce decoding loss into encoder training. The goal is to improve decoding metrics of predicted spikes while maintaining the encoder's own correlation coefficient.

Tested on allensdk single-trial data: decoding metrics improved slightly compared to baseline, while encoding metrics decreased slightly. Overall effect was not significant.

## Control Setup

- **Baseline**: Standard CNN encoder (movie → spike)
- **Regularized**: Decoding-loss regularized encoder (frozen SID, with decode loss)
- Both use the same data, same split, and same SID decoder.

## Structure

- `configs/default.json`: Experiment configuration
- `core/regularized_encoder.py`: Regularized encoder training
- `run_decoding_regularized_encoding.py`: Main script
- `outputs/`: Experiment outputs

## Usage

```bash
python experiments/decoding_regularized_encoding/run_decoding_regularized_encoding.py \
    --config experiments/decoding_regularized_encoding/configs/default.json \
    --mode all
```

Options:

- `--mode train`: Train only
- `--mode eval`: Evaluate only (requires existing models)
- `--skip-existing`: Skip training if model already exists

## Outputs

Generated under `outputs/<case>/`:

- `baseline_cnn/`: Baseline encoder model and predictions
- `regularized_cnn/`: Regularized encoder model and predictions
- `decode_compare/comparison_table.json`: Key comparison results
- `final_report.json`: Full report with conclusion check
