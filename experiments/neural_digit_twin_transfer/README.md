# Neural Digit Twin Transfer Experiment

[中文](README_ZH.md)

## Motivation

Decoding tasks reconstruct images from spikes. Obviously, a decoder cannot produce valid images if it has never seen data from the same or similar distribution as the target. Therefore, the decoder should be exposed to as diverse data as possible during training. However, spike data is scarce and expensive to collect. Meanwhile, encoders require less training data and exhibit better cross-domain generalization.

Our approach: train a lightweight encoder on limited data to synthesize "pseudo-spike" data, expanding the decoder's training set and improving cross-domain transfer decoding performance.

Framework diagram: `Picture/model/transfer.pdf`

## Experiment Design

Three groups of decoders are trained and compared on movie1 test data:

| Group | Training Data | Strategy |
|-------|--------------|----------|
| **Group 1** | movie3 only | Baseline |
| **Group 2** | movie3 + movie1 real | Direct combination |
| **Group 3** | movie3 + pseudo-spike → finetune on movie1 real | Pretrain then finetune |

Expected result: Group 3 > Group 2 > Group 1

## Structure

- `run_neural_digit_twin_transfer.py`: Main entry point
- `core/data.py`: Data loading, resize, split utilities
- `core/models.py`: Encoder/decoder training and evaluation
- `configs/default.json`: Default configuration
- `outputs/`: Experiment outputs

## Usage

```bash
python experiments/neural_digit_twin_transfer/run_neural_digit_twin_transfer.py \
    --config experiments/neural_digit_twin_transfer/configs/default.json \
    --mode all
```

Options:

- `--mode train`: Train only
- `--mode eval`: Evaluate only (requires existing models)
- `--skip-existing`: Skip training if model already exists

## Outputs

Default output directory: `experiments/neural_digit_twin_transfer/outputs/session_721123822/movie3_to_movie1_transfer`

Key files:

- `split_indices.npz`: movie1 real / unlabeled / test split indices
- `split_info.json`: Split summary
- `pseudo_spike.mat`: Pseudo spikes for movie1 unlabeled data
- `group1_movie3_only/metrics.json`
- `group2_movie3_plus_movie1_real/metrics.json`
- `group3_finetune/metrics.json`
- `final_report.json`: Summary, ranking, and `3>2>1` validation
