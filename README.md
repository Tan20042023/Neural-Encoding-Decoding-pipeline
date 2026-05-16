# Neural Encoding-Decoding Pipeline

A computational neuroscience pipeline for visual stimulus encoding and neural spike decoding, using salamander retina multi-electrode array (MEA) recordings and Allen Brain Observatory data.

## Overview

This project implements and compares multiple encoding models (LN, GLM, CNN) that predict neural spike rates from visual stimuli, and a spike-image decoder (SID) that reconstructs images from spike trains. The full pipeline demonstrates that encoding-then-decoding with predicted spikes can outperform direct decoding, especially under noisy conditions.

## Project Structure

```
├── LN/                 # Linear-Nonlinear encoding model
├── GLM/                # Generalized Linear Model (stimulus + spike history)
├── CNN/                # Convolutional Neural Network encoder
├── SID/                # Spike-Image Decoder (dense + autoencoder)
├── encoding_common/    # Shared pipeline utilities (data loading, splitting, etc.)
├── utils/              # Metrics (Pearson correlation in TF/PyTorch)
├── experiments/        # Controlled experiments
│   ├── implicit_denoising/
│   ├── noisy_spike_robustness/
│   ├── decoding_regularized_encoding/
│   └── neural_digit_twin_transfer/
├── Picture/            # Publication figures and plotting scripts
└── Dataset/            # [NOT INCLUDED] See Data Download below
```

## Data Download

The dataset and pretrained model weights are hosted on Google Drive due to file size limitations.

**Download link:** [Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)

After downloading, place the files so that the directory structure looks like:

```
├── Dataset/
│   ├── movie/
│   │   └── movie01.mat              # Salamander retina data (1800 frames, 90x90)
│   └── allenSDK/
│       └── session_721123822/
│           ├── movie1.mat           # Allen Institute data
│           └── movie1_trial1.mat    # Single-trial variant
├── CNN/weights/                      # CNN encoder weights
├── GLM/weights/                      # GLM weights
├── LN/weights/                       # LN weights
└── SID/weights/                      # SID decoder weights
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Tan20042023/Neural-Encoding-Decoding-pipeline.git
cd Neural-Encoding-Decoding-pipeline

# Download data and weights from Google Drive (see link above)
# Place them in the project root following the directory structure

# Train an encoding model (e.g., CNN)
python CNN/CNN_train.py

# Run predictions
python CNN/CNN_pred.py

# Train the decoder
python SID/SID_train.py

# Run experiments
python experiments/implicit_denoising/run_implicit_denoising.py
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| **LN** | Encoding | Single Dense layer + activation (softplus/sigmoid/relu/exp) |
| **GLM** | Encoding | Two-pathway model: stimulus filter + spike history filter |
| **CNN** | Encoding | Conv2D layers with Gaussian noise, predicts spike rates from frames |
| **SID** | Decoding | Dense decoder + autoencoder, reconstructs 64x64 images from spikes |

## Experiments

1. **Implicit Denoising** - Shows CNN predicted spikes + SID outperforms direct SID decoding on single-trial data
2. **Noisy Spike Robustness** - Tests CNN pred_spike vs real spike under dropout/Gaussian/Poisson noise
3. **Decoding-Regularized Encoding** - Trains CNN with frozen SID loss as regularizer
4. **Neural Digital Twin Transfer** - Transfer learning across movie stimuli

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- SciPy
- Matplotlib

## Citation

If you use this code, please cite: [YOUR CITATION HERE]

## License

[YOUR LICENSE HERE]
