# Neural Encoding-Decoding Pipeline

In previous visual neural encoding-decoding research, encoding and decoding have been relatively isolated. This work attempts to combine both into a unified framework, exploring the properties and applications of encoding and decoding models.

Data sources: Salamander retina multi-electrode array (MEA) recordings and Allen Brain Observatory data.

[中文](README_ZH.md)

## Data & Weights

The dataset and pretrained model weights are hosted on Google Drive:

**Download link:** [Google Drive](https://drive.google.com/drive/folders/1PwT9dmvmI2T3dxHGSP2VAOUqdyouwyh1?usp=drive_link)

Expected directory structure after downloading:

```
├── Dataset/
│   ├── movie/
│   │   └── movie01.mat                    # Salamander retina data (1800 frames, 90×90)
│   └── allenSDK/
│       └── session_721123822/
│           ├── movie1.mat                 # Allen Institute data
│           ├── movie1_trial1.mat          # Single-trial variant
│           └── movie3.mat                 # Transfer learning source data
└── weights/
    ├── CNN/                               # CNN encoder weights
    ├── SID/                               # SID decoder weights
    └── Neural_digit_twin/                 # Transfer experiment weights
```

## Environment Setup

- Python 3.9+
- Experiments conducted on Google TPU v5litepod-1

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── LN/                     # Linear-Nonlinear encoding model (reference only)
├── GLM/                    # Generalized Linear Model (reference only)
├── CNN/                    # Convolutional Neural Network encoder
├── SID/                    # Spike-Image Decoder
├── encoding_common/        # Shared utilities (data loading, splitting, etc.)
├── utils/                  # Metrics (Pearson correlation)
├── experiments/            # Experiment directory
│   ├── implicit_denoising/
│   ├── decoding_regularized_encoding/
│   └── neural_digit_twin_transfer/
├── Picture/                # Publication figures and plotting scripts
└── Dataset/                # [NOT INCLUDED] See Data & Weights above
```

**Note:** LN/GLM models and noisy_spike_robustness experiment were not used in the final experiments, provided for reference only.

## Experiments

### 1. Implicit Denoising

**Directory:** `experiments/implicit_denoising/`

Demonstrates the implicit denoising capability of encoding models: encoded spikes decoded through the decoder outperform direct decoding from raw biological spikes, with more significant improvement on single-trial data.

```bash
python experiments/implicit_denoising/run_implicit_denoising.py \
    --config experiments/implicit_denoising/configs/default.json \
    --mode all
```

### 2. Decoding-Regularized Encoding

**Directory:** `experiments/decoding_regularized_encoding/`

Attempts to introduce decoding loss into encoder training (freezing SID, adding decode loss as regularization). Results were modest.

```bash
python experiments/decoding_regularized_encoding/run_decoding_regularized_encoding.py \
    --config experiments/decoding_regularized_encoding/configs/default.json \
    --mode all
```

### 3. Neural Digit Twin Transfer

**Directory:** `experiments/neural_digit_twin_transfer/`

Uses a lightweight encoder to generate pseudo-spike data on movie3, assisting decoder transfer learning on movie1. Compares decoding performance across three training strategies.

```bash
python experiments/neural_digit_twin_transfer/run_neural_digit_twin_transfer.py \
    --config experiments/neural_digit_twin_transfer/configs/default.json \
    --mode all
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| **LN** | Encoding | Single Dense layer + activation function (reference only) |
| **GLM** | Encoding | Two-pathway model: stimulus filter + spike history filter (reference only) |
| **CNN** | Encoding | Conv2D layers with Gaussian noise, predicts spike firing rate from visual frames |
| **SID** | Decoding | Dense decoder + autoencoder, reconstructs 64×64 images from spikes |

## License

This project is licensed under the [MIT License](LICENSE).
