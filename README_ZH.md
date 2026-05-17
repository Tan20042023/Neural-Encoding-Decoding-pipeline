# Neural Encoding-Decoding Pipeline

以往的视觉神经编解码研究中，编码和解码相对孤立。本工作尝试将两者结合到一个统一框架下，探究编码模型和解码模型的性质与应用。

数据来源：蝾螈视网膜多电极阵列（MEA）记录和 Allen Brain Observatory 数据。

[English](README.md)

## 数据与权重

数据集和预训练模型权重托管在 Google Drive：

**下载地址：** [Google Drive](https://drive.google.com/drive/folders/1PwT9dmvmI2T3dxHGSP2VAOUqdyouwyh1?usp=drive_link)

下载后的目录结构：

```
├── Dataset/
│   ├── movie/
│   │   └── movie01.mat                    # 蝾螈视网膜数据（1800 帧，90×90）
│   └── allenSDK/
│       └── session_721123822/
│           ├── movie1.mat                 # Allen Institute 数据
│           ├── movie1_trial1.mat          # 单试次变体
│           └── movie3.mat                 # 迁移学习源数据
└── weights/
    ├── CNN/                               # CNN 编码器权重
    ├── SID/                               # SID 解码器权重
    └── Neural_digit_twin/                 # 迁移实验权重
```

## 环境配置

- Python 3.9+
- 实验在 Google TPU v5litepod-1 上完成

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── LN/                     # 线性-非线性编码模型（仅供参考）
├── GLM/                    # 广义线性模型（仅供参考）
├── CNN/                    # 卷积神经网络编码器
├── SID/                    # 脉冲-图像解码器
├── encoding_common/        # 共享工具（数据加载、划分等）
├── utils/                  # 指标计算（Pearson 相关系数）
├── experiments/            # 实验目录
│   ├── implicit_denoising/
│   ├── decoding_regularized_encoding/
│   └── neural_digit_twin_transfer/
├── Picture/                # 论文图表和绘图脚本
└── Dataset/                # [不包含] 见上方数据下载
```

**说明：** LN/GLM 模型和 noisy_spike_robustness 实验在最终实验中未被启用，仅供参考。

## 实验

### 1. Implicit Denoising（隐式降噪）

**对应目录：** `experiments/implicit_denoising/`

验证编码模型的隐式降噪能力：编码后的脉冲经解码器重建，效果优于直接使用真实生物脉冲解码，尤其在单试次数据上提升更明显。

```bash
python experiments/implicit_denoising/run_implicit_denoising.py \
    --config experiments/implicit_denoising/configs/default.json \
    --mode all
```

### 2. Decoding-Regularized Encoding（解码正则化编码）

**对应目录：** `experiments/decoding_regularized_encoding/`

尝试将解码损失引入编码器训练过程（冻结 SID，引入 decode loss 作为正则化）。实验效果一般。

```bash
python experiments/decoding_regularized_encoding/run_decoding_regularized_encoding.py \
    --config experiments/decoding_regularized_encoding/configs/default.json \
    --mode all
```

### 3. Neural Digit Twin Transfer（神经数字孪生迁移）

**对应目录：** `experiments/neural_digit_twin_transfer/`

通过轻量级编码器在 movie3 上生成伪脉冲数据，辅助解码模型在 movie1 上的迁移学习。比较三种训练策略的解码效果。

```bash
python experiments/neural_digit_twin_transfer/run_neural_digit_twin_transfer.py \
    --config experiments/neural_digit_twin_transfer/configs/default.json \
    --mode all
```

## 模型

| 模型 | 类型 | 说明 |
|------|------|------|
| **LN** | 编码 | 单层 Dense + 激活函数（仅供参考） |
| **GLM** | 编码 | 双通路模型：刺激滤波器 + 脉冲历史滤波器（仅供参考） |
| **CNN** | 编码 | Conv2D 层 + 高斯噪声，从视觉帧预测脉冲发放率 |
| **SID** | 解码 | Dense 解码器 + 自编码器，从脉冲重建 64×64 图像 |

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。
