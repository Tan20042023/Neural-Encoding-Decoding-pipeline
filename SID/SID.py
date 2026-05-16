import numpy as np
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Reshape,
    UpSampling2D,
)
from keras.models import Sequential
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def dense_decoder(ncell):
    model = Sequential(name="dense")
    model.add(Dense(512, input_dim=ncell))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation("sigmoid"))
    model.add(Reshape(target_shape=(64, 64, 1), name="dense_out"))
    return model


def AE(input_shape):
    model = Sequential(name="ae")

    model.add(Conv2D(64, (7, 7), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (7, 7), strides=(1, 1), padding="same"))
    model.add(BatchNormalization(name="ae_out"))
    return model


def cal_performance(src_imgs, dst_imgs):
    src_imgs = src_imgs.astype("float32")
    dst_imgs = dst_imgs.astype("float32")

    img_num = src_imgs.shape[0]
    all_mse = np.zeros(img_num, dtype=np.float32)
    all_psnr = np.zeros(img_num, dtype=np.float32)
    all_ssim = np.zeros(img_num, dtype=np.float32)

    for i in range(img_num):
        all_mse[i] = mean_squared_error(src_imgs[i], dst_imgs[i])
        all_psnr[i] = peak_signal_noise_ratio(src_imgs[i], dst_imgs[i])
        all_ssim[i] = structural_similarity(
            src_imgs[i],
            dst_imgs[i],
            win_size=None,
            data_range=1.0,
            channel_axis=2,
        )

    return float(np.mean(all_mse)), float(np.mean(all_psnr)), float(np.mean(all_ssim))

