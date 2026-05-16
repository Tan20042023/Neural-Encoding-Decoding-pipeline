import json
import os

import cv2
import numpy as np
import scipy.io
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_movie_spike_history(data_path, require_history=False):
    data = scipy.io.loadmat(data_path)
    if "movie" not in data:
        raise KeyError("Missing key 'movie' in MAT file")

    movie = data["movie"].astype(np.float32)
    if "spike" in data:
        spike = data["spike"].astype(np.float32)
    elif "spike_single" in data:
        spike = data["spike_single"].T.astype(np.float32)
    else:
        raise KeyError("Missing key 'spike' or 'spike_single' in MAT file")

    if movie.shape[0] != spike.shape[0]:
        raise ValueError(
            "Sample count mismatch: "
            f"movie={movie.shape[0]}, spike={spike.shape[0]}"
        )

    history = None
    if require_history:
        if "history" not in data:
            raise KeyError("Missing key 'history' in MAT file for GLM")
        history = data["history"].astype(np.float32)
        if history.shape[0] != movie.shape[0]:
            raise ValueError(
                "Sample count mismatch: "
                f"movie={movie.shape[0]}, history={history.shape[0]}"
            )

    return movie, spike, history


def resize_movies(movies, resolution):
    resized = np.zeros((movies.shape[0], resolution, resolution), dtype=np.float32)
    for i in range(movies.shape[0]):
        resized[i] = cv2.resize(movies[i], (resolution, resolution))
    return np.expand_dims(resized, axis=-1).astype(np.float32)


def create_or_load_split_indices(
    num_samples,
    test_split=0.1,
    val_split=0.1,
    seed=42,
    split_path=None,
):
    if split_path and os.path.exists(split_path):
        split = np.load(split_path)
        saved_num = int(split["num_samples"][0])
        if saved_num != int(num_samples):
            raise ValueError(
                f"Split num_samples mismatch: expected {num_samples}, got {saved_num}"
            )
        return split["train_idx"], split["val_idx"], split["test_idx"]

    indices = np.arange(num_samples)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_split,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),
        random_state=seed,
    )

    if split_path:
        split_dir = os.path.dirname(split_path)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
        np.savez(
            split_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            num_samples=np.array([num_samples], dtype=np.int64),
            seed=np.array([seed], dtype=np.int64),
            test_split=np.array([test_split], dtype=np.float32),
            val_split=np.array([val_split], dtype=np.float32),
        )

    return train_idx, val_idx, test_idx


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["mse"], label="Train MSE")
    axes[1].plot(history["val_mse"], label="Val MSE")
    axes[1].set_title("MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["keras_cc"], label="Train CC")
    axes[2].plot(history["val_keras_cc"], label="Val CC")
    axes[2].set_title("Correlation Coefficient")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_predictions_mat(path, movie, spike, pred_spike, frame_indices):
    scipy.io.savemat(
        path,
        {
            "movie": movie,
            "spike": spike,
            "pred_spike": pred_spike,
            "frame_indices": frame_indices,
        },
    )


def save_json(path, data):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

