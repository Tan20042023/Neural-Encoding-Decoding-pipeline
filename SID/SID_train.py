import argparse
import os

import tensorflow as tf
import tf_keras
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import Input
from tf_keras.models import Model

from encoding_common.pipeline import (
    create_or_load_split_indices,
    load_movie_spike_history,
    resize_movies,
    set_seed,
)
from SID.SID import AE, cal_performance, dense_decoder


def _get_tpu_strategy():
    """Detect and initialize TPU. Returns (strategy, is_tpu)."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Running on TPU: {resolver.master()}")
        print(f"Number of replicas: {strategy.num_replicas_in_sync}")
        return strategy, True
    except (ValueError, RuntimeError) as e:
        print(f"TPU initialization failed ({e}), falling back to CPU/GPU")
        return tf.distribute.get_strategy(), False


def plot_training_history(history, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history["mse"], label="MSE")
    axes[0, 1].set_title("MSE")
    axes[0, 1].legend()

    axes[1, 0].plot(history["psnr"], label="PSNR")
    axes[1, 0].set_title("PSNR")
    axes[1, 0].legend()

    axes[1, 1].plot(history["ssim"], label="SSIM")
    axes[1, 1].set_title("SSIM")
    axes[1, 1].legend()

    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def build_sid_models(ncell, resolution, learning_rate):
    input_x = Input(shape=(ncell,), name="spike_input")
    model_dense = dense_decoder(ncell)
    model_ae = AE((resolution, resolution, 1))

    dense_out = model_dense(input_x)
    ae_out = model_ae(dense_out)

    end2end_model = Model(input_x, ae_out, name="SID_end2end")
    end2end_model.compile(loss="mse", optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate))

    multiout_model = Model(input_x, [dense_out, ae_out], name="SID_multiout")
    return end2end_model, multiout_model


def train_SID(
    data_path,
    weight_dir,
    result_dir,
    resolution=64,
    batch_size=256,
    epochs=30,
    num_iter=20,
    learning_rate=1e-3,
    visualization=True,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    split_path=None,
):
    set_seed(seed)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    movie, spike, _ = load_movie_spike_history(data_path, require_history=False)
    if split_path is None:
        split_path = os.path.join(result_dir, "split_indices.npz")
    train_idx, val_idx, test_idx = create_or_load_split_indices(
        num_samples=movie.shape[0],
        test_split=test_split,
        val_split=val_split,
        seed=seed,
        split_path=split_path,
    )

    x_train = spike[train_idx]
    x_val = spike[val_idx]
    x_test = spike[test_idx]
    y_train = resize_movies(movie[train_idx], resolution)
    y_val = resize_movies(movie[val_idx], resolution)
    y_test = resize_movies(movie[test_idx], resolution)

    strategy, is_tpu = _get_tpu_strategy()

    with strategy.scope():
        end2end_model, multiout_model = build_sid_models(
            ncell=x_train.shape[1],
            resolution=resolution,
            learning_rate=learning_rate,
        )

    if is_tpu:
        per_replica_batch = batch_size // strategy.num_replicas_in_sync
        if per_replica_batch < 1:
            per_replica_batch = 1
        batch_size = per_replica_batch * strategy.num_replicas_in_sync
        print(f"TPU batch size: {per_replica_batch} per replica x {strategy.num_replicas_in_sync} replicas = {batch_size} effective")

    checkpoint_path = os.path.join(weight_dir, "SID_best.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=1),
    ]

    history = {
        "train_loss": [],
        "val_loss": [],
        "mse": [],
        "psnr": [],
        "ssim": [],
    }

    for i in range(num_iter):
        iter_history = end2end_model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
        history["train_loss"].extend(iter_history.history["loss"])
        history["val_loss"].extend(iter_history.history["val_loss"])

        _, pred_ae = multiout_model.predict(x_test, batch_size=batch_size, verbose=0)
        mse, psnr, ssim = cal_performance(y_test, pred_ae)
        history["mse"].append(mse)
        history["psnr"].append(psnr)
        history["ssim"].append(ssim)

    plot_training_history(history, os.path.join(result_dir, "SID_training_history.png"))

    return history, end2end_model, multiout_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train SID decoding model")
    parser.add_argument("--data-path", default="Dataset/movie/movie01.mat")
    parser.add_argument("--weight-dir", default="SID/weights/movie/movie01")
    parser.add_argument("--result-dir", default="SID/SID_results/movie/movie01")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-iter", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_SID(
        data_path=args.data_path,
        weight_dir=args.weight_dir,
        result_dir=args.result_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_iter=args.num_iter,
        learning_rate=args.learning_rate,
        visualization=not args.no_visualization,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        split_path=args.split_path,
    )

