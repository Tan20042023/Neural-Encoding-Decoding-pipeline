import argparse
import os

import tensorflow as tf
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import Input
from tf_keras.optimizers import Adam

from CNN.CNN import CNN
from encoding_common.pipeline import (
    create_or_load_split_indices,
    load_movie_spike_history,
    plot_training_history,
    resize_movies,
    set_seed,
)
from utils.metrics import keras_cc


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


def train_CNN(
    data_path,
    weight_dir,
    result_dir,
    batch_size=256,
    epochs=500,
    learning_rate=0.001,
    resolution=64,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    split_path=None,
    visualization=True,
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

    x_train = resize_movies(movie[train_idx], resolution)
    x_val = resize_movies(movie[val_idx], resolution)
    y_train = spike[train_idx]
    y_val = spike[val_idx]

    strategy, is_tpu = _get_tpu_strategy()

    with strategy.scope():
        model = CNN(inputs=Input(shape=x_train.shape[1:]), n_out=y_train.shape[1])
        model.compile(optimizer=Adam(learning_rate), loss="poisson", metrics=["mse", keras_cc])

    best_path = os.path.join(weight_dir, "CNN_best.keras")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=30, verbose=1),
    ]

    if is_tpu:
        per_replica_batch = batch_size // strategy.num_replicas_in_sync
        if per_replica_batch < 1:
            per_replica_batch = 1
        effective_batch = per_replica_batch * strategy.num_replicas_in_sync
        print(f"TPU batch size: {per_replica_batch} per replica x {strategy.num_replicas_in_sync} replicas = {effective_batch} effective")
    else:
        effective_batch = batch_size

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=effective_batch,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = history.history
    plot_training_history(history_dict, os.path.join(result_dir, "CNN_training_history.png"))
    return model, history_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN encoding model")
    parser.add_argument("--data-path", default="Dataset/movie/movie01.mat")
    parser.add_argument("--weight-dir", default="CNN/weights/movie/movie01")
    parser.add_argument("--result-dir", default="CNN/results/movie/movie01")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_CNN(
        data_path=args.data_path,
        weight_dir=args.weight_dir,
        result_dir=args.result_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        split_path=args.split_path,
        visualization=not args.no_visualization,
    )

