import os

import tensorflow as tf
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import Input
from tf_keras.models import Model, load_model
from tf_keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from CNN.CNN import CNN
from SID.SID import AE, cal_performance, dense_decoder
from encoding_common.pipeline import save_json
from utils.metrics import keras_cc

_tpu_strategy = None


def _get_tpu_strategy():
    """Detect and initialize TPU. Returns (strategy, is_tpu). Cached after first call."""
    global _tpu_strategy
    if _tpu_strategy is not None:
        return _tpu_strategy, True
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Running on TPU: {resolver.master()}")
        print(f"Number of replicas: {strategy.num_replicas_in_sync}")
        _tpu_strategy = strategy
        return strategy, True
    except (ValueError, RuntimeError) as e:
        print(f"TPU initialization failed ({e}), falling back to CPU/GPU")
        return tf.distribute.get_strategy(), False


def build_decoder_model(n_cell, resolution, learning_rate):
    spike_input = Input(shape=(n_cell,), name="spike_input")
    dense_model = dense_decoder(n_cell)
    ae_model = AE((resolution, resolution, 1))

    dense_out = dense_model(spike_input)
    recon_out = ae_model(dense_out)

    model = Model(spike_input, recon_out, name="SID_transfer_decoder")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_encoder(
    movie,
    spike,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    val_split,
    seed,
):
    os.makedirs(output_dir, exist_ok=True)
    indices = range(movie.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed)

    x_train = movie[train_idx]
    y_train = spike[train_idx]
    x_val = movie[val_idx]
    y_val = spike[val_idx]

    strategy, is_tpu = _get_tpu_strategy()

    with strategy.scope():
        encoder = CNN(inputs=Input(shape=movie.shape[1:]), n_out=spike.shape[1])
        encoder.compile(optimizer=Adam(learning_rate=learning_rate), loss="poisson", metrics=["mse", keras_cc])

    if is_tpu:
        per_replica = batch_size // strategy.num_replicas_in_sync
        batch_size = max(per_replica, 1) * strategy.num_replicas_in_sync

    best_path = os.path.join(output_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True, verbose=1),
    ]

    history = encoder.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    save_json(os.path.join(output_dir, "history.json"), history.history)
    return best_path, history.history


def train_decoder(
    spike,
    movie,
    output_dir,
    model_name,
    epochs,
    batch_size,
    learning_rate,
    val_split,
    seed,
    init_model_path=None,
):
    os.makedirs(output_dir, exist_ok=True)
    indices = range(spike.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed)

    x_train = spike[train_idx]
    y_train = movie[train_idx]
    x_val = spike[val_idx]
    y_val = movie[val_idx]

    strategy, is_tpu = _get_tpu_strategy()

    if init_model_path is None:
        with strategy.scope():
            model = build_decoder_model(n_cell=spike.shape[1], resolution=movie.shape[1], learning_rate=learning_rate)
    else:
        with strategy.scope():
            model = load_model(init_model_path)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    if is_tpu:
        per_replica = batch_size // strategy.num_replicas_in_sync
        batch_size = max(per_replica, 1) * strategy.num_replicas_in_sync

    best_path = os.path.join(output_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", mode="min", patience=20, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    save_json(os.path.join(output_dir, "history.json"), history.history)
    save_json(
        os.path.join(output_dir, "meta.json"),
        {
            "model_name": model_name,
            "init_model_path": init_model_path,
        },
    )
    return best_path, history.history


def evaluate_decoder(decoder_path, test_spike, test_movie, batch_size):
    model = load_model(decoder_path)
    pred_movie = model.predict(test_spike, batch_size=batch_size, verbose=0)
    mse, psnr, ssim = cal_performance(test_movie, pred_movie)
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }

