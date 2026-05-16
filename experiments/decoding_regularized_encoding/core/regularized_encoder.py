import json
import os

import numpy as np
import scipy.io
import tensorflow as tf
import tf_keras
from tf_keras.callbacks import EarlyStopping
from tf_keras.layers import Input
from tf_keras.models import load_model
from tf_keras.optimizers import Adam

from CNN.CNN import CNN
from SID.SID import cal_performance
from encoding_common.pipeline import create_or_load_split_indices, resize_movies, set_seed
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


def compute_mean_cc_numpy(y_true, y_pred):
    x = y_true - np.mean(y_true, axis=0, keepdims=True)
    y = y_pred - np.mean(y_pred, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    eps = 1e-8
    corr = np.mean(x * y, axis=0, keepdims=True) / (np.maximum(x_std, eps) * np.maximum(y_std, eps))
    return float(np.mean(corr))


def load_single_trial_data(data_path):
    data = scipy.io.loadmat(data_path)
    if "movie" not in data:
        raise KeyError("Expected key 'movie' in data file")
    movie = data["movie"].astype(np.float32)
    if "spike" in data:
        spike = data["spike"].astype(np.float32)
    elif "spike_single" in data:
        spike = data["spike_single"].T.astype(np.float32)
    else:
        raise KeyError("Expected 'spike' or 'spike_single' in data file")
    if movie.shape[0] != spike.shape[0]:
        raise ValueError(f"Sample mismatch: movie={movie.shape[0]}, spike={spike.shape[0]}")
    return movie, spike


class DecoderRegularizedEncoder(tf_keras.Model):
    def __init__(self, encoder, decoder, lambda_decode=0.2, lambda_ssim=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_decode = lambda_decode
        self.lambda_ssim = lambda_ssim

        self.total_loss_tracker = tf_keras.metrics.Mean(name="total_loss")
        self.encode_loss_tracker = tf_keras.metrics.Mean(name="encode_loss")
        self.decode_mse_tracker = tf_keras.metrics.Mean(name="decode_mse")
        self.decode_ssim_tracker = tf_keras.metrics.Mean(name="decode_ssim")
        self.cc_tracker = tf_keras.metrics.Mean(name="keras_cc")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.encode_loss_tracker,
            self.decode_mse_tracker,
            self.decode_ssim_tracker,
            self.cc_tracker,
        ]

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def _compute_losses(self, movie_input, spike_target, movie_target, training):
        pred_spike = self.encoder(movie_input, training=training)
        pred_movie = self.decoder(pred_spike, training=False)

        encode_loss = tf.reduce_mean(tf_keras.losses.poisson(spike_target, pred_spike))
        decode_mse = tf.reduce_mean(tf_keras.losses.mse(movie_target, pred_movie))
        decode_ssim = 1.0 - tf.reduce_mean(tf.image.ssim(movie_target, pred_movie, max_val=1.0))

        total_loss = encode_loss + self.lambda_decode * decode_mse + self.lambda_ssim * decode_ssim
        if self.encoder.losses:
            total_loss += tf.add_n(self.encoder.losses)
        return total_loss, encode_loss, decode_mse, decode_ssim, pred_spike

    def train_step(self, data):
        movie_input, targets = data
        spike_target, movie_target = targets
        with tf.GradientTape() as tape:
            total_loss, encode_loss, decode_mse, decode_ssim, pred_spike = self._compute_losses(
                movie_input, spike_target, movie_target, training=True
            )
        grads = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.encode_loss_tracker.update_state(encode_loss)
        self.decode_mse_tracker.update_state(decode_mse)
        self.decode_ssim_tracker.update_state(decode_ssim)
        self.cc_tracker.update_state(keras_cc(spike_target, pred_spike))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        movie_input, targets = data
        spike_target, movie_target = targets
        total_loss, encode_loss, decode_mse, decode_ssim, pred_spike = self._compute_losses(
            movie_input, spike_target, movie_target, training=False
        )
        self.total_loss_tracker.update_state(total_loss)
        self.encode_loss_tracker.update_state(encode_loss)
        self.decode_mse_tracker.update_state(decode_mse)
        self.decode_ssim_tracker.update_state(decode_ssim)
        self.cc_tracker.update_state(keras_cc(spike_target, pred_spike))
        return {m.name: m.result() for m in self.metrics}


def train_regularized_encoder(
    data_path,
    sid_model_path,
    weight_dir,
    result_dir,
    split_path,
    resolution=64,
    batch_size=128,
    epochs=300,
    learning_rate=1e-3,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    lambda_decode=0.2,
    lambda_ssim=0.0,
):
    set_seed(seed)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    movie, spike = load_single_trial_data(data_path)
    train_idx, val_idx, test_idx = create_or_load_split_indices(
        num_samples=movie.shape[0],
        test_split=test_split,
        val_split=val_split,
        seed=seed,
        split_path=split_path,
    )

    x_train = resize_movies(movie[train_idx], resolution)
    x_val = resize_movies(movie[val_idx], resolution)
    x_test = resize_movies(movie[test_idx], resolution)
    y_train_spike = spike[train_idx]
    y_val_spike = spike[val_idx]
    y_test_spike = spike[test_idx]
    y_train_movie = x_train
    y_val_movie = x_val
    y_test_movie = x_test

    strategy, is_tpu = _get_tpu_strategy()

    with strategy.scope():
        encoder = CNN(inputs=Input(shape=(resolution, resolution, 1)), n_out=y_train_spike.shape[1])
        decoder = load_model(sid_model_path, compile=False)
        decoder.trainable = False
        for layer in decoder.layers:
            layer.trainable = False

        regularized_model = DecoderRegularizedEncoder(
            encoder=encoder,
            decoder=decoder,
            lambda_decode=lambda_decode,
            lambda_ssim=lambda_ssim,
        )
        regularized_model.compile(optimizer=Adam(learning_rate=learning_rate))

    if is_tpu:
        per_replica = batch_size // strategy.num_replicas_in_sync
        batch_size = max(per_replica, 1) * strategy.num_replicas_in_sync
        print(f"TPU batch size: {per_replica} per replica x {strategy.num_replicas_in_sync} replicas = {batch_size} effective")
    early_stop = EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=40,
        restore_best_weights=True,
        verbose=1,
    )
    history = regularized_model.fit(
        x_train,
        (y_train_spike, y_train_movie),
        validation_data=(x_val, (y_val_spike, y_val_movie)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    encoder_path = os.path.join(weight_dir, "CNN_decoder_regularized_best.keras")
    encoder.save(encoder_path)

    pred_spike_test = encoder.predict(x_test, batch_size=batch_size, verbose=0)
    pred_movie_test = decoder.predict(pred_spike_test, batch_size=batch_size, verbose=0)
    encode_mse = float(np.mean((y_test_spike - pred_spike_test) ** 2))
    encode_cc = compute_mean_cc_numpy(y_test_spike, pred_spike_test)
    decode_mse, decode_psnr, decode_ssim = cal_performance(y_test_movie, pred_movie_test)

    metrics = {
        "encode_mse": encode_mse,
        "encode_cc": float(encode_cc),
        "decode_mse": float(decode_mse),
        "decode_psnr": float(decode_psnr),
        "decode_ssim": float(decode_ssim),
        "lambda_decode": float(lambda_decode),
        "lambda_ssim": float(lambda_ssim),
        "sid_model_path": sid_model_path,
        "data_path": data_path,
    }

    with open(os.path.join(result_dir, "regularized_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)
    with open(os.path.join(result_dir, "regularized_train_history.json"), "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=4, ensure_ascii=False)
    scipy.io.savemat(
        os.path.join(result_dir, "regularized_predictions.mat"),
        {
            "movie": movie[test_idx],
            "spike": y_test_spike,
            "pred_spike": pred_spike_test,
            "frame_indices": test_idx,
        },
    )
    return {
        "model_path": encoder_path,
        "metrics": metrics,
        "predictions_path": os.path.join(result_dir, "regularized_predictions.mat"),
    }

