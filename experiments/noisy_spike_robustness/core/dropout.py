import json
import os

import cv2
import numpy as np
import scipy.io
from tf_keras.models import load_model

from SID.SID import cal_performance


def resize_images(images, resolution):
    if images.ndim == 4 and images.shape[-1] == 1:
        images_2d = images[..., 0]
    elif images.ndim == 3:
        images_2d = images
    else:
        raise ValueError(f"Unsupported movie shape for resize: {images.shape}")

    resized = np.zeros((images_2d.shape[0], resolution, resolution), dtype=np.float32)
    for index in range(images_2d.shape[0]):
        resized[index] = cv2.resize(images_2d[index], (resolution, resolution))
    return resized.reshape((images_2d.shape[0], resolution, resolution, 1))


def apply_dropout_noise(array, dropout_rate, rng):
    if dropout_rate < 0.0 or dropout_rate > 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
    mask = rng.random(array.shape) >= dropout_rate
    return array * mask.astype(array.dtype)


def decode_and_score(sid_model, noisy_spike, movie_resized, batch_size):
    predictions = sid_model.predict(noisy_spike, batch_size=batch_size, verbose=0)
    mse, psnr, ssim = cal_performance(movie_resized, predictions)
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def run_experiment(
    sid_model_path,
    predictions_mat_path,
    output_dir,
    dropout_rate,
    resolution=64,
    batch_size=256,
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    data = scipy.io.loadmat(predictions_mat_path)
    required_keys = ["movie", "spike"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing keys in predictions.mat: {missing_keys}")

    movie = data["movie"]
    spike = data["spike"].astype(np.float32)

    if "movie_spike" in data:
        movie_spike = data["movie_spike"].astype(np.float32)
        movie_spike_key = "movie_spike"
    elif "pred_spike" in data:
        movie_spike = data["pred_spike"].astype(np.float32)
        movie_spike_key = "pred_spike"
    else:
        raise KeyError("predictions.mat must contain either 'movie_spike' or 'pred_spike'")

    frame_indices = data.get("frame_indices", np.arange(spike.shape[0])).flatten()

    if spike.shape[0] != movie.shape[0] or movie_spike.shape[0] != movie.shape[0]:
        raise ValueError(
            "Sample count mismatch among movie/spike/movie_spike: "
            f"movie={movie.shape[0]}, spike={spike.shape[0]}, {movie_spike_key}={movie_spike.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    noisy_spike = apply_dropout_noise(spike, dropout_rate, rng)
    noisy_movie_spike = apply_dropout_noise(movie_spike, dropout_rate, rng)

    noisy_mat_path = os.path.join(output_dir, "predictions_dropout.mat")
    scipy.io.savemat(
        noisy_mat_path,
        {
            "movie": movie,
            "spike": spike,
            movie_spike_key: movie_spike,
            "noisy_spike": noisy_spike,
            f"noisy_{movie_spike_key}": noisy_movie_spike,
            "frame_indices": frame_indices,
            "dropout_rate": np.array([[dropout_rate]], dtype=np.float32),
        },
    )

    movie_resized = resize_images(movie, resolution)
    sid_model = load_model(sid_model_path)

    noisy_spike_metrics = decode_and_score(sid_model, noisy_spike, movie_resized, batch_size)
    noisy_movie_spike_metrics = decode_and_score(
        sid_model,
        noisy_movie_spike,
        movie_resized,
        batch_size,
    )

    metrics = {
        "dropout_rate": float(dropout_rate),
        "sid_model_path": sid_model_path,
        "predictions_mat_path": predictions_mat_path,
        "noisy_mat_path": noisy_mat_path,
        "noisy_spike_decode": noisy_spike_metrics,
        f"noisy_{movie_spike_key}_decode": noisy_movie_spike_metrics,
    }

    metrics_path = os.path.join(output_dir, "dropout_decode_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print(f"Dropout experiment completed with rate={dropout_rate}")
    print(f"Noisy MAT saved to: {noisy_mat_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    # Interactive-style batch run: iterate dropout rates 0.1-0.8
    sid_model_path = "SID/weights/movie/movie01/SID_best.keras"
    predictions_mat_path = "CNN/results/movie/movie01/predictions.mat"

    resolution = 64
    batch_size = 256
    seed = 42

    # run for dropout rates 0.1 .. 0.8
    dropout_rates = [round(x, 1) for x in np.arange(0.1, 0.9, 0.1)]
    for rate in dropout_rates:
        output_dir = f"experiments/noisy_spike_robustness/outputs/manual/dropout/dropout_{rate}"
        print(f"Running dropout experiment rate={rate} -> output: {output_dir}")
        run_experiment(
            sid_model_path=sid_model_path,
            predictions_mat_path=predictions_mat_path,
            output_dir=output_dir,
            dropout_rate=rate,
            resolution=resolution,
            batch_size=batch_size,
            seed=seed,
        )
