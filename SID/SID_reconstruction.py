import argparse
import os

from tf_keras.models import load_model
from matplotlib import pyplot as plt

from encoding_common.pipeline import (
    create_or_load_split_indices,
    load_movie_spike_history,
    resize_movies,
    save_json,
    set_seed,
)
from SID.SID import cal_performance


def _save_reconstruction_images(original_movies, reconstructed_movies, frame_indices, output_dir, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    original_dir = os.path.join(output_dir, "original_images")
    reconstructed_dir = os.path.join(output_dir, "reconstructed_images")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    total = len(frame_indices)
    use_count = total if max_images is None else min(total, max_images)
    for idx in range(use_count):
        frame_idx = int(frame_indices[idx])
        plt.imsave(os.path.join(original_dir, f"frame_{frame_idx}.png"), original_movies[idx].squeeze(), cmap="gray")
        plt.imsave(
            os.path.join(reconstructed_dir, f"frame_{frame_idx}.png"),
            reconstructed_movies[idx].squeeze(),
            cmap="gray",
        )


def test_SID(
    data_path,
    model_path,
    result_dir,
    resolution=64,
    batch_size=256,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    split_path=None,
    save_images=True,
    max_images=None,
):
    set_seed(seed)
    os.makedirs(result_dir, exist_ok=True)

    movie, spike, _ = load_movie_spike_history(data_path, require_history=False)
    if split_path is None:
        split_path = os.path.join(result_dir, "split_indices.npz")
    _, _, test_idx = create_or_load_split_indices(
        num_samples=movie.shape[0],
        test_split=test_split,
        val_split=val_split,
        seed=seed,
        split_path=split_path,
    )

    x_test = spike[test_idx]
    y_test = movie[test_idx]
    y_test_resized = resize_movies(y_test, resolution)

    model = load_model(model_path, compile=False)
    predictions = model.predict(x_test, batch_size=batch_size, verbose=0)
    mse, psnr, ssim = cal_performance(y_test_resized, predictions)

    metrics = {"mse": float(mse), "psnr": float(psnr), "ssim": float(ssim)}
    save_json(os.path.join(result_dir, "test_metrics.json"), metrics)
    import scipy.io

    scipy.io.savemat(
        os.path.join(result_dir, "sid_reconstruction_predictions.mat"),
        {
            "movie": y_test,
            "spike": x_test,
            "pred_movie": predictions,
            "frame_indices": test_idx,
        },
    )

    if save_images:
        _save_reconstruction_images(
            original_movies=y_test,
            reconstructed_movies=predictions,
            frame_indices=test_idx,
            output_dir=result_dir,
            max_images=max_images,
        )
    return metrics


def reconstruct_from_pred_spike(
    sid_model_path,
    encoded_spike_path,
    result_dir,
    model_name,
    resolution=64,
    batch_size=256,
    save_images=True,
    max_images=None,
):
    import scipy.io

    os.makedirs(result_dir, exist_ok=True)
    encoded_data = scipy.io.loadmat(encoded_spike_path)

    if "pred_spike" not in encoded_data or "movie" not in encoded_data:
        raise KeyError("encoded_spike_path must contain keys: 'pred_spike' and 'movie'")

    pred_spike = encoded_data["pred_spike"]
    movie = encoded_data["movie"]
    frame_indices = encoded_data.get("frame_indices")
    if frame_indices is None:
        frame_indices = list(range(pred_spike.shape[0]))
    else:
        frame_indices = frame_indices.flatten()

    movie_resized = resize_movies(movie, resolution)
    sid_model = load_model(sid_model_path, compile=False)
    predictions = sid_model.predict(pred_spike, batch_size=batch_size, verbose=0)
    mse, psnr, ssim = cal_performance(movie_resized, predictions)

    metrics_path = os.path.join(result_dir, "test_metrics.json")
    existing_metrics = {}
    if os.path.exists(metrics_path):
        import json

        with open(metrics_path, "r", encoding="utf-8") as file:
            existing_metrics = json.load(file)

    existing_metrics.update(
        {
            f"{model_name}_mse": float(mse),
            f"{model_name}_psnr": float(psnr),
            f"{model_name}_ssim": float(ssim),
        }
    )
    save_json(metrics_path, existing_metrics)

    if save_images:
        pred_dir = os.path.join(result_dir, f"{model_name}_pred_images")
        os.makedirs(pred_dir, exist_ok=True)
        total = len(frame_indices)
        use_count = total if max_images is None else min(total, max_images)
        for idx in range(use_count):
            frame_idx = int(frame_indices[idx])
            plt.imsave(os.path.join(pred_dir, f"frame_{frame_idx}.png"), predictions[idx].squeeze(), cmap="gray")

    return {
        f"{model_name}_mse": float(mse),
        f"{model_name}_psnr": float(psnr),
        f"{model_name}_ssim": float(ssim),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SID and run reconstruction")
    parser.add_argument("--mode", choices=["test", "reconstruct"], default="test")
    parser.add_argument("--data-path", default="Dataset/movie/movie01.mat")
    parser.add_argument("--model-path", default="SID/weights/movie/movie01/SID_best.keras")
    parser.add_argument("--sid-model-path", default="SID/weights/movie/movie01/SID_best.keras")
    parser.add_argument("--encoded-spike-path", default="CNN/results/movie/movie01/predictions.mat")
    parser.add_argument("--result-dir", default="SID/SID_results/movie/movie01")
    parser.add_argument("--model-name", default="CNN")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "test":
        test_SID(
            data_path=args.data_path,
            model_path=args.model_path,
            result_dir=args.result_dir,
            resolution=args.resolution,
            batch_size=args.batch_size,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            split_path=args.split_path,
            save_images=not args.no_save_images,
            max_images=args.max_images,
        )
    else:
        reconstruct_from_pred_spike(
            sid_model_path=args.sid_model_path,
            encoded_spike_path=args.encoded_spike_path,
            result_dir=args.result_dir,
            model_name=args.model_name,
            resolution=args.resolution,
            batch_size=args.batch_size,
            save_images=not args.no_save_images,
            max_images=args.max_images,
        )

