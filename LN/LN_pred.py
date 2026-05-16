import argparse
import os

from tf_keras.models import load_model

from encoding_common.pipeline import (
    create_or_load_split_indices,
    load_movie_spike_history,
    resize_movies,
    save_json,
    save_predictions_mat,
    set_seed,
)
from utils.metrics import keras_cc


def test_LN(
    data_path,
    model_path,
    result_dir,
    resolution=64,
    batch_size=256,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    split_path=None,
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

    x_test = resize_movies(movie[test_idx], resolution)
    y_test = spike[test_idx]

    model = load_model(model_path, custom_objects={"keras_cc": keras_cc}, compile=False)
    model.compile(optimizer="adam", loss="poisson", metrics=["mse", keras_cc])
    pred = model.predict(x_test, batch_size=batch_size, verbose=0)
    loss, mse, cc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    save_json(
        os.path.join(result_dir, "test_metrics.json"),
        {"loss": float(loss), "mse": float(mse), "cc": float(cc)},
    )
    save_predictions_mat(
        os.path.join(result_dir, "predictions.mat"),
        movie=movie[test_idx],
        spike=y_test,
        pred_spike=pred,
        frame_indices=test_idx,
    )
    return {"loss": float(loss), "mse": float(mse), "cc": float(cc)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LN encoding model")
    parser.add_argument("--data-path", default=r"Dataset\movie\movie01.mat")
    parser.add_argument("--model-path", default=r"LN\weights\movie\movie01\LN_best.keras")
    parser.add_argument("--result-dir", default=r"LN\results\movie\movie01")
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_LN(
        data_path=args.data_path,
        model_path=args.model_path,
        result_dir=args.result_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        split_path=args.split_path,
    )

