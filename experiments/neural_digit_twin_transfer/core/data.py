import os

import numpy as np
import scipy.io
import tensorflow as tf


def load_movie_spike(mat_path):
    data = scipy.io.loadmat(mat_path)
    if "movie" not in data:
        raise KeyError(f"Missing key 'movie' in {mat_path}")

    movie = data["movie"].astype(np.float32)
    if "spike" in data:
        spike = data["spike"].astype(np.float32)
    elif "spike_single" in data:
        spike = data["spike_single"].T.astype(np.float32)
    else:
        raise KeyError(f"Missing key 'spike'/'spike_single' in {mat_path}")

    if movie.shape[0] != spike.shape[0]:
        raise ValueError(f"Sample count mismatch in {mat_path}: movie={movie.shape[0]}, spike={spike.shape[0]}")
    return movie, spike


def resize_movies(movies, resolution):
    movies = movies.astype(np.float32)
    movies = np.expand_dims(movies, axis=-1)
    resized = tf.image.resize(movies, [resolution, resolution], method="bilinear").numpy()
    return resized.astype(np.float32)


def create_or_load_movie1_split(
    num_frames,
    real_ratio,
    test_ratio,
    seed,
    split_path,
):
    if split_path and os.path.exists(split_path):
        split = np.load(split_path)
        saved_num = int(split["num_frames"][0])
        if saved_num != int(num_frames):
            raise ValueError(f"Split num_frames mismatch: expected {num_frames}, got {saved_num}")
        return split["real_idx"], split["unlabeled_idx"], split["test_idx"]

    if not 0 < real_ratio < 1:
        raise ValueError("real_ratio must be in (0, 1)")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(num_frames)
    rng.shuffle(all_indices)

    test_count = int(round(num_frames * test_ratio))
    real_count = int(round(num_frames * real_ratio))

    test_count = min(max(test_count, 1), num_frames - 2)
    remain = all_indices[test_count:]
    real_count = min(max(real_count, 1), len(remain) - 1)

    test_idx = np.sort(all_indices[:test_count])
    real_idx = np.sort(remain[:real_count])
    unlabeled_idx = np.sort(remain[real_count:])

    if len(unlabeled_idx) == 0:
        raise ValueError("Unlabeled subset is empty, adjust split ratios")

    split_dir = os.path.dirname(split_path)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    np.savez(
        split_path,
        real_idx=real_idx,
        unlabeled_idx=unlabeled_idx,
        test_idx=test_idx,
        num_frames=np.array([num_frames], dtype=np.int64),
        seed=np.array([seed], dtype=np.int64),
        real_ratio=np.array([real_ratio], dtype=np.float32),
        test_ratio=np.array([test_ratio], dtype=np.float32),
    )
    return real_idx, unlabeled_idx, test_idx


def build_split_info(movie1_frames, movie3_frames, real_ratio, test_ratio, real_idx, unlabeled_idx, test_idx):
    return {
        "movie1_total_frames": int(movie1_frames),
        "movie3_total_frames": int(movie3_frames),
        "movie1_real_ratio": float(real_ratio),
        "movie1_test_ratio": float(test_ratio),
        "movie1_real_frames": int(len(real_idx)),
        "movie1_unlabeled_frames": int(len(unlabeled_idx)),
        "movie1_test_frames": int(len(test_idx)),
        "movie1_real_indices": real_idx.tolist(),
        "movie1_unlabeled_indices": unlabeled_idx.tolist(),
        "movie1_test_indices": test_idx.tolist(),
    }

