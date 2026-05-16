import argparse
import json
import os
from datetime import datetime

import numpy as np
import scipy.io
import tensorflow as tf
from keras.models import load_model

from encoding_common.pipeline import save_json
from experiments.neural_digit_twin_transfer.core.data import (
    build_split_info,
    create_or_load_movie1_split,
    load_movie_spike,
    resize_movies,
)
from experiments.neural_digit_twin_transfer.core.models import (
    evaluate_decoder,
    train_decoder,
    train_encoder,
)
from utils.metrics import keras_cc


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def artifact_paths(output_root):
    return {
        "base": output_root,
        "split_path": os.path.join(output_root, "split_indices.npz"),
        "split_info_path": os.path.join(output_root, "split_info.json"),
        "pseudo_spike_path": os.path.join(output_root, "pseudo_spike.mat"),
        "encoder_dir": os.path.join(output_root, "encoder"),
        "group1_dir": os.path.join(output_root, "group1_movie3_only"),
        "group2_dir": os.path.join(output_root, "group2_movie3_plus_movie1_real"),
        "group3_pretrain_dir": os.path.join(output_root, "group3_pretrain"),
        "group3_finetune_dir": os.path.join(output_root, "group3_finetune"),
        "report_path": os.path.join(output_root, "final_report.json"),
    }


def rank_groups(group_metrics):
    items = list(group_metrics.items())
    mse_rank = [name for name, _ in sorted(items, key=lambda x: x[1]["mse"])]
    psnr_rank = [name for name, _ in sorted(items, key=lambda x: x[1]["psnr"], reverse=True)]
    ssim_rank = [name for name, _ in sorted(items, key=lambda x: x[1]["ssim"], reverse=True)]
    return {
        "mse_best_to_worst": mse_rank,
        "psnr_best_to_worst": psnr_rank,
        "ssim_best_to_worst": ssim_rank,
    }


def run_pipeline(config, mode, skip_existing):
    global_cfg = config["global"]
    paths = artifact_paths(global_cfg["output_root"])
    for key in (
        "encoder_dir",
        "group1_dir",
        "group2_dir",
        "group3_pretrain_dir",
        "group3_finetune_dir",
    ):
        os.makedirs(paths[key], exist_ok=True)

    seed = int(global_cfg["seed"])
    np.random.seed(seed)
    tf.random.set_seed(seed)

    movie1_path = global_cfg["movie1_path"]
    movie3_path = global_cfg["movie3_path"]
    resolution = int(global_cfg["resolution"])
    batch_size = int(global_cfg["batch_size"])
    real_ratio = float(global_cfg["movie1_real_ratio"])
    test_ratio = float(global_cfg["movie1_test_ratio"])
    val_split = float(global_cfg["val_split"])
    encoder_cfg = config["encoder"]
    decoder_cfg = config["decoder"]

    run_train = mode in ("all", "train")
    run_eval = mode in ("all", "eval")

    movie1, spike1 = load_movie_spike(movie1_path)
    movie3, spike3 = load_movie_spike(movie3_path)
    if spike1.shape[1] != spike3.shape[1]:
        raise ValueError(f"Cell dimension mismatch: movie1={spike1.shape[1]}, movie3={spike3.shape[1]}")

    movie1_resized = resize_movies(movie1, resolution)
    movie3_resized = resize_movies(movie3, resolution)
    real_idx, unlabeled_idx, test_idx = create_or_load_movie1_split(
        num_frames=movie1.shape[0],
        real_ratio=real_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split_path=paths["split_path"],
    )
    split_info = build_split_info(
        movie1_frames=movie1.shape[0],
        movie3_frames=movie3.shape[0],
        real_ratio=real_ratio,
        test_ratio=test_ratio,
        real_idx=real_idx,
        unlabeled_idx=unlabeled_idx,
        test_idx=test_idx,
    )
    save_json(paths["split_info_path"], split_info)

    movie1_real_movie = movie1_resized[real_idx]
    movie1_real_spike = spike1[real_idx]
    movie1_unlabeled_movie = movie1_resized[unlabeled_idx]
    movie1_test_movie = movie1_resized[test_idx]
    movie1_test_spike = spike1[test_idx]

    encoder_train_movie = np.concatenate([movie3_resized, movie1_real_movie], axis=0)
    encoder_train_spike = np.concatenate([spike3, movie1_real_spike], axis=0)

    encoder_model_path = os.path.join(paths["encoder_dir"], "best_model.keras")
    if run_train and (not skip_existing or not os.path.exists(encoder_model_path)):
        encoder_model_path, _ = train_encoder(
            movie=encoder_train_movie,
            spike=encoder_train_spike,
            output_dir=paths["encoder_dir"],
            epochs=int(encoder_cfg["epochs"]),
            batch_size=batch_size,
            learning_rate=float(encoder_cfg["learning_rate"]),
            val_split=val_split,
            seed=seed,
        )

    if not os.path.exists(encoder_model_path):
        raise FileNotFoundError(f"Encoder model not found: {encoder_model_path}")

    if run_train and (not skip_existing or not os.path.exists(paths["pseudo_spike_path"])):
        encoder_model = load_model(encoder_model_path, custom_objects={"keras_cc": keras_cc})
        pseudo_spike = encoder_model.predict(movie1_unlabeled_movie, batch_size=batch_size, verbose=0)
        scipy.io.savemat(
            paths["pseudo_spike_path"],
            {
                "movie": movie1[unlabeled_idx],
                "pred_spike": pseudo_spike,
                "frame_indices": unlabeled_idx,
            },
        )
    else:
        pseudo_data = scipy.io.loadmat(paths["pseudo_spike_path"])
        if "pred_spike" not in pseudo_data:
            raise KeyError(f"Missing 'pred_spike' in {paths['pseudo_spike_path']}")
        pseudo_spike = pseudo_data["pred_spike"].astype(np.float32)

    g1_model_path = os.path.join(paths["group1_dir"], "best_model.keras")
    if run_train and (not skip_existing or not os.path.exists(g1_model_path)):
        g1_model_path, _ = train_decoder(
            spike=spike3,
            movie=movie3_resized,
            output_dir=paths["group1_dir"],
            model_name="group1_decoder",
            epochs=int(decoder_cfg["epochs"]),
            batch_size=batch_size,
            learning_rate=float(decoder_cfg["learning_rate"]),
            val_split=val_split,
            seed=seed,
        )

    g2_model_path = os.path.join(paths["group2_dir"], "best_model.keras")
    if run_train and (not skip_existing or not os.path.exists(g2_model_path)):
        g2_train_spike = np.concatenate([spike3, movie1_real_spike], axis=0)
        g2_train_movie = np.concatenate([movie3_resized, movie1_real_movie], axis=0)
        g2_model_path, _ = train_decoder(
            spike=g2_train_spike,
            movie=g2_train_movie,
            output_dir=paths["group2_dir"],
            model_name="group2_decoder",
            epochs=int(decoder_cfg["epochs"]),
            batch_size=batch_size,
            learning_rate=float(decoder_cfg["learning_rate"]),
            val_split=val_split,
            seed=seed,
        )

    g3_pre_path = os.path.join(paths["group3_pretrain_dir"], "best_model.keras")
    if run_train and (not skip_existing or not os.path.exists(g3_pre_path)):
        g3_pretrain_spike = np.concatenate([spike3, pseudo_spike], axis=0)
        g3_pretrain_movie = np.concatenate([movie3_resized, movie1_unlabeled_movie], axis=0)
        g3_pre_path, _ = train_decoder(
            spike=g3_pretrain_spike,
            movie=g3_pretrain_movie,
            output_dir=paths["group3_pretrain_dir"],
            model_name="group3_pretrain_decoder",
            epochs=int(decoder_cfg["pretrain_epochs"]),
            batch_size=batch_size,
            learning_rate=float(decoder_cfg["learning_rate"]),
            val_split=val_split,
            seed=seed,
        )

    g3_ft_path = os.path.join(paths["group3_finetune_dir"], "best_model.keras")
    if run_train and (not skip_existing or not os.path.exists(g3_ft_path)):
        g3_ft_path, _ = train_decoder(
            spike=movie1_real_spike,
            movie=movie1_real_movie,
            output_dir=paths["group3_finetune_dir"],
            model_name="group3_finetune_decoder",
            epochs=int(decoder_cfg["finetune_epochs"]),
            batch_size=batch_size,
            learning_rate=float(decoder_cfg["finetune_learning_rate"]),
            val_split=val_split,
            seed=seed,
            init_model_path=g3_pre_path,
        )

    if not run_eval:
        report = {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "status": "train_only",
            "global": global_cfg,
            "artifacts": paths,
        }
        save_json(paths["report_path"], report)
        return report

    for model_path in (g1_model_path, g2_model_path, g3_ft_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Decoder model not found for eval: {model_path}")

    g1_metrics = evaluate_decoder(g1_model_path, movie1_test_spike, movie1_test_movie, batch_size=batch_size)
    g2_metrics = evaluate_decoder(g2_model_path, movie1_test_spike, movie1_test_movie, batch_size=batch_size)
    g3_metrics = evaluate_decoder(g3_ft_path, movie1_test_spike, movie1_test_movie, batch_size=batch_size)

    save_json(os.path.join(paths["group1_dir"], "metrics.json"), g1_metrics)
    save_json(os.path.join(paths["group2_dir"], "metrics.json"), g2_metrics)
    save_json(os.path.join(paths["group3_finetune_dir"], "metrics.json"), g3_metrics)

    group_metrics = {
        "group1_movie3_only": g1_metrics,
        "group2_movie3_plus_movie1_real": g2_metrics,
        "group3_generated_pretrain_then_real_finetune": g3_metrics,
    }
    ranking = rank_groups(group_metrics)
    checks = {
        "mse_group3_lt_group2_lt_group1": bool(g3_metrics["mse"] < g2_metrics["mse"] < g1_metrics["mse"]),
        "psnr_group3_gt_group2_gt_group1": bool(g3_metrics["psnr"] > g2_metrics["psnr"] > g1_metrics["psnr"]),
        "ssim_group3_gt_group2_gt_group1": bool(g3_metrics["ssim"] > g2_metrics["ssim"] > g1_metrics["ssim"]),
    }

    report = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "global": global_cfg,
        "split": split_info,
        "group_metrics": group_metrics,
        "ranking": ranking,
        "expected_order_checks": checks,
        "artifacts": paths,
    }
    save_json(paths["report_path"], report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run neural digit twin transfer experiment")
    parser.add_argument(
        "--config",
        default=r"experiments\neural_digit_twin_transfer\configs\default.json",
        help="Path to JSON config",
    )
    parser.add_argument("--mode", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, mode=args.mode, skip_existing=args.skip_existing)
    print("Neural digit twin transfer pipeline completed.")

