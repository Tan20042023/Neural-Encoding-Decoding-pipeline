import argparse
import json
import os
import shutil
from datetime import datetime

from CNN.CNN_pred import test_CNN
from CNN.CNN_train import train_CNN
from SID.SID_reconstruction import reconstruct_from_pred_spike
from encoding_common.pipeline import save_json
from experiments.decoding_regularized_encoding.core.regularized_encoder import train_regularized_encoder


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def artifact_paths(output_root):
    return {
        "base": output_root,
        "split_path": os.path.join(output_root, "split_indices.npz"),
        "baseline_weight_dir": os.path.join(output_root, "baseline_cnn", "weights"),
        "baseline_result_dir": os.path.join(output_root, "baseline_cnn", "results"),
        "regularized_weight_dir": os.path.join(output_root, "regularized_cnn", "weights"),
        "regularized_result_dir": os.path.join(output_root, "regularized_cnn", "results"),
        "decode_compare_dir": os.path.join(output_root, "decode_compare"),
        "baseline_model_path": os.path.join(output_root, "baseline_cnn", "weights", "CNN_best.keras"),
        "regularized_model_path": os.path.join(output_root, "regularized_cnn", "weights", "CNN_decoder_regularized_best.keras"),
        "baseline_predictions": os.path.join(output_root, "baseline_cnn", "results", "predictions.mat"),
        "regularized_predictions": os.path.join(output_root, "regularized_cnn", "results", "regularized_predictions.mat"),
    }


def run_pipeline(config, mode, skip_existing):
    global_cfg = config["global"]
    paths = artifact_paths(global_cfg["output_root"])
    for key in (
        "baseline_weight_dir",
        "baseline_result_dir",
        "regularized_weight_dir",
        "regularized_result_dir",
        "decode_compare_dir",
    ):
        os.makedirs(paths[key], exist_ok=True)

    data_path = global_cfg["data_path"]
    sid_model_path = global_cfg["sid_model_path"]
    resolution = int(global_cfg["resolution"])
    seed = int(global_cfg["seed"])
    val_split = float(global_cfg["val_split"])
    test_split = float(global_cfg["test_split"])
    batch_size = int(global_cfg["batch_size"])

    run_train = mode in ("all", "train")
    run_eval = mode in ("all", "eval")

    baseline_cfg = config["baseline_cnn"]
    regularized_cfg = config["regularized_cnn"]

    pretrained_baseline = baseline_cfg.get("pretrained_path")
    if run_train and (not skip_existing or not os.path.exists(paths["baseline_model_path"])):
        if pretrained_baseline and os.path.exists(pretrained_baseline):
            print(f"Loading pretrained baseline CNN from: {pretrained_baseline}")
            shutil.copy2(pretrained_baseline, paths["baseline_model_path"])
            print(f"Copied to: {paths['baseline_model_path']}")
        else:
            if pretrained_baseline:
                print(f"Warning: pretrained_path '{pretrained_baseline}' not found, training from scratch.")
            train_CNN(
                data_path=data_path,
                weight_dir=paths["baseline_weight_dir"],
                result_dir=paths["baseline_result_dir"],
                batch_size=int(baseline_cfg.get("batch_size", batch_size)),
                epochs=int(baseline_cfg.get("epochs", 500)),
                learning_rate=float(baseline_cfg.get("learning_rate", 1e-3)),
                resolution=resolution,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
                split_path=paths["split_path"],
                visualization=bool(baseline_cfg.get("visualization", True)),
            )

    regularized_train_output = None
    if run_train and (not skip_existing or not os.path.exists(paths["regularized_model_path"])):
        regularized_train_output = train_regularized_encoder(
            data_path=data_path,
            sid_model_path=sid_model_path,
            weight_dir=paths["regularized_weight_dir"],
            result_dir=paths["regularized_result_dir"],
            split_path=paths["split_path"],
            resolution=resolution,
            batch_size=int(regularized_cfg.get("batch_size", batch_size)),
            epochs=int(regularized_cfg.get("epochs", 300)),
            learning_rate=float(regularized_cfg.get("learning_rate", 1e-3)),
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            lambda_decode=float(regularized_cfg.get("lambda_decode", 0.2)),
            lambda_ssim=float(regularized_cfg.get("lambda_ssim", 0.0)),
        )

    if not run_eval:
        report = {
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "status": "train_only",
            "artifacts": paths,
            "regularized_train_output": regularized_train_output,
        }
        save_json(os.path.join(paths["base"], "final_report.json"), report)
        return report

    baseline_metrics = test_CNN(
        data_path=data_path,
        model_path=paths["baseline_model_path"],
        result_dir=paths["baseline_result_dir"],
        resolution=resolution,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        split_path=paths["split_path"],
    )

    baseline_decode = reconstruct_from_pred_spike(
        sid_model_path=sid_model_path,
        encoded_spike_path=paths["baseline_predictions"],
        result_dir=paths["decode_compare_dir"],
        model_name="baseline_cnn",
        resolution=resolution,
        batch_size=batch_size,
        save_images=False,
    )
    regularized_decode = reconstruct_from_pred_spike(
        sid_model_path=sid_model_path,
        encoded_spike_path=paths["regularized_predictions"],
        result_dir=paths["decode_compare_dir"],
        model_name="regularized_cnn",
        resolution=resolution,
        batch_size=batch_size,
        save_images=False,
    )

    comparison = {
        "mse_improvement_abs": float(
            baseline_decode["baseline_cnn_mse"] - regularized_decode["regularized_cnn_mse"]
        ),
        "mse_improvement_pct": float(
            (baseline_decode["baseline_cnn_mse"] - regularized_decode["regularized_cnn_mse"])
            / baseline_decode["baseline_cnn_mse"]
            * 100.0
        ),
        "psnr_improvement_abs": float(
            regularized_decode["regularized_cnn_psnr"] - baseline_decode["baseline_cnn_psnr"]
        ),
        "ssim_improvement_abs": float(
            regularized_decode["regularized_cnn_ssim"] - baseline_decode["baseline_cnn_ssim"]
        ),
        "checks": {
            "regularized_mse_better": bool(
                regularized_decode["regularized_cnn_mse"] < baseline_decode["baseline_cnn_mse"]
            ),
            "regularized_psnr_better": bool(
                regularized_decode["regularized_cnn_psnr"] > baseline_decode["baseline_cnn_psnr"]
            ),
            "regularized_ssim_better": bool(
                regularized_decode["regularized_cnn_ssim"] > baseline_decode["baseline_cnn_ssim"]
            ),
        },
    }

    save_json(os.path.join(paths["decode_compare_dir"], "comparison_table.json"), comparison)

    regularized_cnn_metrics = None
    reg_metrics_path = os.path.join(paths["regularized_result_dir"], "regularized_metrics.json")
    if os.path.exists(reg_metrics_path):
        with open(reg_metrics_path, "r", encoding="utf-8") as f:
            reg_metrics = json.load(f)
        regularized_cnn_metrics = {
            "encode_mse": reg_metrics["encode_mse"],
            "encode_cc": reg_metrics["encode_cc"],
            "decode_mse": reg_metrics["decode_mse"],
            "decode_psnr": reg_metrics["decode_psnr"],
            "decode_ssim": reg_metrics["decode_ssim"],
        }

    report = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "global": global_cfg,
        "baseline_cnn_metrics": baseline_metrics,
        "regularized_cnn_metrics": regularized_cnn_metrics,
        "baseline_decode_metrics": baseline_decode,
        "regularized_decode_metrics": regularized_decode,
        "comparison": comparison,
        "artifacts": paths,
    }
    save_json(os.path.join(paths["base"], "final_report.json"), report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run decoding-regularized encoding experiment")
    parser.add_argument(
        "--config",
        default="experiments/decoding_regularized_encoding/configs/default.json",
        help="Path to JSON config file",
    )
    parser.add_argument("--mode", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, mode=args.mode, skip_existing=args.skip_existing)
    print("Decoding-regularized encoding pipeline completed.")

