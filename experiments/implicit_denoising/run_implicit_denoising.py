import argparse
import json
import os
from datetime import datetime

from CNN.CNN_pred import test_CNN
from CNN.CNN_train import train_CNN
from SID.SID_reconstruction import reconstruct_from_pred_spike, test_SID
from SID.SID_train import train_SID
from encoding_common.pipeline import save_json


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def model_artifact_paths(output_root, experiment_name):
    base = os.path.join(output_root, experiment_name)
    return {
        "base": base,
        "split_path": os.path.join(base, "split_indices.npz"),
        "cnn_weight_dir": os.path.join(base, "cnn", "weights"),
        "cnn_result_dir": os.path.join(base, "cnn", "results"),
        "sid_weight_dir": os.path.join(base, "sid", "weights"),
        "sid_result_dir": os.path.join(base, "sid", "results"),
        "comparison_dir": os.path.join(base, "comparison"),
        "cnn_best_model": os.path.join(base, "cnn", "weights", "CNN_best.keras"),
        "sid_best_model": os.path.join(base, "sid", "weights", "SID_best.keras"),
        "cnn_predictions": os.path.join(base, "cnn", "results", "predictions.mat"),
    }


def run_single_experiment(exp_cfg, global_cfg, mode, skip_existing):
    name = exp_cfg["name"]
    data_path = exp_cfg["data_path"]
    resolution = int(global_cfg["resolution"])
    batch_size = int(global_cfg["batch_size"])
    val_split = float(global_cfg["val_split"])
    test_split = float(global_cfg["test_split"])
    seed = int(global_cfg["seed"])
    save_images = bool(global_cfg.get("save_images", False))
    max_images = global_cfg.get("max_images")
    output_root = global_cfg["output_root"]

    paths = model_artifact_paths(output_root=output_root, experiment_name=name)
    for key in ("cnn_weight_dir", "cnn_result_dir", "sid_weight_dir", "sid_result_dir", "comparison_dir"):
        os.makedirs(paths[key], exist_ok=True)

    run_train = mode in ("all", "train")
    run_eval = mode in ("all", "eval")

    cnn_train_cfg = exp_cfg.get("cnn_train", {})
    sid_train_cfg = exp_cfg.get("sid_train", {})

    cnn_best_exists = os.path.exists(paths["cnn_best_model"])
    sid_best_exists = os.path.exists(paths["sid_best_model"])

    if run_train and (not skip_existing or not cnn_best_exists):
        train_CNN(
            data_path=data_path,
            weight_dir=paths["cnn_weight_dir"],
            result_dir=paths["cnn_result_dir"],
            batch_size=int(cnn_train_cfg.get("batch_size", batch_size)),
            epochs=int(cnn_train_cfg.get("epochs", 500)),
            learning_rate=float(cnn_train_cfg.get("learning_rate", 1e-3)),
            resolution=resolution,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            split_path=paths["split_path"],
            visualization=bool(cnn_train_cfg.get("visualization", True)),
        )

    if run_train and (not skip_existing or not sid_best_exists):
        train_SID(
            data_path=data_path,
            weight_dir=paths["sid_weight_dir"],
            result_dir=paths["sid_result_dir"],
            resolution=resolution,
            batch_size=int(sid_train_cfg.get("batch_size", batch_size)),
            epochs=int(sid_train_cfg.get("epochs", 30)),
            num_iter=int(sid_train_cfg.get("num_iter", 20)),
            learning_rate=float(sid_train_cfg.get("learning_rate", 1e-3)),
            visualization=bool(sid_train_cfg.get("visualization", True)),
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            split_path=paths["split_path"],
        )

    if not run_eval:
        return {
            "name": name,
            "dataset_type": exp_cfg["dataset_type"],
            "data_path": data_path,
            "status": "train_only",
        }

    if not os.path.exists(paths["cnn_best_model"]):
        raise FileNotFoundError(f"CNN model not found for eval: {paths['cnn_best_model']}")
    if not os.path.exists(paths["sid_best_model"]):
        raise FileNotFoundError(f"SID model not found for eval: {paths['sid_best_model']}")

    cnn_test_metrics = test_CNN(
        data_path=data_path,
        model_path=paths["cnn_best_model"],
        result_dir=paths["cnn_result_dir"],
        resolution=resolution,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        split_path=paths["split_path"],
    )

    sid_direct_metrics = test_SID(
        data_path=data_path,
        model_path=paths["sid_best_model"],
        result_dir=paths["sid_result_dir"],
        resolution=resolution,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        split_path=paths["split_path"],
        save_images=save_images,
        max_images=max_images,
    )

    sid_from_cnn_metrics = reconstruct_from_pred_spike(
        sid_model_path=paths["sid_best_model"],
        encoded_spike_path=paths["cnn_predictions"],
        result_dir=paths["comparison_dir"],
        model_name="cnn_pred_decode",
        resolution=resolution,
        batch_size=batch_size,
        save_images=save_images,
        max_images=max_images,
    )

    pred_mse = sid_from_cnn_metrics["cnn_pred_decode_mse"]
    pred_psnr = sid_from_cnn_metrics["cnn_pred_decode_psnr"]
    pred_ssim = sid_from_cnn_metrics["cnn_pred_decode_ssim"]

    comparison = {
        "mse_improvement_abs": float(sid_direct_metrics["mse"] - pred_mse),
        "mse_improvement_pct": float((sid_direct_metrics["mse"] - pred_mse) / sid_direct_metrics["mse"] * 100.0),
        "psnr_improvement_abs": float(pred_psnr - sid_direct_metrics["psnr"]),
        "ssim_improvement_abs": float(pred_ssim - sid_direct_metrics["ssim"]),
    }

    summary = {
        "name": name,
        "dataset_type": exp_cfg["dataset_type"],
        "data_path": data_path,
        "cnn_test_metrics": cnn_test_metrics,
        "sid_direct_metrics": sid_direct_metrics,
        "sid_from_cnn_pred_metrics": {
            "mse": float(pred_mse),
            "psnr": float(pred_psnr),
            "ssim": float(pred_ssim),
        },
        "comparison": comparison,
        "artifacts": paths,
    }
    save_json(os.path.join(paths["base"], "metrics_summary.json"), summary)
    save_json(os.path.join(paths["base"], "comparison", "comparison_table.json"), summary["comparison"])
    return summary


def run_pipeline(config, mode, skip_existing):
    global_cfg = config["global"]
    experiments = [e for e in config["experiments"] if e.get("enabled", True)]

    all_results = []
    for exp in experiments:
        result = run_single_experiment(
            exp_cfg=exp,
            global_cfg=global_cfg,
            mode=mode,
            skip_existing=skip_existing,
        )
        all_results.append(result)

    report = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "skip_existing": bool(skip_existing),
        "global": global_cfg,
        "results": all_results,
    }

    output_root = global_cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "final_report.json")
    save_json(report_path, report)
    return report_path, report


def parse_args():
    parser = argparse.ArgumentParser(description="Run implicit denoising experiment pipeline")
    parser.add_argument(
        "--config",
        default=r"experiments\implicit_denoising\configs\default.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval"],
        default="all",
        help="all=train+eval, train=only train, eval=only evaluate",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training stages when best model files already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    report_path, _ = run_pipeline(config=config, mode=args.mode, skip_existing=args.skip_existing)
    print(f"Pipeline completed. Report saved to: {report_path}")

