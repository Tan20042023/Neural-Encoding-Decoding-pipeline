import argparse
import json
import os
from datetime import datetime

import numpy as np

from encoding_common.pipeline import save_json
from experiments.noisy_spike_robustness.core.aggregate import aggregate_and_plot
from experiments.noisy_spike_robustness.core.dropout import run_experiment as run_dropout_experiment
from experiments.noisy_spike_robustness.core.gaussian import run_experiment as run_gaussian_experiment
from experiments.noisy_spike_robustness.core.poisson import run_experiment as run_poisson_experiment


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_levels(noise_cfg):
    levels = noise_cfg.get("levels")
    if levels:
        return [float(x) for x in levels]
    start = float(noise_cfg.get("start", 0.1))
    stop = float(noise_cfg.get("stop", 0.9))
    step = float(noise_cfg.get("step", 0.1))
    return [round(float(x), 6) for x in np.arange(start, stop, step)]


def run_noise_type(case_cfg, global_cfg, case_root, noise_type):
    noise_cfg = case_cfg["noises"].get(noise_type, {})
    if not noise_cfg.get("enabled", True):
        return {"noise_type": noise_type, "status": "disabled"}

    levels = get_levels(noise_cfg)
    noise_root = os.path.join(case_root, noise_type)
    os.makedirs(noise_root, exist_ok=True)

    sid_model_path = case_cfg["sid_model_path"]
    predictions_mat_path = case_cfg["predictions_mat_path"]
    resolution = int(global_cfg["resolution"])
    batch_size = int(global_cfg["batch_size"])
    seed = int(global_cfg["seed"])

    for level in levels:
        run_dir = os.path.join(noise_root, f"{noise_type}_{level:g}")
        if noise_type == "dropout":
            run_dropout_experiment(
                sid_model_path=sid_model_path,
                predictions_mat_path=predictions_mat_path,
                output_dir=run_dir,
                dropout_rate=level,
                resolution=resolution,
                batch_size=batch_size,
                seed=seed,
            )
        elif noise_type == "gaussian":
            run_gaussian_experiment(
                sid_model_path=sid_model_path,
                predictions_mat_path=predictions_mat_path,
                output_dir=run_dir,
                gaussian_strength=level,
                resolution=resolution,
                batch_size=batch_size,
                seed=seed,
            )
        elif noise_type == "poisson":
            run_poisson_experiment(
                sid_model_path=sid_model_path,
                predictions_mat_path=predictions_mat_path,
                output_dir=run_dir,
                poisson_strength=level,
                resolution=resolution,
                batch_size=batch_size,
                seed=seed,
                poisson_peak=float(noise_cfg.get("poisson_peak", 1.5)),
            )
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

    summary_dir = os.path.join(noise_root, f"summary_plots_{noise_type}")
    plots, table_path = aggregate_and_plot(
        results_root=noise_root,
        out_dir=summary_dir,
        noise_type=noise_type,
    )
    return {
        "noise_type": noise_type,
        "levels": levels,
        "results_root": noise_root,
        "summary_dir": summary_dir,
        "plots": plots,
        "aggregated_table": str(table_path),
    }


def run_case(case_cfg, global_cfg):
    case_name = case_cfg["name"]
    output_root = global_cfg["output_root"]
    case_root = os.path.join(output_root, case_name)
    os.makedirs(case_root, exist_ok=True)

    noise_results = []
    for noise_type in ("dropout", "gaussian", "poisson"):
        if noise_type in case_cfg.get("noises", {}):
            noise_results.append(run_noise_type(case_cfg, global_cfg, case_root, noise_type))

    case_report = {
        "name": case_name,
        "sid_model_path": case_cfg["sid_model_path"],
        "predictions_mat_path": case_cfg["predictions_mat_path"],
        "noise_results": noise_results,
    }
    save_json(os.path.join(case_root, "case_report.json"), case_report)
    return case_report


def run_pipeline(config):
    global_cfg = config["global"]
    enabled_cases = [c for c in config["cases"] if c.get("enabled", True)]
    all_results = [run_case(case_cfg=case_cfg, global_cfg=global_cfg) for case_cfg in enabled_cases]

    report = {
        "generated_at": datetime.now().isoformat(),
        "global": global_cfg,
        "results": all_results,
    }
    os.makedirs(global_cfg["output_root"], exist_ok=True)
    report_path = os.path.join(global_cfg["output_root"], "final_report.json")
    save_json(report_path, report)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run noisy spike robustness experiment")
    parser.add_argument(
        "--config",
        default="experiments/noisy_spike_robustness/configs/default.json",
        help="Path to JSON config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    final_report = run_pipeline(cfg)
    print(f"Noisy robustness pipeline completed. Report: {final_report}")

