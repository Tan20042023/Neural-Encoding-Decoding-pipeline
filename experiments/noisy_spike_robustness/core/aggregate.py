import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_metrics(results_root, noise_type="dropout"):
    results_root = Path(results_root)
    prefix = f"{noise_type}_"
    dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    collected = []
    for d in dirs:
        metric_candidates = [
            d / f"{noise_type}_decode_metrics.json",
            d / "dropout_decode_metrics.json",
            d / "poisson_decode_metrics.json",
            d / "gaussian_decode_metrics.json",
        ]
        json_path = next((p for p in metric_candidates if p.exists()), None)
        if json_path is None:
            print(f"Warning: missing metrics in {d}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if noise_type == "dropout":
            noise_level = float(data.get("dropout_rate", d.name.replace(prefix, "")))
        elif noise_type == "poisson":
            noise_level = float(data.get("poisson_strength", d.name.replace(prefix, "")))
        elif noise_type == "gaussian":
            noise_level = float(data.get("gaussian_strength", d.name.replace(prefix, "")))
        else:
            noise_level = float(data.get("noise_level", d.name.replace(prefix, "")))

        entry = {"noise_level": noise_level, "dir": str(d)}

        # expected keys: noisy_spike_decode and noisy_pred_spike_decode or noisy_movie_spike_decode
        for key in ["noisy_spike_decode", "noisy_pred_spike_decode", "noisy_movie_spike_decode"]:
            if key in data:
                entry[key] = data[key]
        collected.append(entry)

    if not collected:
        raise RuntimeError(f"No valid {noise_type} result directories found under {results_root}")

    # sort by noise level
    collected.sort(key=lambda x: x["noise_level"])
    return collected


def prepare_series(collected, decode_key, metric):
    rates = []
    vals = []
    for e in collected:
        if decode_key in e:
            rates.append(e["noise_level"])
            vals.append(e[decode_key].get(metric, np.nan))
    return np.array(rates), np.array(vals, dtype=float)


def plot_metric(collected, metric, out_path, noise_type="dropout"):
    plt.figure()
    # two possible decodes
    series = ["noisy_spike_decode", "noisy_pred_spike_decode", "noisy_movie_spike_decode"]
    plotted = 0
    for key in series:
        rates, vals = prepare_series(collected, key, metric)
        if rates.size == 0:
            continue
        plt.plot(rates, vals, marker="o", label=key)
        plotted += 1

    if plotted == 0:
        print(f"No data for metric {metric}")
        return None

    x_label_map = {
        "dropout": "dropout_rate",
        "poisson": "poisson_strength",
        "gaussian": "gaussian_strength",
    }
    x_label = x_label_map.get(noise_type, f"{noise_type}_strength")
    plt.xlabel(x_label)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs {x_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def aggregate_and_plot(results_root, out_dir=None, noise_type="dropout"):
    collected = collect_metrics(results_root, noise_type=noise_type)
    if out_dir is None:
        out_dir = Path(results_root) / f"summary_plots_{noise_type}"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["mse", "psnr", "ssim"]
    outputs = {}
    for m in metrics:
        out_path = out_dir / f"{m}_vs_{noise_type}.png"
        res = plot_metric(collected, m, out_path, noise_type=noise_type)
        outputs[m] = str(res) if res is not None else None

    # Save aggregated numeric table
    table = {}
    for e in collected:
        noise_level = e["noise_level"]
        table[str(noise_level)] = {"noise_type": noise_type}
        for key in ["noisy_spike_decode", "noisy_pred_spike_decode", "noisy_movie_spike_decode"]:
            if key in e:
                table[str(noise_level)][key] = e[key]

    table_path = out_dir / "aggregated_metrics.json"
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=4, ensure_ascii=False)

    print(f"Saved plots to: {out_dir}")
    print(f"Saved aggregated metrics to: {table_path}")
    return outputs, table_path


if __name__ == "__main__":
    # Edit these paths if needed (interactive style)
    results_root = r"experiments\noisy_spike_robustness\outputs\manual\gaussian"
    noise_type = "gaussian"  # choices: dropout / poisson / gaussian
    out_dir = None  # default under results_root/summary_plots

    aggregate_and_plot(results_root, out_dir, noise_type=noise_type)
