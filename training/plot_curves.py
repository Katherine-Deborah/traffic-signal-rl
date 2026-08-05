"""
Plot training curves (episode reward + mean waiting time) for the most recent
DQN and PPO training runs, read from MLflow.

Usage
-----
    python training/plot_curves.py
    python training/plot_curves.py --output results/plots/training_curves.png
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _latest_run_id(client: MlflowClient, experiment_name: str, algo: str) -> Optional[str]:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.algorithm = '{algo}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _metric_history(client: MlflowClient, run_id: str, key: str) -> List[float]:
    history = client.get_metric_history(run_id, key)
    history.sort(key=lambda m: m.step)
    return [m.value for m in history]


def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_curves(config: dict, output: str) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    client = MlflowClient(tracking_uri=config["mlflow"]["tracking_uri"])
    experiment = config["mlflow"]["experiment_name"]

    curves: Dict[str, Dict[str, List[float]]] = {}
    for algo in ["dqn", "ppo"]:
        run_id = _latest_run_id(client, experiment, algo)
        if run_id is None:
            print(f"  No MLflow run found for '{algo}' — skipping.")
            continue
        curves[algo.upper()] = {
            "reward": _metric_history(client, run_id, "episode_reward"),
            "waiting": _metric_history(client, run_id, "mean_waiting_time"),
        }

    if not curves:
        print("No training runs found in MLflow. Train first: python training/train.py")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = dict(zip(curves.keys(), sns.color_palette("Set2", len(curves))))

    outlier_notes = []
    for name, data in curves.items():
        for ax, key, label in [
            (axes[0], "reward", "Episode Reward"),
            (axes[1], "waiting", "Mean Waiting Time (s)"),
        ]:
            raw = data[key]
            if not raw:
                continue
            ax.plot(raw, alpha=0.25, color=colors[name])
            smoothed = _smooth(raw)
            ax.plot(
                np.arange(len(raw) - len(smoothed), len(raw)),
                smoothed,
                label=name,
                color=colors[name],
                linewidth=2,
            )
            ax.set_xlabel("Episode")
            ax.set_ylabel(label)
            ax.set_title(label)

            # Rare episodes (e.g. an unlucky exploration action triggering
            # gridlock that never clears since time-to-teleport is disabled)
            # can dwarf everything else on a linear axis. Clip the y-range to
            # the 1st-99th percentile of the smoothed curve so normal training
            # progress stays visible, and call out what got clipped.
            arr = np.asarray(raw)
            lo, hi = np.percentile(arr, [1, 99])
            pad = 0.1 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)

            outliers = np.where((arr < lo) | (arr > hi))[0]
            if len(outliers):
                worst = outliers[np.argmax(np.abs(arr[outliers] - np.median(arr)))]
                outlier_notes.append(
                    f"{name} ep {worst}: {label.lower()}={raw[worst]:.0f} (off-scale, clipped)"
                )

    axes[1].axhline(y=0, color="none")  # keep axis anchored sensibly
    for ax in axes:
        ax.legend()

    if outlier_notes:
        fig.text(
            0.5, -0.04, "Clipped for readability — " + "; ".join(sorted(set(outlier_notes))),
            ha="center", fontsize=8.5, style="italic", color="dimgray",
        )

    fig.suptitle("Training Curves — DQN vs PPO", fontsize=14, fontweight="bold")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Training curves saved: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot training curves from MLflow")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default="results/plots/training_curves.png")
    args = p.parse_args()

    plot_training_curves(load_config(args.config), args.output)
