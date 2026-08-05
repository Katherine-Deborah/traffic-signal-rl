"""
Experiment 2 — Reward Function Ablation

Trains DQN with three different reward formulations and compares them against
the fixed-time baseline.

    waiting_time  →  reward = -sum(waiting_times)
    queue_length  →  reward = -sum(queue_lengths)
    combined      →  reward = -(0.5*waiting + 0.5*queue)

Usage
-----
    python experiments/reward_ablation.py
    python experiments/reward_ablation.py --episodes 200 --scenario normal

MLflow
------
    All runs land in experiment "traffic_rl" under group "reward_ablation".
    Compare them at http://localhost:5000 after: mlflow ui --backend-store-uri mlruns
"""

import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — see training/evaluate.py
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from baselines.fixed_time import FixedTimeController
from env.traffic_env import TrafficEnv
from training.train import set_seed, _train_dqn, _load_progress
from training.evaluate import evaluate


REWARD_VARIANTS = ["waiting_time", "queue_length", "combined", "delta_waiting"]


# ──────────────────────────────────────────────
#  Train one reward variant
# ──────────────────────────────────────────────

def _train_variant(
    base_config: Dict[str, Any],
    reward_type: str,
    scenario:    str,
    n_episodes:  int,
    resume:      bool = False,
) -> str:
    """
    Train DQN with a specific reward type. Returns path to best checkpoint.

    Reuses training.train._train_dqn (the same loop used by the main training
    entry point) instead of a duplicated copy, so this ablation gets the same
    --resume support for free: each variant gets its own checkpoint
    subdirectory, and a resumed run picks back up from whatever episode it
    last saved (every training.latest_interval episodes) rather than
    restarting the variant from scratch.
    """
    config = copy.deepcopy(base_config)
    config["reward"]["type"] = reward_type
    config["training"]["num_episodes"] = n_episodes

    ckpt_dir = os.path.join(config["training"]["checkpoint_dir"], "ablation_reward", reward_type)
    os.makedirs(ckpt_dir, exist_ok=True)
    config["training"]["checkpoint_dir"] = ckpt_dir

    set_seed(config["training"]["seed"])

    # Build env to get state/action sizes
    cfg_env = copy.deepcopy(config)
    cfg_env["environment"]["route_file"] = f"routes_{scenario}.rou.xml"
    env = TrafficEnv(cfg_env)
    _, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.close()

    device = get_device(config)
    agent  = DQNAgent(state_size, num_actions, config, device)

    ckpt_path = os.path.join(ckpt_dir, "dqn_best.pt")

    start_episode = 0
    best_reward   = -np.inf
    if resume:
        progress    = _load_progress(ckpt_dir, "dqn")
        latest_ckpt = os.path.join(ckpt_dir, "dqn_latest.pt")
        if progress is not None and os.path.exists(latest_ckpt):
            start_episode, best_reward = progress
            agent.load(latest_ckpt)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run_name   = f"reward_ablation_{reward_type}_{int(time.time())}"
    tb_log_dir = os.path.join(config["logging"]["tensorboard_dir"], run_name)
    writer     = SummaryWriter(log_dir=tb_log_dir)

    print(f"\n  Training DQN with reward='{reward_type}' for {n_episodes} episodes...")

    env = TrafficEnv(cfg_env)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment":  "reward_ablation",
            "reward_type": reward_type,
            "scenario":    scenario,
            "episodes":    n_episodes,
        })
        _train_dqn(env, agent, config, writer, run_name, start_episode, best_reward)

    writer.close()
    env.close()
    print(f"  Checkpoint: {ckpt_path}")
    return ckpt_path


# ──────────────────────────────────────────────
#  Plot ablation results
# ──────────────────────────────────────────────

def _plot_ablation(
    results:   List[Dict[str, Any]],
    plots_dir: str,
    title:     str,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    controllers = [r["controller"] for r in results]
    wait_means  = [r.get("mean_waiting_time_mean", 0) for r in results]
    wait_stds   = [r.get("mean_waiting_time_std",  0) for r in results]
    queue_means = [r.get("mean_queue_length_mean", 0) for r in results]
    queue_stds  = [r.get("mean_queue_length_std",  0) for r in results]

    palette = sns.color_palette("Set2", len(controllers))

    for ax, means, stds, ylabel in [
        (axes[0], wait_means,  wait_stds,  "Mean Waiting Time (s)"),
        (axes[1], queue_means, queue_stds, "Mean Queue Length"),
    ]:
        bars = ax.bar(controllers, means, yerr=stds, capsize=5,
                      color=palette, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticklabels(controllers, rotation=20, ha="right")
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "reward_ablation.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def reward_ablation(
    config:     Dict[str, Any],
    n_episodes: int = 200,
    scenario:   str = "normal",
    resume:     bool = False,
) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  Experiment 2: Reward Function Ablation")
    print("═" * 60)

    results: List[Dict[str, Any]] = []

    # ── Fixed-time baseline ──────────────────────────────────────────────────
    env = TrafficEnv({**config, "environment": {
        **config["environment"], "route_file": f"routes_{scenario}.rou.xml"
    }})
    _, _ = env.reset()
    num_actions = env.action_space.n
    env.close()

    baseline = FixedTimeController(
        cycle_time  = 60,
        delta_time  = config["environment"]["delta_time"],
        num_phases  = num_actions,
    )
    results.append(evaluate(config, baseline, "Fixed-time", n_episodes=5, scenario=scenario))

    # ── Train + evaluate each reward variant ─────────────────────────────────
    for reward_type in REWARD_VARIANTS:
        ckpt = _train_variant(config, reward_type, scenario, n_episodes, resume=resume)

        # Load best checkpoint for evaluation
        _cfg = copy.deepcopy(config)
        _cfg["reward"]["type"] = reward_type
        device = get_device(_cfg)

        _env = TrafficEnv({**_cfg, "environment": {
            **_cfg["environment"], "route_file": f"routes_{scenario}.rou.xml"
        }})
        _, _ = _env.reset()
        _ss  = _env.observation_space.shape[0]
        _na  = _env.action_space.n
        _env.close()

        agent = DQNAgent(_ss, _na, _cfg, device)
        if os.path.exists(ckpt):
            agent.load(ckpt)

        results.append(
            evaluate(_cfg, agent, f"DQN ({reward_type})", n_episodes=5, scenario=scenario)
        )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n── Results ──")
    for r in results:
        wt = r.get("mean_waiting_time_mean", 0)
        print(f"  {r['controller']:<25}  waiting={wt:.1f}s")

    _plot_ablation(results, config["logging"]["plots_dir"], "Reward Function Ablation")

    df = pd.DataFrame(results)
    out = os.path.join(config["logging"]["metrics_dir"], "reward_ablation.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  CSV saved: {out}")

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--scenario", default="normal")
    p.add_argument("--resume",   action="store_true",
                   help="Resume each variant from its last checkpoint if present.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reward_ablation(cfg, n_episodes=args.episodes, scenario=args.scenario, resume=args.resume)
