"""
Experiment 3 — State Representation Ablation

Trains DQN with different subsets of state features to measure which features
contribute most to learning quality.

Feature sets tested
-------------------
    full          : queue + density + waiting_time + phase + duration  (baseline)
    no_density    : remove per-lane density
    no_waiting    : remove cumulative waiting time
    no_phase      : remove one-hot current phase
    queue_only    : only queue lengths + phase

Usage
-----
    python experiments/state_ablation.py
    python experiments/state_ablation.py --episodes 200
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


# Each variant: name → state config overrides
STATE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {
        "use_queue": True,
        "use_density": True,
        "use_waiting_time": True,
        "use_phase": True,
    },
    "no_density": {
        "use_queue": True,
        "use_density": False,
        "use_waiting_time": True,
        "use_phase": True,
    },
    "no_waiting": {
        "use_queue": True,
        "use_density": True,
        "use_waiting_time": False,
        "use_phase": True,
    },
    "no_phase": {
        "use_queue": True,
        "use_density": True,
        "use_waiting_time": True,
        "use_phase": False,
    },
    "queue_only": {
        "use_queue": True,
        "use_density": False,
        "use_waiting_time": False,
        "use_phase": True,
    },
}


# ──────────────────────────────────────────────
#  Train one state variant
# ──────────────────────────────────────────────

def _train_variant(
    base_config:   Dict[str, Any],
    variant_name:  str,
    state_overrides: Dict[str, Any],
    scenario:      str,
    n_episodes:    int,
    resume:        bool = False,
) -> str:
    """
    Reuses training.train._train_dqn (same as reward_ablation.py) instead of
    a duplicated loop, so this gets --resume support for free: each variant
    gets its own checkpoint subdirectory and picks back up from its last
    saved episode rather than restarting from scratch.
    """
    config = copy.deepcopy(base_config)
    config["state"].update(state_overrides)
    config["training"]["num_episodes"] = n_episodes

    ckpt_dir = os.path.join(config["training"]["checkpoint_dir"], "ablation_state", variant_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    config["training"]["checkpoint_dir"] = ckpt_dir

    set_seed(config["training"]["seed"])

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
    run_name   = f"state_ablation_{variant_name}_{int(time.time())}"
    tb_log_dir = os.path.join(config["logging"]["tensorboard_dir"], run_name)
    writer     = SummaryWriter(log_dir=tb_log_dir)

    print(f"\n  Training DQN with state='{variant_name}' (size={state_size}) ...")

    env = TrafficEnv(cfg_env)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment":    "state_ablation",
            "state_variant": variant_name,
            "state_size":    state_size,
            "scenario":      scenario,
            "episodes":      n_episodes,
            **{f"state.{k}": v for k, v in state_overrides.items()},
        })
        _train_dqn(env, agent, config, writer, run_name, start_episode, best_reward)

    writer.close()
    env.close()
    print(f"  Checkpoint: {ckpt_path}")
    return ckpt_path


# ──────────────────────────────────────────────
#  Plot
# ──────────────────────────────────────────────

def _plot_ablation(
    results:   List[Dict[str, Any]],
    plots_dir: str,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    controllers = [r["controller"] for r in results]
    wait_means  = [r.get("mean_waiting_time_mean", 0) for r in results]
    wait_stds   = [r.get("mean_waiting_time_std",  0) for r in results]

    palette = sns.color_palette("Set2", len(controllers))
    fig, ax  = plt.subplots(figsize=(10, 5))

    bars = ax.bar(controllers, wait_means, yerr=wait_stds, capsize=5,
                  color=palette, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mean Waiting Time (s)")
    ax.set_title("State Feature Ablation — Mean Waiting Time")
    ax.set_xticklabels(controllers, rotation=20, ha="right")

    for bar, m in zip(bars, wait_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{m:.1f}s", ha="center", va="bottom", fontsize=9,
        )

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "state_ablation.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def state_ablation(
    config:     Dict[str, Any],
    n_episodes: int = 200,
    scenario:   str = "normal",
    resume:     bool = False,
) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  Experiment 3: State Representation Ablation")
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
        cycle_time = 60,
        delta_time = config["environment"]["delta_time"],
        num_phases = num_actions,
    )
    results.append(evaluate(config, baseline, "Fixed-time", n_episodes=5, scenario=scenario))

    # ── Train + evaluate each state variant ──────────────────────────────────
    for variant_name, overrides in STATE_VARIANTS.items():
        ckpt = _train_variant(config, variant_name, overrides, scenario, n_episodes, resume=resume)

        _cfg = copy.deepcopy(config)
        _cfg["state"].update(overrides)
        _cfg["environment"]["route_file"] = f"routes_{scenario}.rou.xml"

        _env = TrafficEnv(_cfg)
        _, _ = _env.reset()
        _ss  = _env.observation_space.shape[0]
        _na  = _env.action_space.n
        _env.close()

        device = get_device(_cfg)
        agent  = DQNAgent(_ss, _na, _cfg, device)
        if os.path.exists(ckpt):
            agent.load(ckpt)

        results.append(
            evaluate(_cfg, agent, f"DQN ({variant_name})", n_episodes=5, scenario=scenario)
        )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n── Results ──")
    for r in results:
        wt = r.get("mean_waiting_time_mean", 0)
        print(f"  {r['controller']:<30}  waiting={wt:.1f}s")

    _plot_ablation(results, config["logging"]["plots_dir"])

    df  = pd.DataFrame(results)
    out = os.path.join(config["logging"]["metrics_dir"], "state_ablation.csv")
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

    state_ablation(cfg, n_episodes=args.episodes, scenario=args.scenario, resume=args.resume)
