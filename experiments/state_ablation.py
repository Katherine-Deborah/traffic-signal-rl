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

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from baselines.fixed_time import FixedTimeController
from env.traffic_env import TrafficEnv
from training.train import set_seed
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
) -> str:
    config = copy.deepcopy(base_config)
    config["state"].update(state_overrides)
    config["training"]["num_episodes"] = n_episodes

    ckpt_dir  = os.path.join(config["training"]["checkpoint_dir"], "ablation_state")
    ckpt_path = os.path.join(ckpt_dir, f"dqn_state_{variant_name}_best.pt")
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

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run_name = f"state_ablation_{variant_name}_{int(time.time())}"

    print(f"\n  Training DQN with state='{variant_name}' (size={state_size}) ...")

    env = TrafficEnv(cfg_env)
    best_reward = -np.inf

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "experiment":    "state_ablation",
            "state_variant": variant_name,
            "state_size":    state_size,
            "scenario":      scenario,
            "episodes":      n_episodes,
            **{f"state.{k}": v for k, v in state_overrides.items()},
        })

        base_seed = config["training"]["seed"]
        for episode in range(n_episodes):
            state, _ = env.reset(seed=base_seed + episode)
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Bootstrap across time-limit truncation (store terminated only)
                agent.store_transition(state, action, reward, next_state, float(terminated))
                agent.learn()
                ep_reward += reward
                state = next_state

            agent.decay_epsilon()
            metrics = env.get_episode_metrics()
            mlflow.log_metrics({"episode_reward": ep_reward, **metrics}, step=episode)

            if ep_reward > best_reward:
                best_reward = ep_reward
                agent.save(ckpt_path)

            if (episode + 1) % 50 == 0:
                print(
                    f"    [{variant_name}] ep {episode+1:3d}  "
                    f"reward={ep_reward:7.2f}  "
                    f"wait={metrics.get('mean_waiting_time', 0):6.1f}s"
                )

    env.close()
    print(f"  ✓ Best checkpoint: {ckpt_path}")
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
        ckpt = _train_variant(config, variant_name, overrides, scenario, n_episodes)

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
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    state_ablation(cfg, n_episodes=args.episodes, scenario=args.scenario)
