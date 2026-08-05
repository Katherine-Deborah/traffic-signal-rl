"""
Evaluation script — compare trained DQN / PPO agents against the fixed-time baseline.

Usage
-----
    python training/evaluate.py                          # uses best checkpoints
    python training/evaluate.py --dqn results/checkpoints/dqn_final.pt
    python training/evaluate.py --ppo results/checkpoints/ppo_final.pt
    python training/evaluate.py --episodes 10 --scenario ns_peak
    python training/evaluate.py --gui       # watch one episode per controller

Output
------
    Console table of mean/std metrics
    results/plots/comparison.png
    results/metrics/evaluation_results.csv
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids GUI (Tk) init, which
                        # can hang when running as a detached/background
                        # process with no window session attached.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from baselines.fixed_time import FixedTimeController
from baselines.max_pressure import MaxPressureController
from env.traffic_env import TrafficEnv


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(config: Dict[str, Any], scenario: str, gui: bool = False) -> TrafficEnv:
    cfg = dict(config)
    cfg["environment"] = dict(config["environment"])
    cfg["environment"]["route_file"] = f"routes_{scenario}.rou.xml"
    if gui:
        cfg["sumo"] = dict(config["sumo"])
        cfg["sumo"]["gui"] = True
    return TrafficEnv(cfg, render_mode="human" if gui else None)


# ──────────────────────────────────────────────
#  Run one episode, return metrics
# ──────────────────────────────────────────────

def run_episode(
    env: TrafficEnv, controller, is_ppo: bool = False, seed: Optional[int] = None
) -> Dict[str, float]:
    state, _ = env.reset(seed=seed)
    done      = False
    ep_reward = 0.0

    if hasattr(controller, "reset"):
        controller.reset()

    while not done:
        if is_ppo:
            action, _, _ = controller.select_action(state, explore=False)
        else:
            action = controller.select_action(state, explore=False)

        state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        ep_reward += reward

        if hasattr(controller, "step"):
            controller.step()

    metrics = env.get_episode_metrics()
    metrics["episode_reward"] = ep_reward
    return metrics


# ──────────────────────────────────────────────
#  Evaluate N episodes for one controller
# ──────────────────────────────────────────────

def evaluate(
    config:      Dict[str, Any],
    controller,
    controller_name: str,
    n_episodes:  int   = 5,
    scenario:    str   = "normal",
    gui:         bool  = False,
    is_ppo:      bool  = False,
    controller_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    controller_factory: if given, called as controller_factory(env) *after*
    the env is created, and its return value used instead of `controller`.
    Needed for controllers like MaxPressureController that read live queue
    state via the env's own traci connection rather than the state vector.
    """
    env = make_env(config, scenario, gui)
    if controller_factory is not None:
        controller = controller_factory(env)

    all_metrics: List[Dict[str, float]] = []
    print(f"\n  Evaluating [{controller_name}] for {n_episodes} episodes...")

    for ep in range(n_episodes):
        # Fixed eval seed range, disjoint from training seeds (42..541), so
        # every controller faces the same set of distinct traffic realisations.
        m = run_episode(env, controller, is_ppo=is_ppo, seed=10_000 + ep)
        all_metrics.append(m)
        print(
            f"    ep {ep+1:2d}  reward={m['episode_reward']:7.2f}  "
            f"wait={m.get('mean_waiting_time',0):6.1f}s  "
            f"queue={m.get('mean_queue_length',0):5.1f}"
        )

    env.close()

    # Aggregate
    keys = all_metrics[0].keys()
    summary: Dict[str, Any] = {"controller": controller_name}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))

    return summary


# ──────────────────────────────────────────────
#  Plots (research paper style)
# ──────────────────────────────────────────────

def _set_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def plot_comparison(
    results:   List[Dict[str, Any]],
    plots_dir: str,
    filename:  str = "comparison.png",
    subtitle:  str = "",
) -> None:
    _set_paper_style()

    metrics_to_plot = [
        ("mean_waiting_time_mean", "Mean Waiting Time (s)"),
        ("mean_queue_length_mean", "Mean Queue Length (vehicles)"),
        ("episode_reward_mean",    "Mean Episode Reward"),
    ]

    controllers = [r["controller"] for r in results]
    colors      = sns.color_palette("Set2", len(controllers))

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))

    for ax, (metric_key, ylabel) in zip(axes, metrics_to_plot):
        means  = [r.get(metric_key, 0)         for r in results]
        stds   = [r.get(metric_key.replace("_mean", "_std"), 0) for r in results]

        bars = ax.bar(controllers, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticklabels(controllers, rotation=15, ha="right")

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10
            )

    title = "Traffic Signal Controller Comparison" + (f" — {subtitle}" if subtitle else "")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    fig.savefig(save_path)
    print(f"\nComparison plot saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Results table
# ──────────────────────────────────────────────

def print_results_table(results: List[Dict[str, Any]], baseline_name: str = "Fixed-time") -> None:
    keys = [
        "mean_waiting_time_mean",
        "mean_queue_length_mean",
        "max_waiting_time_mean",
        "episode_reward_mean",
    ]
    labels = [
        "Waiting Time (s)",
        "Queue Length",
        "Max Wait (s)",
        "Episode Reward",
    ]

    baseline = next((r for r in results if r["controller"] == baseline_name), None)

    print("\n" + "═" * 72)
    print(f"  {'Controller':<18}", end="")
    for label in labels:
        print(f"  {label:>16}", end="")
    print()
    print("─" * 72)

    for r in results:
        print(f"  {r['controller']:<18}", end="")
        for key in keys:
            val = r.get(key, 0)
            print(f"  {val:>16.1f}", end="")
        print()

    if baseline:
        print("─" * 72)
        print("  Improvement vs Fixed-time baseline:")
        for rl_result in results:
            if rl_result["controller"] == baseline_name:
                continue
            wait_bl  = baseline.get("mean_waiting_time_mean", 1)
            wait_rl  = rl_result.get("mean_waiting_time_mean", 1)
            pct_wait = (wait_bl - wait_rl) / wait_bl * 100 if wait_bl > 0 else 0
            print(f"    {rl_result['controller']}: {pct_wait:+.1f}% waiting time")

    print("═" * 72)


def save_results_csv(
    results: List[Dict[str, Any]], metrics_dir: str, filename: str = "evaluation_results.csv"
) -> None:
    os.makedirs(metrics_dir, exist_ok=True)
    df   = pd.DataFrame(results)
    path = os.path.join(metrics_dir, filename)
    df.to_csv(path, index=False)
    print(f"Results saved: {path}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and compare traffic controllers")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--dqn",      default=None, help="Path to DQN checkpoint")
    p.add_argument("--ppo",      default=None, help="Path to PPO checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--scenario", default="normal",
                   choices=["normal", "ns_peak", "ew_peak", "high"],
                   help="Used if --scenarios is not given.")
    p.add_argument("--scenarios", default=None,
                   help="Comma-separated list of scenarios to evaluate, e.g. "
                        "'normal,high'. Overrides --scenario. Produces one "
                        "comparison{_scenario}.png / evaluation_results{_scenario}.csv "
                        "per scenario (no suffix for 'normal', to match existing paths).")
    p.add_argument("--gui",      action="store_true")
    return p.parse_args()


def run_evaluation(config: Dict[str, Any], scenario: str, args: argparse.Namespace) -> None:
    device = get_device(config)

    # Discover state/action size from environment
    _env       = make_env(config, scenario)
    _s, _      = _env.reset()
    state_size  = _env.observation_space.shape[0]
    num_actions = _env.action_space.n
    _env.close()

    print(f"\n{'#' * 60}\n  Scenario: {scenario}\n{'#' * 60}")

    results: List[Dict[str, Any]] = []

    # ── Fixed-time baseline ──────────────────────────────────────────────────
    baseline = FixedTimeController(
        cycle_time = 60,
        delta_time = config["environment"]["delta_time"],
        num_phases = num_actions,
    )
    results.append(
        evaluate(config, baseline, "Fixed-time", args.episodes, scenario, args.gui)
    )

    # ── Max-Pressure baseline (static, no training) ──────────────────────────
    results.append(
        evaluate(
            config, None, "Max-Pressure", args.episodes, scenario, args.gui,
            controller_factory=MaxPressureController,
        )
    )

    # ── DQN agent ────────────────────────────────────────────────────────────
    dqn_path = args.dqn or os.path.join(config["training"]["checkpoint_dir"], "dqn_best.pt")
    if os.path.exists(dqn_path):
        dqn_agent = DQNAgent(state_size, num_actions, config, device)
        dqn_agent.load(dqn_path)
        results.append(
            evaluate(config, dqn_agent, "DQN", args.episodes, scenario, args.gui)
        )
    else:
        print(f"\n[DQN] checkpoint not found at '{dqn_path}' — skipping.")

    # ── PPO agent ────────────────────────────────────────────────────────────
    ppo_path = args.ppo or os.path.join(config["training"]["checkpoint_dir"], "ppo_best.pt")
    if os.path.exists(ppo_path):
        ppo_agent = PPOAgent(state_size, num_actions, config, device)
        ppo_agent.load(ppo_path)
        results.append(
            evaluate(config, ppo_agent, "PPO", args.episodes, scenario, args.gui, is_ppo=True)
        )
    else:
        print(f"\n[PPO] checkpoint not found at '{ppo_path}' — skipping.")

    # ── Report ───────────────────────────────────────────────────────────────
    suffix = "" if scenario == "normal" else f"_{scenario}"
    print_results_table(results)
    save_results_csv(results, config["logging"]["metrics_dir"], f"evaluation_results{suffix}.csv")
    plot_comparison(results, config["logging"]["plots_dir"], f"comparison{suffix}.png", subtitle=scenario)


if __name__ == "__main__":
    args   = _parse_args()
    config = load_config(args.config)

    scenarios = (
        [s.strip() for s in args.scenarios.split(",")] if args.scenarios else [args.scenario]
    )
    for scenario in scenarios:
        run_evaluation(config, scenario, args)
