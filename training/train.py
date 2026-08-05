"""
Unified training entry point for DQN and PPO agents.

Usage
-----
    # From project root
    python training/train.py --algo dqn
    python training/train.py --algo ppo
    python training/train.py --algo dqn --episodes 300 --scenario ns_peak
    python training/train.py --algo dqn --gui          # watch in SUMO GUI

MLflow UI
---------
    mlflow ui --backend-store-uri mlruns
    # then open http://localhost:5000
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from env.traffic_env import TrafficEnv


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: Dict[str, Any], scenario: str, gui: bool) -> TrafficEnv:
    cfg = dict(config)  # shallow copy
    cfg["environment"] = dict(config["environment"])
    cfg["environment"]["route_file"] = f"routes_{scenario}.rou.xml"
    if gui:
        cfg["sumo"] = dict(config["sumo"])
        cfg["sumo"]["gui"] = True
    return TrafficEnv(cfg, render_mode="human" if gui else None)


def flat_params(config: Dict[str, Any], algo: str) -> Dict[str, Any]:
    """Flatten config for MLflow parameter logging."""
    params: Dict[str, Any] = {}
    for section in ["environment", "state", "reward", algo]:
        for k, v in config.get(section, {}).items():
            params[f"{section}.{k}"] = v
    params["training.seed"]     = config["training"]["seed"]
    params["training.episodes"] = config["training"]["num_episodes"]
    return params


# ──────────────────────────────────────────────
#  Resume support
# ──────────────────────────────────────────────
#
# A single training run (500 episodes) can take well over an hour, longer
# than some execution environments allow a single process to run
# uninterrupted. Rather than only checkpointing every save_interval (100)
# episodes and losing everything since, a lightweight "latest" checkpoint +
# JSON progress sidecar is written every latest_interval episodes, and
# --resume picks it back up — so an interruption loses at most
# latest_interval episodes of progress, not the whole run.

def _progress_path(ckpt_dir: str, algo: str) -> str:
    return os.path.join(ckpt_dir, f"{algo}_progress.json")


def _save_progress(ckpt_dir: str, algo: str, episode: int, best_reward: float) -> None:
    with open(_progress_path(ckpt_dir, algo), "w") as f:
        json.dump({"episode": episode, "best_reward": best_reward}, f)


def _load_progress(ckpt_dir: str, algo: str) -> Optional[Tuple[int, float]]:
    path = _progress_path(ckpt_dir, algo)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data["episode"], data["best_reward"]


# ──────────────────────────────────────────────
#  DQN training loop
# ──────────────────────────────────────────────

def _train_dqn(
    env:     TrafficEnv,
    agent:   DQNAgent,
    config:  Dict[str, Any],
    writer:  SummaryWriter,
    run_name: str,
    start_episode: int = 0,
    best_reward: float = -np.inf,
) -> None:
    num_episodes    = config["training"]["num_episodes"]
    log_interval    = config["training"]["log_interval"]
    save_interval   = config["training"]["save_interval"]
    latest_interval = config["training"].get("latest_interval", 20)
    ckpt_dir        = config["training"]["checkpoint_dir"]
    base_seed       = config["training"]["seed"]

    if start_episode >= num_episodes:
        print(f"[DQN] Already complete ({start_episode}/{num_episodes} episodes). Nothing to do.")
        return

    if start_episode > 0:
        print(f"[DQN] Resuming from episode {start_episode}/{num_episodes} "
              f"(best_reward so far: {best_reward:.2f})")

    global_step = 0
    t0          = time.time()

    for episode in range(start_episode, num_episodes):
        # Distinct seed per episode → different (but reproducible) traffic
        state, _ = env.reset(seed=base_seed + episode)
        ep_reward = 0.0
        done      = False

        while not done:
            action     = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done       = terminated or truncated

            # Store only true terminals as done: episodes here end by time
            # limit (truncation), which is not a real terminal state, so the
            # Q-target must still bootstrap V(s') across the episode boundary
            # (Pardo et al. 2018, "Time Limits in Reinforcement Learning").
            agent.store_transition(state, action, reward, next_state, float(terminated))
            result = agent.learn()

            if result:
                writer.add_scalar("train/loss",    result["loss"], global_step)
                mlflow.log_metric("loss",           result["loss"], step=global_step)

            ep_reward  += reward
            state       = next_state
            global_step += 1

        agent.decay_epsilon()   # per-episode decay (see DQNAgent.decay_epsilon)

        metrics      = env.get_episode_metrics()
        agent_metrics = agent.get_metrics()

        # TensorBoard
        writer.add_scalar("train/episode_reward",    ep_reward,                        episode)
        writer.add_scalar("train/epsilon",           agent_metrics.get("epsilon", 0),  episode)
        writer.add_scalar("train/mean_waiting_time", metrics.get("mean_waiting_time", 0), episode)
        writer.add_scalar("train/mean_queue_length", metrics.get("mean_queue_length", 0), episode)

        # MLflow
        mlflow.log_metrics(
            {
                "episode_reward":    ep_reward,
                "epsilon":           agent_metrics.get("epsilon", 0),
                **metrics,
            },
            step=episode,
        )

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(ckpt_dir, f"dqn_best.pt"))

        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"[DQN] Ep {episode+1:4d}/{num_episodes}  "
                f"reward={ep_reward:7.2f}  "
                f"wait={metrics.get('mean_waiting_time', 0):6.1f}s  "
                f"queue={metrics.get('mean_queue_length', 0):5.1f}  "
                f"ε={agent_metrics.get('epsilon', 0):.3f}  "
                f"elapsed={elapsed:.0f}s"
            )

        if (episode + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"dqn_ep{episode+1}.pt")
            agent.save(ckpt_path)
            if config["mlflow"]["log_artifacts"]:
                mlflow.log_artifact(ckpt_path)

        if (episode + 1) % latest_interval == 0:
            agent.save(os.path.join(ckpt_dir, "dqn_latest.pt"))
            _save_progress(ckpt_dir, "dqn", episode + 1, best_reward)

    # Save final
    final_path = os.path.join(ckpt_dir, "dqn_final.pt")
    agent.save(final_path)
    if config["mlflow"]["log_artifacts"]:
        mlflow.log_artifact(final_path)
    _save_progress(ckpt_dir, "dqn", num_episodes, best_reward)
    print(f"\n[DQN] Training complete. Best reward: {best_reward:.2f}")


# ──────────────────────────────────────────────
#  PPO training loop
# ──────────────────────────────────────────────

def _train_ppo(
    env:     TrafficEnv,
    agent:   PPOAgent,
    config:  Dict[str, Any],
    writer:  SummaryWriter,
    run_name: str,
    start_episode: int = 0,
    best_reward: float = -np.inf,
) -> None:
    num_episodes    = config["training"]["num_episodes"]
    log_interval    = config["training"]["log_interval"]
    save_interval   = config["training"]["save_interval"]
    latest_interval = config["training"].get("latest_interval", 20)
    ckpt_dir        = config["training"]["checkpoint_dir"]
    base_seed       = config["training"]["seed"]

    if start_episode >= num_episodes:
        print(f"[PPO] Already complete ({start_episode}/{num_episodes} episodes). Nothing to do.")
        return

    if start_episode > 0:
        print(f"[PPO] Resuming from episode {start_episode}/{num_episodes} "
              f"(best_reward so far: {best_reward:.2f})")

    episode     = start_episode
    global_step = 0
    ep_reward   = 0.0
    t0          = time.time()

    state, _ = env.reset(seed=base_seed + episode)

    while episode < num_episodes:
        # ── Collect rollout ──────────────────────────────────────────────────
        for _ in range(agent.rollout_steps):
            action, log_prob, value = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Time-limit truncation is not a real terminal: preserve the value
            # of the cut-off future by bootstrapping it into the stored reward
            # (partial-episode bootstrapping, Pardo et al. 2018). done=1 still
            # stops GAE from propagating across the episode boundary.
            store_reward = reward
            if truncated and not terminated:
                store_reward += agent.gamma * agent.get_value(next_state)

            agent.store_transition(state, action, store_reward, log_prob, value, float(done))

            ep_reward  += reward
            state       = next_state
            global_step += 1

            if done:
                metrics = env.get_episode_metrics()

                writer.add_scalar("train/episode_reward",    ep_reward,                          episode)
                writer.add_scalar("train/mean_waiting_time", metrics.get("mean_waiting_time", 0), episode)
                writer.add_scalar("train/mean_queue_length", metrics.get("mean_queue_length", 0), episode)

                mlflow.log_metrics(
                    {"episode_reward": ep_reward, **metrics},
                    step=episode,
                )

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    agent.save(os.path.join(ckpt_dir, "ppo_best.pt"))

                if (episode + 1) % log_interval == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[PPO] Ep {episode+1:4d}/{num_episodes}  "
                        f"reward={ep_reward:7.2f}  "
                        f"wait={metrics.get('mean_waiting_time', 0):6.1f}s  "
                        f"queue={metrics.get('mean_queue_length', 0):5.1f}  "
                        f"elapsed={elapsed:.0f}s"
                    )

                if (episode + 1) % save_interval == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"ppo_ep{episode+1}.pt")
                    agent.save(ckpt_path)
                    if config["mlflow"]["log_artifacts"]:
                        mlflow.log_artifact(ckpt_path)

                if (episode + 1) % latest_interval == 0:
                    agent.save(os.path.join(ckpt_dir, "ppo_latest.pt"))
                    _save_progress(ckpt_dir, "ppo", episode + 1, best_reward)

                episode   += 1
                ep_reward  = 0.0
                state, _   = env.reset(seed=base_seed + episode)

                if episode >= num_episodes:
                    break

        # ── Update policy ────────────────────────────────────────────────────
        # Bootstrap the rollout tail with V(current state); ignored via the
        # done mask if the rollout happened to end exactly at an episode end.
        update_metrics = agent.learn(last_value=agent.get_value(state))
        if update_metrics:
            for k, v in update_metrics.items():
                writer.add_scalar(f"train/{k}", v, agent._update_count)
            mlflow.log_metrics(update_metrics, step=agent._update_count)

    final_path = os.path.join(ckpt_dir, "ppo_final.pt")
    agent.save(final_path)
    if config["mlflow"]["log_artifacts"]:
        mlflow.log_artifact(final_path)
    _save_progress(ckpt_dir, "ppo", num_episodes, best_reward)
    print(f"\n[PPO] Training complete. Best reward: {best_reward:.2f}")


# ──────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────

def train(
    config: Dict[str, Any], algo: str, scenario: str = "normal", gui: bool = False,
    resume: bool = False,
) -> None:
    set_seed(config["training"]["seed"])
    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    env    = make_env(config, scenario, gui)
    device = get_device(config)

    # Build environment once to get correct state_size / num_actions
    state, _ = env.reset()
    state_size  = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.close()

    # Re-create env fresh for training
    env = make_env(config, scenario, gui)

    if algo == "dqn":
        agent = DQNAgent(state_size, num_actions, config, device)
    elif algo == "ppo":
        agent = PPOAgent(state_size, num_actions, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose 'dqn' or 'ppo'.")

    start_episode = 0
    best_reward   = -np.inf
    if resume:
        progress = _load_progress(ckpt_dir, algo)
        latest_ckpt = os.path.join(ckpt_dir, f"{algo}_latest.pt")
        if progress is not None and os.path.exists(latest_ckpt):
            start_episode, best_reward = progress
            agent.load(latest_ckpt)
        else:
            print(f"[{algo.upper()}] --resume given but no progress checkpoint found; starting fresh.")

    # ── Tracking setup ───────────────────────────────────────────────────────
    run_name   = f"{algo}_{scenario}_{int(time.time())}"
    tb_log_dir = os.path.join(config["logging"]["tensorboard_dir"], run_name)
    writer     = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs: {tb_log_dir}")
    print(f"Run: tensorboard --logdir {config['logging']['tensorboard_dir']}")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flat_params(config, algo))
        mlflow.log_param("algorithm", algo)
        mlflow.log_param("scenario",  scenario)
        mlflow.log_param("state_size",  state_size)
        mlflow.log_param("num_actions", num_actions)
        print(f"MLflow run id: {run.info.run_id}")

        if algo == "dqn":
            _train_dqn(env, agent, config, writer, run_name, start_episode, best_reward)
        else:
            _train_ppo(env, agent, config, writer, run_name, start_episode, best_reward)

    writer.close()
    env.close()


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL traffic signal agent")
    p.add_argument("--algo",      default="dqn",    choices=["dqn", "ppo"])
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--episodes",  type=int, default=None, help="Override num_episodes")
    p.add_argument("--scenario",  default="normal",
                   choices=["normal", "ns_peak", "ew_peak", "high"])
    p.add_argument("--gui",       action="store_true", help="Open SUMO GUI")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from {algo}_latest.pt / {algo}_progress.json in the "
                        "checkpoint dir if present (saved every training.latest_interval "
                        "episodes). No-op if nothing to resume.")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    config = load_config(args.config)

    if args.episodes:
        config["training"]["num_episodes"] = args.episodes

    print(f"\n{'='*55}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Scenario  : {args.scenario}")
    print(f"  Episodes  : {config['training']['num_episodes']}")
    print(f"  Resume    : {args.resume}")
    print(f"{'='*55}\n")

    train(config, algo=args.algo, scenario=args.scenario, gui=args.gui, resume=args.resume)
