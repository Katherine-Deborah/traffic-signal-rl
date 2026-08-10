"""
watch_controller.py — open sumo-gui, zoomed on the intersection, and play out
one controller's behaviour live so you can screen-record it (e.g. with
ScreenToGif) for the README/portfolio. No automated screenshotting — you do
the recording.

Flow: fast-forward through a quiet warm-up (so there's already traffic when
recording starts) -> countdown so you can arm your recorder -> plays the
"record window" out in near-real time -> leaves the window open at the end
(closes on Ctrl+C or when you close the window yourself).

Usage
-----
    python scripts/watch_controller.py --controller fixed_time
    python scripts/watch_controller.py --controller dqn
    python scripts/watch_controller.py --controller ppo
    python scripts/watch_controller.py --controller max_pressure
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

import sumolib
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from baselines.fixed_time import FixedTimeController
from baselines.max_pressure import MaxPressureController
from env.traffic_env import TrafficEnv

VIEW_ID = "View #0"
GUI_SCHEME = "real world"


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_gui_env(config: Dict[str, Any], scenario: str) -> TrafficEnv:
    cfg = dict(config)
    cfg["environment"] = dict(config["environment"])
    cfg["environment"]["route_file"] = f"routes_{scenario}.rou.xml"
    cfg["sumo"] = dict(config["sumo"])
    cfg["sumo"]["gui"] = True
    return TrafficEnv(cfg, render_mode="human")


def set_gui_view(env: TrafficEnv, zoom_radius: float) -> None:
    net = sumolib.net.readNet(env.net_file)
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    env._traci.gui.setSchema(VIEW_ID, GUI_SCHEME)
    env._traci.gui.setBoundary(
        VIEW_ID, cx - zoom_radius, cy - zoom_radius, cx + zoom_radius, cy + zoom_radius
    )


def build_controller(name: str, env: TrafficEnv, config: Dict[str, Any], device: Any,
                      dqn_path: str, ppo_path: str):
    state_size, num_actions = env.observation_space.shape[0], env.action_space.n
    if name == "fixed_time":
        return FixedTimeController(cycle_time=60, delta_time=config["environment"]["delta_time"],
                                    num_phases=num_actions), False
    if name == "max_pressure":
        return MaxPressureController(env), False
    if name == "dqn":
        agent = DQNAgent(state_size, num_actions, config, device)
        agent.load(dqn_path)
        return agent, False
    if name == "ppo":
        agent = PPOAgent(state_size, num_actions, config, device)
        agent.load(ppo_path)
        return agent, True
    raise ValueError(f"Unknown controller '{name}'")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", required=True,
                    choices=["fixed_time", "max_pressure", "dqn", "ppo"])
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--dqn", default=None)
    p.add_argument("--ppo", default=None)
    p.add_argument("--scenario", default="normal",
                    choices=["normal", "ns_peak", "ew_peak", "high"])
    p.add_argument("--seed", type=int, default=10_000,
                    help="Same seed across controllers = same traffic for each.")
    p.add_argument("--warmup", type=int, default=300, help="Sim-seconds fast-forwarded, unpaced.")
    p.add_argument("--record", type=int, default=90, help="Sim-seconds played out live.")
    p.add_argument("--pace", type=float, default=0.15,
                    help="Real seconds to sleep per sim-second during the record window.")
    p.add_argument("--countdown", type=int, default=6,
                    help="Seconds to wait (after warm-up) before the record window starts.")
    p.add_argument("--zoom-radius", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    device = get_device(config)
    dqn_path = args.dqn or os.path.join(config["training"]["checkpoint_dir"], "dqn_best.pt")
    ppo_path = args.ppo or os.path.join(config["training"]["checkpoint_dir"], "ppo_best.pt")

    env = make_gui_env(config, args.scenario)
    state, _ = env.reset(seed=args.seed)
    set_gui_view(env, args.zoom_radius)
    controller, is_ppo = build_controller(args.controller, env, config, device, dqn_path, ppo_path)
    if hasattr(controller, "reset"):
        controller.reset()

    delta_time = config["environment"]["delta_time"]
    warmup_steps = args.warmup // delta_time
    record_steps = args.record // delta_time

    def act() -> int:
        if is_ppo:
            action, _, _ = controller.select_action(state, explore=False)
        else:
            action = controller.select_action(state, explore=False)
        return action

    print(f"\n[{args.controller}] warming up ({args.warmup}s sim time, unpaced)...")
    for _ in range(warmup_steps):
        action = act()
        state, _, _, _, _ = env.step(action)
        if hasattr(controller, "step"):
            controller.step()

    print(f"[{args.controller}] ready. Arm your screen recorder now.")
    for s in range(args.countdown, 0, -1):
        print(f"  starting in {s}...", flush=True)
        time.sleep(1)

    print(f"[{args.controller}] RECORDING — playing {args.record}s of sim time live.")

    def pace(_seconds: float = args.pace):
        time.sleep(_seconds)

    for _ in range(record_steps):
        action = act()
        state, _, _, _, _ = env.step(action, on_sim_step=pace)
        if hasattr(controller, "step"):
            controller.step()

    print(f"[{args.controller}] done — stop your recording now.")
    print("Leaving the SUMO window open; close it yourself, or Ctrl+C here to force-close it.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
