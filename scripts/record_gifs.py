"""
record_gifs.py — capture a short SUMO-GUI clip of each controller and save it
as a looping GIF, for the README / portfolio.

For a fair visual comparison, every controller faces the *same* traffic
realisation (fixed eval seed, scenario "normal" by default): each episode is
fast-forwarded through a warm-up window (so queues have actually built up)
before frames start being captured, then a short window of simulation is
recorded frame-by-frame via traci.gui.screenshot().

Usage
-----
    python scripts/record_gifs.py                       # all 4 controllers
    python scripts/record_gifs.py --controllers dqn,ppo
    python scripts/record_gifs.py --warmup 300 --record 90 --fps 12
"""

import argparse
import os
import shutil
import time
import sys
from typing import Any, Callable, Dict, List, Optional

import sumolib
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import get_device
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from baselines.fixed_time import FixedTimeController
from baselines.max_pressure import MaxPressureController
from env.traffic_env import TrafficEnv

VIEW_ID = "View #0"
GUI_SCHEME = "real world"

CONTROLLER_LABELS = {
    "fixed_time": "Fixed-time",
    "max_pressure": "Max-Pressure",
    "dqn": "DQN",
    "ppo": "PPO",
}


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
    """Zoom the SUMO-GUI view tightly around the intersection and switch to
    a more portfolio-friendly colour scheme than the plain-white default."""
    net = sumolib.net.readNet(env.net_file)
    xmin, ymin, xmax, ymax = net.getBoundary()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    env._traci.gui.setSchema(VIEW_ID, GUI_SCHEME)
    env._traci.gui.setBoundary(
        VIEW_ID, cx - zoom_radius, cy - zoom_radius, cx + zoom_radius, cy + zoom_radius
    )


def build_controller(
    name: str, env: TrafficEnv, config: Dict[str, Any], device: Any,
    dqn_path: str, ppo_path: str,
) -> tuple:
    """Returns (controller, is_ppo)."""
    state_size  = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if name == "fixed_time":
        return FixedTimeController(
            cycle_time=60, delta_time=config["environment"]["delta_time"],
            num_phases=num_actions,
        ), False
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


def record_one(
    name: str,
    config: Dict[str, Any],
    device: Any,
    scenario: str,
    seed: int,
    warmup_seconds: int,
    record_seconds: int,
    frame_dir: str,
    dqn_path: str,
    ppo_path: str,
    frame_width: int,
    frame_height: int,
    zoom_radius: float,
) -> List[str]:
    """Run one episode, capturing frames only during the recording window.
    Returns the list of frame file paths, in order."""
    os.makedirs(frame_dir, exist_ok=True)
    for f in os.listdir(frame_dir):
        os.remove(os.path.join(frame_dir, f))

    env = make_gui_env(config, scenario)
    state, _ = env.reset(seed=seed)
    set_gui_view(env, zoom_radius)
    controller, is_ppo = build_controller(name, env, config, device, dqn_path, ppo_path)
    if hasattr(controller, "reset"):
        controller.reset()

    delta_time = config["environment"]["delta_time"]
    warmup_steps = warmup_seconds // delta_time
    record_steps = record_seconds // delta_time

    frames: List[str] = []
    frame_idx = 0

    def capture() -> None:
        nonlocal frame_idx
        path = os.path.join(frame_dir, f"frame_{frame_idx:05d}.png")
        env._traci.gui.screenshot(VIEW_ID, path, frame_width, frame_height)
        frames.append(path)
        frame_idx += 1
        # sumo-gui's render/event loop needs a little breathing room between
        # back-to-back traci calls when driven non-interactively, or it can
        # stop servicing the socket entirely (observed hanging on Windows).
        time.sleep(0.03)

    step = 0
    done = False
    while not done:
        if is_ppo:
            action, _, _ = controller.select_action(state, explore=False)
        else:
            action = controller.select_action(state, explore=False)

        recording = warmup_steps <= step < warmup_steps + record_steps
        cb: Optional[Callable[[], None]] = capture if recording else None
        state, _, terminated, truncated, _ = env.step(action, on_sim_step=cb)
        done = terminated or truncated

        if hasattr(controller, "step"):
            controller.step()

        step += 1
        if step >= warmup_steps + record_steps:
            break

    env.close()
    # traci.gui.screenshot() saves asynchronously (queued for "the next
    # simulationStep"); very occasionally a requested frame never actually
    # lands on disk before the sumo-gui process goes away. Drop any that
    # didn't materialise rather than fail the whole recording over it.
    written = [p for p in frames if os.path.exists(p)]
    if len(written) < len(frames):
        print(f"    ({len(frames) - len(written)} of {len(frames)} frames didn't "
              f"write to disk in time — skipping them)")
    return written


def frames_to_gif(frames: List[str], out_path: str, fps: int) -> None:
    images = [Image.open(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    duration_ms = int(1000 / fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"    saved {out_path}  ({len(images)} frames, {size_kb:.0f} KB)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record per-controller SUMO GIFs")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--dqn", default=None, help="Path to DQN checkpoint")
    p.add_argument("--ppo", default=None, help="Path to PPO checkpoint")
    p.add_argument("--scenario", default="normal",
                    choices=["normal", "ns_peak", "ew_peak", "high"])
    p.add_argument("--seed", type=int, default=10_000,
                    help="Same seed used for every controller, so they face identical traffic.")
    p.add_argument("--warmup", type=int, default=300,
                    help="Sim-seconds to fast-forward before recording starts.")
    p.add_argument("--record", type=int, default=90,
                    help="Sim-seconds of footage to capture.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--zoom-radius", type=float, default=120.0,
                    help="Metres from the intersection centre shown in frame.")
    p.add_argument("--controllers", default="fixed_time,max_pressure,dqn,ppo")
    p.add_argument("--out-dir", default="results/gifs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    device = get_device(config)

    dqn_path = args.dqn or os.path.join(config["training"]["checkpoint_dir"], "dqn_best.pt")
    ppo_path = args.ppo or os.path.join(config["training"]["checkpoint_dir"], "ppo_best.pt")

    scratch = os.path.join(args.out_dir, "_frames")
    names = [n.strip() for n in args.controllers.split(",") if n.strip()]

    for name in names:
        label = CONTROLLER_LABELS.get(name, name)
        print(f"\n[{label}] recording {args.record}s of sim time "
              f"(after {args.warmup}s warm-up)...")
        frames = record_one(
            name, config, device, args.scenario, args.seed,
            args.warmup, args.record, scratch, dqn_path, ppo_path,
            args.width, args.height, args.zoom_radius,
        )
        if not frames:
            print(f"    [{label}] no frames captured — skipping GIF.")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.gif")
        frames_to_gif(frames, out_path, args.fps)

    shutil.rmtree(scratch, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
