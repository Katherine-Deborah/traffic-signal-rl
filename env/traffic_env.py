"""
TrafficEnv — Gymnasium wrapper for a single SUMO traffic intersection.

Observation
-----------
For each of the 4 incoming edges: [queue, density, waiting_time]  (all normalised)
Current green phase (one-hot)
Phase duration (normalised)

Action
------
Discrete(N) — choose which green phase to activate next.
Yellow transitions are inserted automatically; minimum-green enforcement is built-in.

Reward
------
Configurable: negative waiting time, negative queue length, combined, or pressure.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml


class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ──────────────────────────────────────────
    #  Construction
    # ──────────────────────────────────────────

    def __init__(
        self,
        config: Dict[str, Any],
        render_mode: Optional[str] = None,
        label: str = "default",
    ) -> None:
        super().__init__()

        self.config      = config
        self.render_mode = render_mode
        self.label       = label   # unique label per env instance (multi-agent)

        # ── SUMO paths ──────────────────────────────────────────────────────
        sumo_cfg = config["sumo"]
        self.sumo_home   = sumo_cfg["home"]
        self.use_gui     = render_mode == "human" or sumo_cfg.get("gui", False)
        self.step_length = float(sumo_cfg.get("step_length", 1.0))
        self.yellow_time = int(sumo_cfg.get("yellow_time", 3))

        # Ensure SUMO_HOME is set and tools/ is on sys.path before importing traci
        os.environ["SUMO_HOME"] = self.sumo_home
        _tools = os.path.join(self.sumo_home, "tools")
        if _tools not in sys.path:
            sys.path.insert(0, _tools)

        import traci  # noqa: PLC0415  (late import intentional)
        self._traci = traci

        # ── Environment config ───────────────────────────────────────────────
        env_cfg = config["environment"]
        net_dir           = env_cfg["network_dir"]
        self.net_file     = os.path.abspath(os.path.join(net_dir, env_cfg["net_file"]))
        self.route_file   = os.path.abspath(os.path.join(net_dir, env_cfg["route_file"]))
        self.tl_id        = env_cfg["tl_id"]
        self.incoming_edges: List[str] = env_cfg["incoming_edges"]
        self.delta_time   = int(env_cfg["delta_time"])
        self.min_green    = int(env_cfg["min_green"])
        self.max_steps    = int(env_cfg["max_steps"])

        # ── State config ─────────────────────────────────────────────────────
        state_cfg = config["state"]
        self.use_queue   = state_cfg.get("use_queue", True)
        self.use_density = state_cfg.get("use_density", True)
        self.use_waiting = state_cfg.get("use_waiting_time", True)
        self.use_phase   = state_cfg.get("use_phase", True)
        self.normalize   = state_cfg.get("normalize", True)
        self.max_queue   = float(state_cfg.get("max_queue", 30))
        self.max_waiting = float(state_cfg.get("max_waiting", 300.0))

        # ── Reward config ────────────────────────────────────────────────────
        reward_cfg        = config["reward"]
        self.reward_type  = reward_cfg.get("type", "waiting_time")
        self.reward_scale = float(reward_cfg.get("scale", 0.001))
        self.wait_weight  = float(reward_cfg.get("waiting_weight", 0.5))
        self.queue_weight = float(reward_cfg.get("queue_weight", 0.5))

        # ── SUMO binary ──────────────────────────────────────────────────────
        bin_name = "sumo-gui.exe" if self.use_gui else "sumo.exe"
        self._sumo_binary = os.path.join(self.sumo_home, "bin", bin_name)
        if not os.path.exists(self._sumo_binary):
            self._sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        self._sumo_cmd: List[str] = [
            self._sumo_binary,
            "--net-file",             self.net_file,
            "--route-files",          self.route_file,
            "--no-step-log",          "true",
            "--waiting-time-memory",  "3600",
            "--time-to-teleport",     "-1",
            "--collision.action",     "warn",
            "--step-length",          str(self.step_length),
        ]

        # ── Phase info (filled after first SUMO connection) ──────────────────
        self._green_phase_indices: List[int] = []
        self._phase_states:        List[str] = []
        self._num_green_phases: int = config["action"].get("num_phases", 2)

        # ── Spaces (may be resized after _discover_phases) ───────────────────
        self._state_size = self._compute_state_size()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._state_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self._num_green_phases)

        # ── Episode state ────────────────────────────────────────────────────
        self.current_phase_idx: int  = 0
        self.green_duration:    int  = 0   # seconds in current green phase
        self.step_count:        int  = 0
        self._sumo_running:     bool = False

        # ── Metric accumulators (reset each episode) ─────────────────────────
        self._ep_waiting: List[float] = []
        self._ep_queues:  List[float] = []

    # ──────────────────────────────────────────
    #  Gym API
    # ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._close_sumo()

        self._traci.start(self._sumo_cmd, label=self.label)
        self._sumo_running = True

        self._discover_phases()

        # Reset episode counters
        self.current_phase_idx = 0
        self.green_duration    = 0
        self.step_count        = 0
        self._ep_waiting       = []
        self._ep_queues        = []

        # Activate initial phase and warm up
        self._set_green_phase(self.current_phase_idx)
        for _ in range(5):
            self._traci.simulationStep()

        return self._get_state(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        # ── Enforce minimum green time ───────────────────────────────────────
        if self.green_duration < self.min_green and action != self.current_phase_idx:
            action = self.current_phase_idx

        # ── Phase transition ─────────────────────────────────────────────────
        if action != self.current_phase_idx:
            self._set_yellow_phase()
            for _ in range(self.yellow_time):
                self._traci.simulationStep()
            sim_steps = max(1, self.delta_time - self.yellow_time)
            self.green_duration = sim_steps        # reset; newly entered phase
        else:
            sim_steps = self.delta_time
            self.green_duration += self.delta_time

        self._set_green_phase(action)
        for _ in range(sim_steps):
            self._traci.simulationStep()

        self.current_phase_idx = action
        self.step_count       += 1

        # ── Collect outcomes ─────────────────────────────────────────────────
        state   = self._get_state()
        reward  = self._get_reward()
        waiting = self._total_waiting_time()
        queue   = float(self._total_queue())

        self._ep_waiting.append(waiting)
        self._ep_queues.append(queue)

        terminated = False
        truncated  = self.step_count >= self.max_steps

        info: Dict[str, Any] = {
            "waiting_time": waiting,
            "queue_length": queue,
            "step":         self.step_count,
        }
        return state, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_sumo()

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    def _close_sumo(self) -> None:
        if self._sumo_running:
            try:
                self._traci.close()
            except Exception:
                pass
            self._sumo_running = False

    def _discover_phases(self) -> None:
        """
        Read the TL program generated by SUMO/netconvert and identify
        which phases are "green" phases (contain 'G', no 'y').
        Called once after SUMO starts so we work with any network.
        """
        programs = self._traci.trafficlight.getAllProgramLogics(self.tl_id)
        if not programs:
            raise RuntimeError(
                f"No traffic light program found for tl_id='{self.tl_id}'.\n"
                "Run: python network/generate_network.py"
            )

        phases = programs[0].phases
        self._phase_states = [p.state for p in phases]
        self._green_phase_indices = [
            i for i, p in enumerate(phases)
            if "G" in p.state and "y" not in p.state
        ]

        n = len(self._green_phase_indices)
        if n == 0:
            raise RuntimeError(
                "No green phases found in traffic light program. "
                "Check the network file."
            )

        # Resize spaces if actual phase count differs from config
        if n != self._num_green_phases:
            self._num_green_phases = n
            self._state_size = self._compute_state_size()
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self._state_size,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(self._num_green_phases)

    def _set_green_phase(self, phase_idx: int) -> None:
        sumo_idx   = self._green_phase_indices[phase_idx]
        state_str  = self._phase_states[sumo_idx]
        self._traci.trafficlight.setRedYellowGreenState(self.tl_id, state_str)

    def _set_yellow_phase(self) -> None:
        """Build a yellow state from the current green state and apply it."""
        current = self._traci.trafficlight.getRedYellowGreenState(self.tl_id)
        yellow  = current.replace("G", "y").replace("g", "y")
        self._traci.trafficlight.setRedYellowGreenState(self.tl_id, yellow)

    # ──────────────────────────────────────────
    #  State
    # ──────────────────────────────────────────

    def _compute_state_size(self) -> int:
        n = len(self.incoming_edges)
        size = 0
        if self.use_queue:   size += n
        if self.use_density: size += n
        if self.use_waiting: size += n
        if self.use_phase:   size += self._num_green_phases  # one-hot
        size += 1  # phase duration
        return size

    def _get_state(self) -> np.ndarray:
        features: List[float] = []

        for edge in self.incoming_edges:
            if self.use_queue:
                q = self._traci.edge.getLastStepHaltingNumber(edge)
                features.append(min(q / self.max_queue, 1.0) if self.normalize else float(q))

            if self.use_density:
                d = self._traci.edge.getLastStepOccupancy(edge) / 100.0
                features.append(d)

            if self.use_waiting:
                w = self._traci.edge.getWaitingTime(edge)
                features.append(
                    min(w / self.max_waiting, 1.0) if self.normalize else w
                )

        if self.use_phase:
            one_hot = np.zeros(self._num_green_phases, dtype=np.float32)
            one_hot[self.current_phase_idx] = 1.0
            features.extend(one_hot.tolist())

        # Phase duration normalised by 2 minutes (reasonable max green time)
        features.append(min(self.green_duration / 120.0, 1.0))

        return np.array(features, dtype=np.float32)

    # ──────────────────────────────────────────
    #  Reward
    # ──────────────────────────────────────────

    def _get_reward(self) -> float:
        if self.reward_type == "waiting_time":
            raw = -self._total_waiting_time()
        elif self.reward_type == "queue_length":
            raw = -float(self._total_queue())
        elif self.reward_type == "combined":
            raw = -(
                self.wait_weight  * self._total_waiting_time()
                + self.queue_weight * float(self._total_queue())
            )
        elif self.reward_type == "pressure":
            raw = -self._pressure()
        else:
            raw = -self._total_waiting_time()

        return raw * self.reward_scale

    def _total_waiting_time(self) -> float:
        return sum(
            self._traci.edge.getWaitingTime(e) for e in self.incoming_edges
        )

    def _total_queue(self) -> int:
        return sum(
            self._traci.edge.getLastStepHaltingNumber(e) for e in self.incoming_edges
        )

    def _pressure(self) -> float:
        """
        Pressure reward: |queue on red edges  –  queue on green edges|.
        Encourages the agent to serve the most congested direction.
        """
        if not self._green_phase_indices:
            return 0.0

        current_state = self._traci.trafficlight.getRedYellowGreenState(self.tl_id)
        links         = self._traci.trafficlight.getControlledLinks(self.tl_id)

        green_edges: set = set()
        red_edges:   set = set()

        for i, link_group in enumerate(links):
            if i < len(current_state):
                for link in link_group:
                    if link:
                        edge = link[0].split("_")[0]
                        if current_state[i] in ("G", "g"):
                            green_edges.add(edge)
                        elif current_state[i] == "r":
                            red_edges.add(edge)

        incoming = set(self.incoming_edges)
        green_q  = sum(self._traci.edge.getLastStepHaltingNumber(e) for e in green_edges & incoming)
        red_q    = sum(self._traci.edge.getLastStepHaltingNumber(e) for e in red_edges   & incoming)
        return abs(red_q - green_q)

    # ──────────────────────────────────────────
    #  Episode metrics (used by training loop)
    # ──────────────────────────────────────────

    def get_episode_metrics(self) -> Dict[str, float]:
        if not self._ep_waiting:
            return {}
        return {
            "mean_waiting_time": float(np.mean(self._ep_waiting)),
            "max_waiting_time":  float(np.max(self._ep_waiting)),
            "total_waiting_time": float(np.sum(self._ep_waiting)),
            "mean_queue_length": float(np.mean(self._ep_queues)),
            "max_queue_length":  float(np.max(self._ep_queues)),
        }

    def __del__(self) -> None:
        self._close_sumo()
