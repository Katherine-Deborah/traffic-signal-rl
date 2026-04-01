"""
FixedTimeController — deterministic baseline controller.

Each phase gets an equal share of the cycle time (default 60 s, so 30 s per phase).
This mimics real-world pre-timed signals that cannot adapt to traffic conditions.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class FixedTimeController:
    """
    Fixed-time traffic signal controller (baseline).

    Parameters
    ----------
    cycle_time   : total cycle length in seconds
    phase_splits : fraction of cycle assigned to each phase;
                   must sum to 1.0 and match the number of green phases.
                   Default: equal splits.
    delta_time   : seconds per agent step (must match env.delta_time)
    """

    def __init__(
        self,
        cycle_time:   int   = 60,
        phase_splits: Optional[List[float]] = None,
        delta_time:   int   = 5,
        num_phases:   int   = 2,
    ) -> None:
        self.cycle_time   = cycle_time
        self.delta_time   = delta_time
        self.num_phases   = num_phases
        self.phase_splits = phase_splits or [1.0 / num_phases] * num_phases

        assert abs(sum(self.phase_splits) - 1.0) < 1e-6, "phase_splits must sum to 1.0"
        assert len(self.phase_splits) == self.num_phases

        self._step_count: int = 0

    # ── Public API (mirrors agent interface) ─────────────────────────────────

    def select_action(self, state: Any = None, explore: bool = False) -> int:
        """Return the phase index determined by the fixed schedule."""
        sim_time     = self._step_count * self.delta_time
        time_in_cycle = sim_time % self.cycle_time

        cumulative = 0.0
        for i, split in enumerate(self.phase_splits):
            cumulative += split * self.cycle_time
            if time_in_cycle < cumulative:
                return i

        return self.num_phases - 1

    def step(self) -> None:
        """Advance the internal clock by one agent step."""
        self._step_count += 1

    def reset(self) -> None:
        self._step_count = 0

    def get_metrics(self) -> Dict[str, float]:
        return {}

    # ── Convenience ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FixedTimeController":
        env_cfg = config["environment"]
        return cls(
            cycle_time  = 60,
            delta_time  = int(env_cfg["delta_time"]),
            num_phases  = int(env_cfg.get("num_phases", 2)),
        )
