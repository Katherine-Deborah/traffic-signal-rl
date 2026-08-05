"""
MaxPressureController — static (non-learned) baseline based on Max-Pressure
control (Varaiya, 2013). At every decision point, picks the candidate green
phase whose served movements have the largest "pressure": total queued
vehicles on the incoming lanes that phase serves, minus queued vehicles on
the corresponding outgoing lanes.

Max-Pressure is the strongest non-learned baseline in the traffic-signal-RL
literature (used throughout RESCO, PressLight, MPLight, etc.) — it requires
no training and is provably throughput-maximizing under mild assumptions.
Comparing RL agents against it, not just a naive fixed-time schedule, is
the standard fair comparison: beating fixed-time by a wide margin is not
hard, beating Max-Pressure is.

The pressure math is factored into a pure function (`compute_phase_pressures`)
that takes plain data — phase state strings, controlled-link tuples, and a
queue lookup function — so it's unit-testable without a running SUMO/traci
instance. `MaxPressureController` is a thin traci-backed wrapper around it.
"""

from typing import Any, Callable, Dict, List, Sequence, Tuple

LinkTuple = Tuple[str, str, str]  # (fromLane, toLane, viaLane)


def compute_phase_pressures(
    phase_states:     Sequence[str],
    controlled_links: Sequence[Sequence[LinkTuple]],
    queue_fn:          Callable[[str], float],
) -> List[float]:
    """
    phase_states: one RYG state string per *candidate* phase (already
        filtered to green phases only, in action-index order).
    controlled_links: traci.trafficlight.getControlledLinks(tl_id) output —
        indexed by signal/link position, each entry a list of
        (fromLane, toLane, viaLane) tuples controlled by that position.
    queue_fn: lane_id -> queued-vehicle count (e.g. halting number).

    Returns one pressure value per phase, in the same order as phase_states.
    A link position is counted as "served" by a phase if its character in
    that phase's state string is 'G' or 'g' (protected or permitted green).
    """
    pressures: List[float] = []
    for state in phase_states:
        pressure = 0.0
        for i, link_group in enumerate(controlled_links):
            if i >= len(state) or state[i] not in ("G", "g"):
                continue
            for link in link_group:
                if not link:
                    continue
                in_lane, out_lane = link[0], link[1]
                pressure += queue_fn(in_lane) - queue_fn(out_lane)
        pressures.append(pressure)
    return pressures


class MaxPressureController:
    """
    Mirrors the FixedTimeController/agent interface used by
    training/evaluate.py: select_action(state, explore) -> int.

    Needs live access to the environment's traci connection and its
    discovered phase structure, so it's constructed with a reference to an
    already-reset TrafficEnv rather than operating on the state vector alone
    (the state vector doesn't carry per-outgoing-lane queue info).
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def select_action(self, state: Any = None, explore: bool = False) -> int:
        env = self.env
        phase_states = [env._phase_states[idx] for idx in env._green_phase_indices]
        links = env._traci.trafficlight.getControlledLinks(env.tl_id)
        queue_fn = env._traci.lane.getLastStepHaltingNumber

        pressures = compute_phase_pressures(phase_states, links, queue_fn)
        return max(range(len(pressures)), key=lambda i: pressures[i])

    def get_metrics(self) -> Dict[str, float]:
        return {}
