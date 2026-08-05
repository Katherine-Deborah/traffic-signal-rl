"""Tests for the fixed-time baseline controller."""

import pytest

from baselines.fixed_time import FixedTimeController


def _make(cycle=60, delta=5, phases=2):
    return FixedTimeController(cycle_time=cycle, delta_time=delta, num_phases=phases)


def test_equal_split_schedule():
    """60 s cycle, 5 s steps, 2 phases → phase 0 for steps 0–5, phase 1 for 6–11."""
    ctrl = _make()
    actions = []
    for _ in range(12):
        actions.append(ctrl.select_action())
        ctrl.step()
    assert actions == [0] * 6 + [1] * 6


def test_cycle_wraps():
    ctrl = _make()
    for _ in range(12):  # one full cycle
        ctrl.select_action()
        ctrl.step()
    assert ctrl.select_action() == 0  # back to phase 0


def test_reset():
    ctrl = _make()
    for _ in range(8):
        ctrl.step()
    ctrl.reset()
    assert ctrl.select_action() == 0


def test_custom_splits():
    """75/25 split: phase 0 for 45 s (steps 0–8), phase 1 for 15 s (steps 9–11)."""
    ctrl = FixedTimeController(
        cycle_time=60, delta_time=5, num_phases=2, phase_splits=[0.75, 0.25]
    )
    actions = []
    for _ in range(12):
        actions.append(ctrl.select_action())
        ctrl.step()
    assert actions == [0] * 9 + [1] * 3


def test_invalid_splits_rejected():
    with pytest.raises(AssertionError):
        FixedTimeController(num_phases=2, phase_splits=[0.7, 0.7])
