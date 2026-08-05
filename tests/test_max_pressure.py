"""Tests for the Max-Pressure baseline's pure pressure computation."""

from baselines.max_pressure import compute_phase_pressures


def _queue_fn(queues):
    """Build a queue_fn from a {lane_id: count} dict."""
    return lambda lane: queues.get(lane, 0.0)


def test_picks_phase_with_larger_incoming_queue():
    # Phase A serves lane "N_in" (queue 10) -> "N_out" (queue 0): pressure 10
    # Phase B serves lane "E_in" (queue 2)  -> "E_out" (queue 0): pressure 2
    phase_states = ["Gr", "rG"]
    controlled_links = [
        [("N_in", "N_out", "")],
        [("E_in", "E_out", "")],
    ]
    queues = {"N_in": 10, "E_in": 2}
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn(queues))
    assert pressures == [10.0, 2.0]


def test_subtracts_downstream_queue():
    # Same incoming queue, but phase A's outgoing lane is jammed -> lower pressure
    phase_states = ["Gr", "rG"]
    controlled_links = [
        [("N_in", "N_out", "")],
        [("E_in", "E_out", "")],
    ]
    queues = {"N_in": 10, "N_out": 8, "E_in": 10, "E_out": 0}
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn(queues))
    assert pressures == [2.0, 10.0]  # phase B now clearly preferred


def test_permitted_green_counted_same_as_protected():
    # lowercase 'g' (permitted/yielding) must count the same as uppercase 'G'
    phase_states = ["g"]
    controlled_links = [[("N_in", "N_out", "")]]
    queues = {"N_in": 5}
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn(queues))
    assert pressures == [5.0]


def test_red_links_not_counted():
    phase_states = ["Gr"]
    controlled_links = [
        [("N_in", "N_out", "")],
        [("E_in", "E_out", "")],  # red ('r') in this phase, must be excluded
    ]
    queues = {"N_in": 3, "E_in": 999}
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn(queues))
    assert pressures == [3.0]  # E_in's huge queue must not leak in


def test_multiple_links_at_same_index_are_summed():
    # A single signal index can control >1 physical lane connection (e.g. a
    # shared through+right lane) — both must contribute to that phase's pressure.
    phase_states = ["G"]
    controlled_links = [
        [("N_in", "N_out_s", ""), ("N_in", "N_out_r", "")],
    ]
    queues = {"N_in": 4, "N_out_s": 1, "N_out_r": 1}
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn(queues))
    # pressure = (4-1) + (4-1) = 6
    assert pressures == [6.0]


def test_empty_link_group_ignored():
    phase_states = ["G"]
    controlled_links = [[]]  # signal index with no physical connection
    pressures = compute_phase_pressures(phase_states, controlled_links, _queue_fn({}))
    assert pressures == [0.0]
