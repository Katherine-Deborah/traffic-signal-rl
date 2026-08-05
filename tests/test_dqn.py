"""Tests for DQN components: Q-network, replay buffer, epsilon schedule."""

import numpy as np
import torch

from agents.dqn_agent import DQNAgent, QNetwork, ReplayBuffer

STATE_SIZE = 15
NUM_ACTIONS = 2


def _config():
    return {
        "dqn": {
            "lr": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
            "buffer_size": 100,
            "batch_size": 8,
            "target_update": 10,
            "hidden_dim": 32,
            "num_layers": 2,
        }
    }


def _agent():
    return DQNAgent(STATE_SIZE, NUM_ACTIONS, _config(), torch.device("cpu"))


def test_qnetwork_output_shape():
    net = QNetwork(STATE_SIZE, NUM_ACTIONS, hidden_dim=32, num_layers=2)
    out = net(torch.zeros(4, STATE_SIZE))
    assert out.shape == (4, NUM_ACTIONS)


def test_replay_buffer_capacity_eviction():
    buf = ReplayBuffer(capacity=5)
    s = np.zeros(STATE_SIZE, dtype=np.float32)
    for i in range(10):
        buf.push(s, 0, float(i), s, 0.0)
    assert len(buf) == 5
    # Oldest rewards (0–4) evicted; only 5–9 remain
    rewards = {t[2] for t in buf._buf}
    assert rewards == {5.0, 6.0, 7.0, 8.0, 9.0}


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=50)
    s = np.zeros(STATE_SIZE, dtype=np.float32)
    for _ in range(20):
        buf.push(s, 1, 0.5, s, 0.0)
    states, actions, rewards, next_states, dones = buf.sample(8, torch.device("cpu"))
    assert states.shape == (8, STATE_SIZE)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, STATE_SIZE)
    assert dones.shape == (8,)


def test_epsilon_decays_per_call_and_respects_floor():
    agent = _agent()
    assert agent.epsilon == 1.0
    agent.decay_epsilon()
    assert np.isclose(agent.epsilon, 0.99)
    for _ in range(2000):
        agent.decay_epsilon()
    assert agent.epsilon == agent.epsilon_end  # clamped at floor


def test_epsilon_schedule_spans_training():
    """With per-episode decay 0.99, the floor is reached around episode 300 —
    not episode 8, which was the bug with per-step decay."""
    agent = _agent()
    for _ in range(250):
        agent.decay_epsilon()
    assert agent.epsilon > agent.epsilon_end  # still exploring at ep 250
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon == agent.epsilon_end  # floored by ep 350


def test_greedy_action_is_deterministic():
    agent = _agent()
    state = np.random.rand(STATE_SIZE).astype(np.float32)
    a1 = agent.select_action(state, explore=False)
    a2 = agent.select_action(state, explore=False)
    assert a1 == a2
    assert a1 in range(NUM_ACTIONS)


def test_learn_updates_weights():
    agent = _agent()
    s = np.random.rand(STATE_SIZE).astype(np.float32)
    for _ in range(20):
        agent.store_transition(s, 0, -1.0, s, 0.0)
    before = [p.clone() for p in agent.q_net.parameters()]
    result = agent.learn()
    assert result is not None and "loss" in result
    changed = any(
        not torch.equal(b, a) for b, a in zip(before, agent.q_net.parameters())
    )
    assert changed
