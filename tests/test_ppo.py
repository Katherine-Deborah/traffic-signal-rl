"""Tests for PPO components: actor-critic shapes and GAE correctness."""

import numpy as np
import torch

from agents.ppo_agent import ActorCritic, PPOAgent

STATE_SIZE = 15
NUM_ACTIONS = 2


def _config(rollout_steps=4, gamma=1.0, gae_lambda=1.0):
    return {
        "ppo": {
            "lr": 3e-4,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "update_epochs": 2,
            "rollout_steps": rollout_steps,
            "mini_batch_size": 2,
            "hidden_dim": 32,
            "num_layers": 2,
        }
    }


def _agent(**kwargs):
    return PPOAgent(STATE_SIZE, NUM_ACTIONS, _config(**kwargs), torch.device("cpu"))


def test_actor_critic_shapes():
    net = ActorCritic(STATE_SIZE, NUM_ACTIONS, hidden_dim=32, num_layers=2)
    logits, value = net(torch.zeros(4, STATE_SIZE))
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4, 1)


def test_select_action_returns_triple():
    agent = _agent()
    state = np.random.rand(STATE_SIZE).astype(np.float32)
    action, log_prob, value = agent.select_action(state, explore=True)
    assert action in range(NUM_ACTIONS)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_gae_undiscounted_no_baseline_is_reward_to_go():
    """With gamma=1, lambda=1, V=0 everywhere and no dones, the advantage at t
    must equal the sum of rewards from t onward plus the bootstrap value."""
    agent = _agent(rollout_steps=4, gamma=1.0, gae_lambda=1.0)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    values = torch.zeros(4)
    dones = torch.zeros(4)

    adv, ret = agent._compute_gae(rewards, values, dones, last_value=10.0)

    # advantage[t] = sum(rewards[t:]) + last_value
    expected = torch.tensor([1 + 2 + 3 + 4 + 10, 2 + 3 + 4 + 10, 3 + 4 + 10, 4 + 10.0])
    assert torch.allclose(adv, expected)
    assert torch.allclose(ret, expected)  # returns = adv + values, values are 0


def test_gae_done_stops_propagation():
    """A done at t=1 must prevent rewards after it from leaking into t<=1."""
    agent = _agent(rollout_steps=4, gamma=1.0, gae_lambda=1.0)
    rewards = torch.tensor([1.0, 2.0, 100.0, 100.0])
    values = torch.zeros(4)
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0])

    adv, _ = agent._compute_gae(rewards, values, dones, last_value=0.0)

    assert torch.allclose(adv[:2], torch.tensor([3.0, 2.0]))  # episode 1: 1+2, 2
    assert torch.allclose(adv[2:], torch.tensor([200.0, 100.0]))  # episode 2


def test_gae_bootstraps_rollout_tail():
    """The final transition must use last_value, not 0 — the rollout cutting
    off mid-episode is not a terminal state."""
    agent = _agent(rollout_steps=2, gamma=0.5, gae_lambda=1.0)
    rewards = torch.tensor([0.0, 0.0])
    values = torch.zeros(2)
    dones = torch.zeros(2)

    adv_boot, _ = agent._compute_gae(rewards, values, dones, last_value=8.0)
    adv_zero, _ = agent._compute_gae(rewards, values, dones, last_value=0.0)

    assert torch.allclose(adv_boot, torch.tensor([2.0, 4.0]))  # 0.5^2*8, 0.5*8
    assert torch.allclose(adv_zero, torch.zeros(2))


def test_learn_only_when_buffer_full():
    agent = _agent(rollout_steps=4)
    s = np.random.rand(STATE_SIZE).astype(np.float32)
    agent.store_transition(s, 0, 0.0, -0.5, 0.0, 0.0)
    assert agent.learn() is None  # not full yet
    for _ in range(3):
        agent.store_transition(s, 0, 0.0, -0.5, 0.0, 0.0)
    metrics = agent.learn()
    assert metrics is not None
    assert {"policy_loss", "value_loss", "entropy"} <= metrics.keys()
