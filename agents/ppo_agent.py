"""
PPO Agent — Proximal Policy Optimisation for traffic signal control.

Architecture: Shared feature extractor → Actor head (action logits) + Critic head (state value)
Training:     GAE advantage estimation, clipped surrogate objective, multiple update epochs
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent


# ──────────────────────────────────────────────
#  Actor-Critic network
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size:  int,
        num_actions: int,
        hidden_dim:  int,
        num_layers:  int,
    ) -> None:
        super().__init__()

        # Shared trunk
        trunk: List[nn.Module] = [nn.Linear(state_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            trunk += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)

        # Separate heads
        self.actor  = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.actor(features), self.critic(features)

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(state)
        dist     = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# ──────────────────────────────────────────────
#  Rollout buffer
# ──────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, size: int, state_size: int, device: torch.device) -> None:
        self.size       = size
        self.device     = device
        self.states     = np.zeros((size, state_size), dtype=np.float32)
        self.actions    = np.zeros(size, dtype=np.int64)
        self.rewards    = np.zeros(size, dtype=np.float32)
        self.log_probs  = np.zeros(size, dtype=np.float32)
        self.values     = np.zeros(size, dtype=np.float32)
        self.dones      = np.zeros(size, dtype=np.float32)
        self._ptr       = 0

    def push(
        self,
        state:    np.ndarray,
        action:   int,
        reward:   float,
        log_prob: float,
        value:    float,
        done:     float,
    ) -> None:
        if self._ptr < self.size:
            self.states[self._ptr]    = state
            self.actions[self._ptr]   = action
            self.rewards[self._ptr]   = reward
            self.log_probs[self._ptr] = log_prob
            self.values[self._ptr]    = value
            self.dones[self._ptr]     = done
            self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr >= self.size

    def get(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.FloatTensor(self.states).to(self.device),
            torch.LongTensor(self.actions).to(self.device),
            torch.FloatTensor(self.rewards).to(self.device),
            torch.FloatTensor(self.log_probs).to(self.device),
            torch.FloatTensor(self.values).to(self.device),
            torch.FloatTensor(self.dones).to(self.device),
        )

    def clear(self) -> None:
        self._ptr = 0


# ──────────────────────────────────────────────
#  PPO Agent
# ──────────────────────────────────────────────

class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_size:  int,
        num_actions: int,
        config:      Dict[str, Any],
        device:      torch.device,
    ) -> None:
        super().__init__(state_size, num_actions, config, device)

        ppo_cfg = config["ppo"]

        # Hyperparameters
        self.lr            = float(ppo_cfg["lr"])
        self.gamma         = float(ppo_cfg["gamma"])
        self.gae_lambda    = float(ppo_cfg["gae_lambda"])
        self.clip_epsilon  = float(ppo_cfg["clip_epsilon"])
        self.entropy_coef  = float(ppo_cfg["entropy_coef"])
        self.value_coef    = float(ppo_cfg["value_coef"])
        self.max_grad_norm = float(ppo_cfg["max_grad_norm"])
        self.update_epochs = int(ppo_cfg["update_epochs"])
        self.rollout_steps = int(ppo_cfg["rollout_steps"])
        self.mini_batch    = int(ppo_cfg["mini_batch_size"])

        hidden_dim = int(ppo_cfg["hidden_dim"])
        num_layers = int(ppo_cfg.get("num_layers", 2))

        self.actor_critic = ActorCritic(
            state_size, num_actions, hidden_dim, num_layers
        ).to(device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)
        self.buffer    = RolloutBuffer(self.rollout_steps, state_size, device)

        # Counters and metric history
        self._update_count:   int         = 0
        self._policy_losses:  List[float] = []
        self._value_losses:   List[float] = []
        self._entropies:      List[float] = []

    # ── Interface ────────────────────────────────────────────────────────────

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> Tuple[int, float, float]:
        """
        Returns (action, log_prob, value).
        The training loop must unpack all three; the Gym step uses only action.
        """
        with torch.no_grad():
            s = self._to_tensor(state).unsqueeze(0)
            if explore:
                action, log_prob, _, value = self.actor_critic.get_action_and_value(s)
            else:
                logits, value = self.actor_critic(s)
                action   = logits.argmax(dim=-1)
                log_prob = Categorical(logits=logits).log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(
        self,
        state:    np.ndarray,
        action:   int,
        reward:   float,
        log_prob: float,
        value:    float,
        done:     float,
    ) -> None:
        self.buffer.push(state, action, reward, log_prob, value, done)

    def learn(self) -> Optional[Dict[str, float]]:
        if not self.buffer.is_full():
            return None

        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        # ── GAE advantage estimation ─────────────────────────────────────────
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Mini-batch updates ───────────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        indices = np.arange(self.rollout_steps)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.rollout_steps, self.mini_batch):
                mb = indices[start : start + self.mini_batch]

                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    states[mb], actions[mb]
                )

                ratio  = (new_log_probs - old_log_probs[mb]).exp()
                adv_mb = advantages[mb]

                # Clipped surrogate objective
                surr1  = ratio * adv_mb
                surr2  = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipping optional but commonly used)
                value_loss = F.mse_loss(new_values, returns[mb])

                loss = (
                    policy_loss
                    + self.value_coef   * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates         += 1

        self.buffer.clear()
        self._update_count += 1

        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss":  total_value_loss  / n_updates,
            "entropy":     total_entropy     / n_updates,
        }
        self._policy_losses.append(metrics["policy_loss"])
        self._value_losses.append(metrics["value_loss"])
        self._entropies.append(metrics["entropy"])
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer":    self.optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_count = ckpt["update_count"]

    def get_metrics(self) -> Dict[str, float]:
        recent_p = self._policy_losses[-100:] if self._policy_losses else [0.0]
        recent_v = self._value_losses[-100:]  if self._value_losses  else [0.0]
        recent_e = self._entropies[-100:]     if self._entropies      else [0.0]
        return {
            "policy_loss": float(np.mean(recent_p)),
            "value_loss":  float(np.mean(recent_v)),
            "entropy":     float(np.mean(recent_e)),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
        dones:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae        = 0.0

        for t in reversed(range(self.rollout_steps)):
            next_val  = 0.0 if t == self.rollout_steps - 1 else values[t + 1].item()
            delta     = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae       = float(delta) + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns
