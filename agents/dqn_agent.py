"""
DQN Agent — Deep Q-Network for traffic signal control.

Architecture: FC → ReLU → FC → ReLU → Q-values
Training:     Experience replay + target network + epsilon-greedy
"""

import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent


# ──────────────────────────────────────────────
#  Network
# ──────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_size: int, num_actions: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        dims  = [state_size] + [hidden_dim] * num_layers + [num_actions]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


# ──────────────────────────────────────────────
#  Replay buffer
# ──────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       float,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────

class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_size:  int,
        num_actions: int,
        config:      Dict[str, Any],
        device:      torch.device,
    ) -> None:
        super().__init__(state_size, num_actions, config, device)

        dqn_cfg = config["dqn"]

        # Hyperparameters
        self.lr            = float(dqn_cfg["lr"])
        self.gamma         = float(dqn_cfg["gamma"])
        self.epsilon       = float(dqn_cfg["epsilon_start"])
        self.epsilon_end   = float(dqn_cfg["epsilon_end"])
        self.epsilon_decay = float(dqn_cfg["epsilon_decay"])
        self.batch_size    = int(dqn_cfg["batch_size"])
        self.target_update = int(dqn_cfg["target_update"])

        hidden_dim  = int(dqn_cfg["hidden_dim"])
        num_layers  = int(dqn_cfg.get("num_layers", 2))

        # Networks
        self.q_net = QNetwork(state_size, num_actions, hidden_dim, num_layers).to(device)
        self.tgt_net = QNetwork(state_size, num_actions, hidden_dim, num_layers).to(device)
        self.tgt_net.load_state_dict(self.q_net.state_dict())
        self.tgt_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay    = ReplayBuffer(int(dqn_cfg["buffer_size"]))

        # Counters and metric history
        self._learn_steps: int         = 0
        self._losses:      List[float] = []

    # ── Interface ────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            q = self.q_net(self._to_tensor(state).unsqueeze(0))
        return int(q.argmax().item())

    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       float,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size, self.device
        )

        # Current Q
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q  (no gradient)
        with torch.no_grad():
            max_next_q = self.tgt_net(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)   # Huber loss — more stable than MSE

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._learn_steps += 1

        # Hard-update target network
        if self._learn_steps % self.target_update == 0:
            self.tgt_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        loss_val = loss.item()
        self._losses.append(loss_val)
        return {"loss": loss_val}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net":       self.q_net.state_dict(),
                "tgt_net":     self.tgt_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "epsilon":     self.epsilon,
                "learn_steps": self._learn_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.tgt_net.load_state_dict(ckpt["tgt_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon       = ckpt["epsilon"]
        self._learn_steps  = ckpt["learn_steps"]

    def get_metrics(self) -> Dict[str, float]:
        recent = self._losses[-100:] if self._losses else [0.0]
        return {
            "epsilon":   self.epsilon,
            "mean_loss": float(np.mean(recent)),
        }
