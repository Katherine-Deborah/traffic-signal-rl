"""
BaseAgent — common interface for DQN and PPO agents.
Also provides get_device() for automatic GPU/CPU selection.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


# ──────────────────────────────────────────────
#  Device selection
# ──────────────────────────────────────────────

def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Resolve compute device from config["training"]["device"].
      "auto"  → CUDA if available, else CPU
      "cuda"  → CUDA (warns + falls back if unavailable)
      "cpu"   → always CPU
    """
    preference = config.get("training", {}).get("device", "auto")

    if preference == "cpu":
        device = torch.device("cpu")
        print("Device: CPU (forced by config)")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: GPU — {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        if preference == "cuda":
            print("Warning: CUDA requested but no GPU found. Falling back to CPU.")
        else:
            print("Device: CPU (no GPU detected)")
        device = torch.device("cpu")

    return device


# ──────────────────────────────────────────────
#  Abstract base
# ──────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Shared interface used by the training loop.

    Concrete classes must implement:
        select_action(state, explore) → int
        learn()                       → dict | None
        save(path)
        load(path)
        get_metrics()                 → dict
    """

    def __init__(
        self,
        state_size:  int,
        num_actions: int,
        config:      Dict[str, Any],
        device:      torch.device,
    ) -> None:
        self.state_size  = state_size
        self.num_actions = num_actions
        self.config      = config
        self.device      = device

    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> Any:
        """Return action (and optionally log_prob, value for PPO)."""

    @abstractmethod
    def learn(self) -> Optional[Dict[str, float]]:
        """Perform a learning update. Returns a dict of training metrics or None."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights and optimiser state to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model weights and optimiser state from disk."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return loggable metrics (epsilon, loss, etc.) for the current step."""

    # ── Shared utility ───────────────────────────────────────────────────────

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(x).to(self.device)
