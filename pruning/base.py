"""Abstract base class for visual token pruning strategies.

A Pruner receives:
  - hidden_states after LLM layer 0 (shape [batch, seq, hidden])
  - a boolean mask saying which positions are visual tokens
  - a boolean mask saying which positions are the question text

It returns a boolean mask of which visual tokens to keep. The patcher handles
the actual sequence-level deletion - pruners just decide what to keep.

We assume batch_size=1 throughout (HuatuoGPT's eval.py is hardcoded to batch_size=1).
The interface is batch-shaped anyway so we can revisit later if needed.
"""
from abc import ABC, abstractmethod
import torch


class Pruner(ABC):
    """Subclass and implement select_indices()."""

    def __init__(self, keep_ratio: float):
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.keep_ratio = keep_ratio

    @abstractmethod
    def select_indices(
        self,
        hidden_states: torch.Tensor,         # [batch, seq, hidden]
        visual_mask: torch.Tensor,           # [batch, seq] bool, True at visual positions
        text_question_mask: torch.Tensor,    # [batch, seq] bool, True at question positions
    ) -> torch.Tensor:
        """Return a bool tensor [batch, n_visual] of which visual tokens to keep."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_kr{self.keep_ratio:.2f}"

    def n_keep(self, n_visual: int) -> int:
        """How many to keep, with a floor of 1 (can't keep 0 tokens)."""
        return max(1, round(n_visual * self.keep_ratio))
