"""Random visual token pruning. Baseline floor."""
import torch
from .base import Pruner


class RandomPruner(Pruner):
    """Picks visual tokens uniformly at random. Seeded for reproducibility.

    The generator state advances per call, so different samples in a benchmark
    get different random selections (we don't want to always drop the same
    indices - that would be informative-but-deterministic, not random).
    """

    def __init__(self, keep_ratio: float, seed: int = 42):
        super().__init__(keep_ratio)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def select_indices(self, hidden_states, visual_mask, text_question_mask):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        keep_masks = []
        for b in range(batch_size):
            n_visual = int(visual_mask[b].sum().item())
            n_keep = self.n_keep(n_visual)
            # randperm uses the seeded generator on CPU, then move to device
            perm = torch.randperm(n_visual, generator=self.generator)
            keep_mask = torch.zeros(n_visual, dtype=torch.bool)
            keep_mask[perm[:n_keep]] = True
            keep_masks.append(keep_mask.to(device))
        return torch.stack(keep_masks)
