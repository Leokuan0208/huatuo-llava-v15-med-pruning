"""Nested random visual-token pruning.

Unlike RandomPruner (which advances a single generator per call, giving an
INDEPENDENT random subset at each keep-ratio), this pruner derives every
keep-ratio's selection from the SAME per-sample master permutation. The
kept set at a lower keep-ratio is therefore a strict SUBSET of the kept set
at a higher keep-ratio for the same sample.

Why: for the evidence-curve / flip analysis, nested budgets isolate "less
evidence" from "different evidence" — shrinking kr only ever REMOVES tokens,
never swaps them. This makes a flip across budgets attributable to evidence
QUANTITY rather than evidence CONTENT.

Reproducibility: the master permutation for sample i is seeded as
(base_seed * LARGE + i), so:
  - the same sample gets the same master permutation across different kr runs
    (enabling true nesting across the kr sweep), and
  - the whole run is deterministic given base_seed.

The per-sample index is tracked internally; it increments once per
select_indices call, matching the per-sample call pattern of the eval loop
(batch_size=1). If the eval batching changes, this counter must be revisited.
"""
import torch
from .base import Pruner


class NestedRandomPruner(Pruner):

    def __init__(self, keep_ratio: float, seed: int = 42):
        super().__init__(keep_ratio)
        self.base_seed = seed
        self._sample_counter = 0

    @property
    def name(self) -> str:
        return f"NestedRandomPruner_kr{self.keep_ratio:.2f}"

    @torch.no_grad()
    def select_indices(self, hidden_states, visual_mask, text_question_mask):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        keep_masks = []
        for b in range(batch_size):
            n_visual = int(visual_mask[b].sum().item())
            n_keep = self.n_keep(n_visual)

            # Per-sample seed: identical for this sample across ALL kr runs,
            # so prefixes nest. LARGE stride avoids collisions between
            # base_seed and sample index.
            per_sample_seed = self.base_seed * 1_000_003 + self._sample_counter
            g = torch.Generator()
            g.manual_seed(per_sample_seed)

            perm = torch.randperm(n_visual, generator=g)   # the master permutation
            keep_mask = torch.zeros(n_visual, dtype=torch.bool)
            keep_mask[perm[:n_keep]] = True                 # prefix = nested subset
            keep_masks.append(keep_mask.to(device))

            self._sample_counter += 1

        return torch.stack(keep_masks)
