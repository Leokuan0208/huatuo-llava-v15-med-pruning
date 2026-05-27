"""GridPrune-style visual token pruning.

Implements the "guide-globally, select-locally" approach from:
  Duan et al. "GridPrune: From 'Where to Look' to 'What to Select' in
  Visual Token Pruning for MLLMs" arxiv:2511.10081 (Nov 2025)

Two-stage per image:
  1. Score every token: fused = α·text_relevance + (1-α)·visual_saliency
  2. Partition into M×M zones; allocate per-zone budget by aggregate
     text_relevance (proportional + largest-remainder rounding); local
     top-K within each zone by fused score.

Saliency proxy note: the paper uses CLIP CLS-to-patches attention from
the last vision-encoder layer for visual_saliency. We use L2 norm of
the post-projector visual token as a proxy, which is faster to ship and
is a known saliency signal in the pruning literature (high-norm tokens
correlate with content-bearing patches). A future ablation can hook the
vision tower and swap in true CLS attention.

Spatial layout: visual tokens are flattened row-major from a 24×24 grid
(LLaVA-v1.5 vision tower output).
"""
import math
import torch
import torch.nn.functional as F
from .base import Pruner


def _compute_zone_indices(n_visual, block_size, device):
    """Zone index per token, assuming row-major 24×24 flatten.
    Returns (zone_idx [n_visual], n_zones)."""
    grid_side = int(math.sqrt(n_visual))
    assert grid_side * grid_side == n_visual, (
        f"n_visual={n_visual} is not a perfect square; can't infer grid")
    assert grid_side % block_size == 0, (
        f"grid_side={grid_side} must be divisible by block_size={block_size}")
    n_zones_per_side = grid_side // block_size
    indices = torch.arange(n_visual, device=device)
    row = indices // grid_side
    col = indices % grid_side
    zone_idx = (row // block_size) * n_zones_per_side + (col // block_size)
    return zone_idx, n_zones_per_side * n_zones_per_side


def _allocate_budget(zone_scores, target_K, n_zones):
    """Largest-remainder allocation of target_K across n_zones.

    Falls back to uniform if all scores are non-positive.
    Returns long tensor [n_zones] summing exactly to target_K.
    """
    device = zone_scores.device
    pos_scores = torch.clamp(zone_scores, min=0.0)

    if pos_scores.sum() <= 1e-8:
        per_zone = target_K // n_zones
        budgets = torch.full((n_zones,), per_zone, dtype=torch.long, device=device)
        shortfall = target_K - budgets.sum().item()
        if shortfall > 0:
            budgets[:shortfall] += 1
        return budgets

    proportions = pos_scores / pos_scores.sum()
    raw = proportions * target_K
    int_budgets = torch.floor(raw).long()
    remainders = raw - int_budgets.float()
    shortfall = target_K - int_budgets.sum().item()
    if shortfall > 0:
        _, top_idx = torch.topk(remainders, k=shortfall)
        int_budgets[top_idx] += 1
    return int_budgets


class GridPrunePruner(Pruner):
    """GridPrune: zonal text-relevance + saliency fused selection.

    Args:
        keep_ratio: float in (0, 1]
        block_size: zone partition. 2 → 12×12=144 zones of 4 tokens (paper default).
            Larger block_size = coarser zoning, more within-zone variation.
        alpha: weight on text_relevance in fused score. α=1 → pure text;
            α=0 → pure saliency; paper default 0.5 (balanced).
    """

    def __init__(self, keep_ratio: float, block_size: int = 2, alpha: float = 0.5):
        super().__init__(keep_ratio)
        self.block_size = block_size
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"GridPrune_bs{self.block_size}_a{self.alpha:.2f}_kr{self.keep_ratio:.2f}"

    @torch.no_grad()
    def select_indices(self, hidden_states, visual_mask, text_question_mask):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        keep_masks = []

        for b in range(batch_size):
            visual_feats = hidden_states[b][visual_mask[b]]       # [n_visual, hidden]
            text_feats = hidden_states[b][text_question_mask[b]]   # [n_text,   hidden]
            n_visual = visual_feats.shape[0]
            n_keep = self.n_keep(n_visual)

            if text_feats.shape[0] == 0:
                # Degenerate: no question text. Deterministic fallback.
                keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
                keep_mask[:n_keep] = True
                keep_masks.append(keep_mask)
                continue

            # --- Text-conditional relevance (mean cosine, fp32) ---
            visual_norm = F.normalize(visual_feats.float(), dim=-1)
            text_norm = F.normalize(text_feats.float(), dim=-1)
            sim_matrix = visual_norm @ text_norm.T
            text_rel = sim_matrix.mean(dim=-1)               # [n_visual]

            # --- Visual saliency (L2-norm proxy) ---
            saliency = visual_feats.float().norm(dim=-1)     # [n_visual]

            # Min-max normalize both to [0,1] for fair fusion
            def minmax(x):
                xmin, xmax = x.min(), x.max()
                if (xmax - xmin) < 1e-8:
                    return torch.zeros_like(x)
                return (x - xmin) / (xmax - xmin)

            text_rel_n = minmax(text_rel)
            saliency_n = minmax(saliency)

            # --- Fused score ---
            fused = self.alpha * text_rel_n + (1.0 - self.alpha) * saliency_n

            # --- Zone partitioning ---
            zone_idx, n_zones = _compute_zone_indices(n_visual, self.block_size, device)

            # --- Per-zone aggregate text relevance (sum drives budget) ---
            zone_scores = torch.zeros(n_zones, device=device)
            zone_scores.index_add_(0, zone_idx, text_rel_n)

            # --- Budget allocation ---
            budgets = _allocate_budget(zone_scores, n_keep, n_zones)

            # --- Local top-K within each zone ---
            keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
            for z in range(n_zones):
                B_z = budgets[z].item()
                if B_z == 0:
                    continue
                in_zone = (zone_idx == z).nonzero(as_tuple=True)[0]
                if in_zone.numel() == 0:
                    continue
                actual_k = min(B_z, in_zone.numel())
                _, top_local = torch.topk(fused[in_zone], k=actual_k, largest=True)
                keep_mask[in_zone[top_local]] = True

            keep_masks.append(keep_mask)

        return torch.stack(keep_masks)
