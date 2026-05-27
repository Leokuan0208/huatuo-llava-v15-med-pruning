"""FASP+GridPrune: anatomy-aware foreground filter + zonal selection.

Stage 1 (FASP — Foreground-Anatomy Stratified Pruning):
  Detect background tokens by L2-norm percentile thresholding. Tokens whose
  L2 norm falls in the bottom bg_fraction get labeled "background" and are
  excluded from selection (with a fallback path for when foreground is smaller
  than n_keep — see below).

Stage 2 (GridPrune-style within-foreground):
  Score remaining tokens by α·text_relevance + (1-α)·saliency, partition
  into zones, allocate per-zone budget by joint signal of text relevance
  × zone-level anatomy fraction (so zones with both high relevance AND
  high anatomy density get the most budget). Local top-K within each zone,
  restricted to anatomy tokens.

Novelty positioning: domain-agnostic methods (GridPrune, DivPrune, etc.)
treat all visual tokens as equally-eligible candidates. Medical images
have non-uniform information density — CT/X-ray have 30-50% diagnostic
background; microscopy has dense informative content throughout. FASP
exploits this with a cheap norm-based foreground filter; the within-zone
selection then concentrates the kr budget on diagnostic regions.

kr semantics: keep_ratio is fraction of ORIGINAL 576 tokens kept, not
fraction of foreground. This keeps comparisons apples-to-apples with
Random / QSim / GridPrune at the same kr value.

Edge case: if FASP labels MORE than (1-kr) of tokens as background, we
have fewer foreground candidates than n_keep. Fallback: keep all anatomy
tokens, then fill the remainder with highest-norm background tokens. This
ensures the LLM always receives exactly n_keep tokens, regardless of
image content. In practice with bg_fraction=0.30 and kr ≥ 0.10, the
foreground (~404 tokens) is always larger than n_keep (≥58), so this
fallback is rarely triggered.
"""
import math
import torch
import torch.nn.functional as F
from .base import Pruner
from .gridprune_pruner import _compute_zone_indices


def _allocate_budget_capped(zone_scores, target_K, caps):
    """Largest-remainder allocation with per-zone capacity caps.

    caps[z] = maximum tokens that can be drawn from zone z (i.e., anatomy
    token count in zone z). Iteratively redistributes any shortfall created
    by capping until target_K is met or all capacity is exhausted.
    """
    n_zones = zone_scores.shape[0]
    device = zone_scores.device
    caps_long = caps.long()

    pos_scores = torch.clamp(zone_scores, min=0.0) * (caps > 0).float()

    if pos_scores.sum() <= 1e-8:
        # Zero relevance everywhere with capacity — round-robin uniform
        budgets = torch.zeros(n_zones, dtype=torch.long, device=device)
        non_empty = (caps > 0).nonzero(as_tuple=True)[0]
        if non_empty.numel() == 0:
            return budgets
        remaining = target_K
        i = 0
        safety = target_K * 10 + n_zones
        while remaining > 0 and (caps_long > budgets).any() and safety > 0:
            z = non_empty[i % non_empty.numel()].item()
            if budgets[z] < caps_long[z]:
                budgets[z] += 1
                remaining -= 1
            i += 1
            safety -= 1
        return budgets

    proportions = pos_scores / pos_scores.sum()
    raw = proportions * target_K
    int_budgets = torch.minimum(torch.floor(raw).long(), caps_long)

    shortfall = target_K - int_budgets.sum().item()
    max_iter = max(shortfall * 2, 1)
    for _ in range(max_iter):
        if shortfall <= 0:
            break
        slack = caps_long - int_budgets
        if slack.sum() == 0:
            break
        eligible = slack > 0
        remainders = (raw - int_budgets.float()) * eligible.float()
        if remainders.max() <= 0:
            # Eligible zones already at/above proportional share; give to most-slack zone
            slack_score = slack.float() * eligible.float()
            best = slack_score.argmax()
        else:
            best = remainders.argmax()
        int_budgets[best] += 1
        shortfall -= 1

    return int_budgets


class FASPGridPrunePruner(Pruner):
    """Foreground-Anatomy Stratified Pruning + GridPrune-style selection.

    Args:
        keep_ratio: float in (0, 1]
        block_size: zone partition (default 2, matching GridPrune)
        alpha: text_relevance weight in fused score (default 0.5)
        bg_fraction: fraction of tokens labeled background by L2-norm percentile
            (default 0.30 — ~40% diagnostic capacity in typical CT/X-ray).
    """

    def __init__(self, keep_ratio, block_size=2, alpha=0.5, bg_fraction=0.30):
        super().__init__(keep_ratio)
        self.block_size = block_size
        self.alpha = alpha
        self.bg_fraction = bg_fraction

    @property
    def name(self) -> str:
        return (f"FASPGrid_bs{self.block_size}_bg{self.bg_fraction:.2f}"
                f"_a{self.alpha:.2f}_kr{self.keep_ratio:.2f}")

    @torch.no_grad()
    def select_indices(self, hidden_states, visual_mask, text_question_mask):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        keep_masks = []

        for b in range(batch_size):
            visual_feats = hidden_states[b][visual_mask[b]]
            text_feats = hidden_states[b][text_question_mask[b]]
            n_visual = visual_feats.shape[0]
            n_keep = self.n_keep(n_visual)

            saliency = visual_feats.float().norm(dim=-1)  # [n_visual]

            if text_feats.shape[0] == 0:
                keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
                keep_mask[:n_keep] = True
                keep_masks.append(keep_mask)
                continue

            # --- Stage 1: FASP foreground/background labels ---
            n_background = int(self.bg_fraction * n_visual)
            if n_background > 0:
                threshold = torch.kthvalue(saliency, n_background).values
                anatomy_mask = saliency > threshold
            else:
                anatomy_mask = torch.ones(n_visual, dtype=torch.bool, device=device)
            n_anatomy = int(anatomy_mask.sum().item())

            # Edge case: foreground too small for the requested keep count.
            # Keep all anatomy + fill from highest-norm background tokens.
            if n_anatomy <= n_keep:
                keep_mask = anatomy_mask.clone()
                shortfall = n_keep - n_anatomy
                if shortfall > 0:
                    bg_indices = (~anatomy_mask).nonzero(as_tuple=True)[0]
                    if bg_indices.numel() > 0:
                        fill_k = min(shortfall, bg_indices.numel())
                        _, top_bg = torch.topk(saliency[bg_indices], k=fill_k, largest=True)
                        keep_mask[bg_indices[top_bg]] = True
                keep_masks.append(keep_mask)
                continue

            # --- Stage 2a: text-conditional relevance ---
            visual_norm_v = F.normalize(visual_feats.float(), dim=-1)
            text_norm_v = F.normalize(text_feats.float(), dim=-1)
            sim_matrix = visual_norm_v @ text_norm_v.T
            text_rel = sim_matrix.mean(dim=-1)

            def minmax(x):
                xmin, xmax = x.min(), x.max()
                if (xmax - xmin) < 1e-8:
                    return torch.zeros_like(x)
                return (x - xmin) / (xmax - xmin)

            text_rel_n = minmax(text_rel)
            saliency_n = minmax(saliency)
            fused = self.alpha * text_rel_n + (1.0 - self.alpha) * saliency_n

            # --- Stage 2b: zoning + per-zone stats ---
            zone_idx, n_zones = _compute_zone_indices(n_visual, self.block_size, device)

            zone_anatomy_count = torch.zeros(n_zones, device=device)
            zone_anatomy_count.index_add_(0, zone_idx, anatomy_mask.float())

            anatomy_text_rel = text_rel_n * anatomy_mask.float()
            zone_text_rel = torch.zeros(n_zones, device=device)
            zone_text_rel.index_add_(0, zone_idx, anatomy_text_rel)

            zone_size = self.block_size * self.block_size
            zone_anatomy_fraction = zone_anatomy_count / zone_size
            zone_joint = zone_text_rel * zone_anatomy_fraction  # high only when both are high

            # --- Stage 2c: capped budget allocation ---
            budgets = _allocate_budget_capped(zone_joint, n_keep, zone_anatomy_count)

            # --- Stage 2d: local top-K within each zone, anatomy only ---
            keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
            for z in range(n_zones):
                B_z = budgets[z].item()
                if B_z == 0:
                    continue
                in_zone_anatomy = ((zone_idx == z) & anatomy_mask).nonzero(as_tuple=True)[0]
                if in_zone_anatomy.numel() == 0:
                    continue
                actual_k = min(B_z, in_zone_anatomy.numel())
                _, top_local = torch.topk(fused[in_zone_anatomy], k=actual_k, largest=True)
                keep_mask[in_zone_anatomy[top_local]] = True

            keep_masks.append(keep_mask)

        return torch.stack(keep_masks)
