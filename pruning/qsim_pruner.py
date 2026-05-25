"""Question-Similarity (QSim) visual token pruning.

For each visual token, compute its mean cosine similarity to all question text
tokens. Keep the visual tokens whose representations best match the question.

Question-aware: the score depends on what the question text is. A medical
image's "spleen" region scores high when the question mentions spleen, and low
when the question asks about liver - assuming the LLM has learned visual
representations that align with medical vocabulary (HuatuoGPT-Vision did
during its medical fine-tuning).
"""
import torch
import torch.nn.functional as F
from .base import Pruner


class QSimPruner(Pruner):
    """Cosine similarity between visual tokens and question text tokens.

    Args:
        keep_ratio: float in (0, 1]
        reduction: 'mean' or 'max'. How to aggregate similarity across text
            tokens. 'mean' is stable to noise from punctuation/stopwords;
            'max' gives more weight to the single most-relevant word.
    """

    def __init__(self, keep_ratio: float, reduction: str = "mean"):
        super().__init__(keep_ratio)
        if reduction not in {"mean", "max"}:
            raise ValueError(f"reduction must be 'mean' or 'max', got {reduction}")
        self.reduction = reduction

    @property
    def name(self) -> str:
        return f"QSim_{self.reduction}_kr{self.keep_ratio:.2f}"

    @torch.no_grad()
    def select_indices(self, hidden_states, visual_mask, text_question_mask):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        keep_masks = []

        for b in range(batch_size):
            visual_feats = hidden_states[b][visual_mask[b]]          # [n_visual, hidden]
            text_feats = hidden_states[b][text_question_mask[b]]     # [n_text, hidden]
            n_visual = visual_feats.shape[0]
            n_keep = self.n_keep(n_visual)

            # Edge case: no question text identified (degenerate prompt).
            # Fall back to deterministic first-N rather than random, so the run
            # is reproducible.
            if text_feats.shape[0] == 0:
                keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
                keep_mask[:n_keep] = True
                keep_masks.append(keep_mask)
                continue

            # Cast to fp32 for cosine similarity - bf16 accumulates noticeable
            # error in dot products. Only the scoring computation runs in fp32;
            # the model itself stays bf16.
            visual_norm = F.normalize(visual_feats.float(), dim=-1)
            text_norm = F.normalize(text_feats.float(), dim=-1)

            # Similarity matrix [n_visual, n_text]
            sim_matrix = visual_norm @ text_norm.T

            # Aggregate per visual token
            if self.reduction == "mean":
                scores = sim_matrix.mean(dim=-1)  # [n_visual]
            else:  # max
                scores = sim_matrix.max(dim=-1).values

            # Top-k
            _, top_idx = torch.topk(scores, k=n_keep, largest=True, sorted=False)
            keep_mask = torch.zeros(n_visual, dtype=torch.bool, device=device)
            keep_mask[top_idx] = True
            keep_masks.append(keep_mask)

        return torch.stack(keep_masks)
