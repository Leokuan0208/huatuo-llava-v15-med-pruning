"""Patcher v2: prune visual tokens BEFORE the LLM trunk runs.

Architectural rationale (see bugs page for the v1 post-mortem):
  v1 ran pruning after LLM layer 0 inside a Qwen2Model.forward override.
  This required reconciling pruned-frame state (our KV cache, hidden states)
  with HF generate's unpruned-frame state (attention_mask, position_ids) on
  every decode step. The reconciliation produced a chain of bugs: KV cache
  length drift, 4D attention mask shape mismatch, and rotary-position-
  embedding index-out-of-bounds. Each fix exposed the next.

  v2 prunes at the multimodal-prep hook, before any LLM layer runs:

      LlavaQwen2.generate
        -> prepare_inputs_labels_for_multimodal_new   <-- we patch here
           [original: splice 576 visual embeddings into text embedding seq]
           [our wrapper: score visual tokens, slice inputs_embeds]
        -> super().generate(inputs_embeds=pruned, attention_mask=None, ...)
           [HF generate builds attention_mask/position_ids from PRUNED length]

  HF generate never sees the unpruned sequence. No drift. No slicing inside
  the LLM trunk. No mid-trunk patching needed at all.

Approach matches the integration style of pre-LLM token-selection methods
(VisionZip, SparseVLM v1, FastV's k=0 mode).

Scoring is done on the post-projector, pre-LLM-trunk embeddings. Qwen2 uses
causal attention, so visual tokens (positions 5..580) cannot attend to
question tokens (581..631) at any layer including layer 0; scoring on raw
embeddings vs layer-0 outputs is approximately equivalent for our visual
tokens. Documented as a known approximation; the v1 vs v2 ablation is a
clean way to measure it later.

Constraints:
  - batch_size == 1, one image per sample (matches HuatuoGPT eval).
"""
import functools
from typing import Optional
import torch


IMAGE_TOKEN_INDEX = -200
N_VISUAL_TOKENS = 576


_STATE = {
    "pruner": None,
    "latency_tracker": None,
}

# Once-per-run sentinel. One line at startup confirms pruning fired, then
# silence. Per-sample verification lives in the LatencyTracker JSON.
_FIRST_PRUNE_LOGGED = False


def configure(pruner=None, latency_tracker=None):
    """Set the active pruner and (optional) latency tracker."""
    _STATE["pruner"] = pruner
    _STATE["latency_tracker"] = latency_tracker


def _find_visual_start(input_ids):
    """Find the (start, end) span of visual tokens in the POST-splice
    sequence, by locating IMAGE_TOKEN_INDEX in the pre-splice input_ids.

    HuatuoGPT-Vision's prep accepts input_ids as either a [B, L] tensor or
    a list of [L] tensors. We handle both. Returns None if no image token
    is found (decode step, or text-only call).
    """
    if input_ids is None:
        return None

    if isinstance(input_ids, torch.Tensor):
        ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    elif isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], torch.Tensor):
        ids = input_ids[0]
    else:
        return None

    matches = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    start = matches[0].item()
    return (start, start + N_VISUAL_TOKENS)


def patch_model(model):
    """Patch HuatuoGPT-Vision's prepare_inputs_labels_for_multimodal_new to
    prune visual tokens after splicing but before the LLM trunk consumes
    the embeddings."""
    original = model.prepare_inputs_labels_for_multimodal_new

    @functools.wraps(original)
    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs):
        global _FIRST_PRUNE_LOGGED

        # Read visual span from input_ids BEFORE original consumes them.
        visual_span = _find_visual_start(input_ids)

        # Run the original splice.
        result = original(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs)
        ret_input_ids, ret_position_ids, ret_attention_mask, ret_past_kv, ret_inputs_embeds, ret_labels = result

        pruner = _STATE["pruner"]

        # Pass through when: no pruner configured, decode step (original
        # returns inputs_embeds=None), or call had no image (no -200 in ids).
        if pruner is None or ret_inputs_embeds is None or visual_span is None:
            return result

        # Score and slice on the spliced embeddings.
        batch_size, seq_length, _ = ret_inputs_embeds.shape
        v_start, v_end = visual_span
        v_end = min(v_end, seq_length)
        n_visual = v_end - v_start
        device = ret_inputs_embeds.device

        visual_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        question_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        visual_mask[:, v_start:v_end] = True
        question_mask[:, v_end:seq_length] = True

        keep_visual = pruner.select_indices(ret_inputs_embeds, visual_mask, question_mask)

        full_keep = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
        full_keep[:, v_start:v_end] = keep_visual[:, :n_visual]

        if _STATE["latency_tracker"] is not None:
            n_keep = int(keep_visual[0, :n_visual].sum().item())
            _STATE["latency_tracker"].set_visual_counts(pre=n_visual, post=n_keep)

        keep_indices = full_keep[0].nonzero(as_tuple=True)[0]
        new_seq_len = keep_indices.shape[0]

        if not _FIRST_PRUNE_LOGGED:
            print(f"[PATCHER v2] first prune confirmed: seq {seq_length} -> {new_seq_len}, "
                  f"visual {n_visual} -> {int(keep_visual[0, :n_visual].sum().item())} "
                  f"(logged once per run)", flush=True)
            _FIRST_PRUNE_LOGGED = True

        new_inputs_embeds = ret_inputs_embeds.index_select(1, keep_indices)

        # At prefill from LlavaQwen2.generate, attention_mask/position_ids/labels
        # all come back as None (HuatuoGPT-Vision's prep sets them None when its
        # _attention_mask/_position_ids/_labels inputs were None). HF generate
        # then constructs them from the (now pruned) sequence length. We still
        # guard for the general case in which the prep may have returned values.
        new_attention_mask = (
            ret_attention_mask.index_select(1, keep_indices)
            if ret_attention_mask is not None and ret_attention_mask.dim() == 2
            else ret_attention_mask
        )
        new_position_ids = (
            ret_position_ids.index_select(1, keep_indices)
            if ret_position_ids is not None and ret_position_ids.dim() == 2
            else ret_position_ids
        )
        new_labels = (
            ret_labels.index_select(1, keep_indices)
            if ret_labels is not None and ret_labels.dim() == 2
            else ret_labels
        )

        return (ret_input_ids, new_position_ids, new_attention_mask,
                ret_past_kv, new_inputs_embeds, new_labels)

    model.prepare_inputs_labels_for_multimodal_new = wrapped
