"""Patcher v2: prune visual tokens BEFORE the LLM trunk runs.

[Original docstring preserved — see git log 85cb249 for the architectural
rationale and the v1 → v2 transition story.]

ADDITIONS (2026-05-27):
  - prune_time bracket around pruner.select_indices, written to LatencyTracker
  - LM body forward wrap (model.model.forward) to time prefill vs decode
    separately. Detection: first call per generate() has empty/None
    past_key_values (prefill); subsequent calls have populated cache (decode).
"""
import functools
import time
from typing import Optional
import torch


IMAGE_TOKEN_INDEX = -200
N_VISUAL_TOKENS = 576


_STATE = {
    "pruner": None,
    "latency_tracker": None,
}

_FIRST_PRUNE_LOGGED = False


def configure(pruner=None, latency_tracker=None):
    _STATE["pruner"] = pruner
    _STATE["latency_tracker"] = latency_tracker


def _find_visual_start(input_ids):
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


def _is_prefill_step(past_key_values):
    """Detect prefill (first forward in generate) vs decode (subsequent forwards).

    Handles both the new transformers Cache API and the legacy tuple format.
    Conservative default: treat ambiguous cases as prefill.
    """
    if past_key_values is None:
        return True
    if hasattr(past_key_values, 'get_seq_length'):
        try:
            return past_key_values.get_seq_length() == 0
        except Exception:
            return False
    if isinstance(past_key_values, (list, tuple)):
        if len(past_key_values) == 0:
            return True
        first = past_key_values[0]
        if first is None:
            return True
        if isinstance(first, (list, tuple)) and len(first) > 0 and first[0] is not None:
            try:
                return first[0].size(-2) == 0
            except Exception:
                return False
    return True


def patch_model(model):
    """Patch prepare_inputs_labels_for_multimodal_new AND wrap the LM body
    forward for phase-resolved timing."""

    # === Part 1: multimodal-prep wrapper (existing logic + prune timer) ===
    original_prep = model.prepare_inputs_labels_for_multimodal_new

    @functools.wraps(original_prep)
    def wrapped_prep(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs):
        global _FIRST_PRUNE_LOGGED

        visual_span = _find_visual_start(input_ids)
        result = original_prep(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs)
        ret_input_ids, ret_position_ids, ret_attention_mask, ret_past_kv, ret_inputs_embeds, ret_labels = result

        pruner = _STATE["pruner"]
        if pruner is None or ret_inputs_embeds is None or visual_span is None:
            return result

        batch_size, seq_length, _ = ret_inputs_embeds.shape
        v_start, v_end = visual_span
        v_end = min(v_end, seq_length)
        n_visual = v_end - v_start
        device = ret_inputs_embeds.device

        visual_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        question_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        visual_mask[:, v_start:v_end] = True
        question_mask[:, v_end:seq_length] = True

        # === NEW: time pruning compute ===
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_prune_start = time.perf_counter()

        keep_visual = pruner.select_indices(ret_inputs_embeds, visual_mask, question_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prune_elapsed = time.perf_counter() - t_prune_start

        if _STATE["latency_tracker"] is not None:
            _STATE["latency_tracker"].add_prune_time(prune_elapsed)
        # === END NEW ===

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

    model.prepare_inputs_labels_for_multimodal_new = wrapped_prep

    # === Part 2: LM body forward wrap for prefill/decode timing ===
    if not hasattr(model, 'model'):
        print("[PATCHER v2] warning: model.model not found; "
              "prefill/decode timing disabled", flush=True)
        return

    lm_body = model.model
    original_lm_forward = lm_body.forward  # bound method

    @functools.wraps(original_lm_forward)
    def wrapped_lm_forward(*args, **kwargs):
        # past_key_values is the 4th positional arg in Qwen2Model.forward,
        # or a kwarg. Try kwargs first, fall back to args.
        past_kv = kwargs.get('past_key_values', None)
        if past_kv is None and len(args) >= 4:
            past_kv = args[3]
        is_prefill = _is_prefill_step(past_kv)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        result = original_lm_forward(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        if _STATE["latency_tracker"] is not None:
            if is_prefill:
                _STATE["latency_tracker"].add_prefill_time(elapsed)
            else:
                _STATE["latency_tracker"].add_decode_time(elapsed)

        return result

    lm_body.forward = wrapped_lm_forward