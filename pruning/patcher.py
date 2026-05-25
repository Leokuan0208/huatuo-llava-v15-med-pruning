"""Monkey-patches HuatuoGPT-Vision's model to prune visual tokens after LLM layer 0.

Architecture:
  - patch_model(): entry point. Patches two things:
      1. llava_arch.prepare_inputs_labels_for_multimodal -> records visual span
         in module-level state. Doesn't change return value.
      2. Qwen2Model.forward -> our replacement that runs layer 0, prunes, then
         runs layers 1-31 with sliced mask and renumbered position_ids.

  - The replacement Qwen2Model.forward is a near-verbatim copy of the
    transformers 4.37.2 source. We only change two things:
      a) After the layer-0 call, we prune the sequence (only on prefill,
         identified by seq_length > 1).
      b) The 4D attention mask gets sliced to the new sequence length, and
         layer 0's KV cache also gets sliced (otherwise decode steps mismatch).

Known constraints (deliberately):
  - batch_size MUST equal 1 (HuatuoGPT's eval.py is hardcoded to that).
    The mask-slicing logic assumes single-sample batches.
  - One image per sample (HuatuoGPT's default).
  - We skip pruning during decode steps (seq_length == 1) since there are no
    visual tokens to prune mid-generation.

This code targets transformers==4.37.2 specifically. If the transformers
version in the container differs, the Qwen2Model.forward signature may
change and this will need to be re-verified against that version's source.
"""
import functools
from typing import Optional, List, Tuple, Union
import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast


# IMAGE_TOKEN_INDEX is -200 per LLaVA-v1.5 convention.
# Number of visual tokens HuatuoGPT-Vision produces per image: 576 (CLIP ViT-L/14 @ 336^2, 24x24 patches).
IMAGE_TOKEN_INDEX = -200
N_VISUAL_TOKENS = 576


# Module-level state. The prepare_inputs_labels_for_multimodal patch writes
# here; the Qwen2Model.forward patch reads from here. We never run more than
# one model in one process so this is safe.
_STATE = {
    "pruner": None,
    "latency_tracker": None,
    "visual_span": None,         # (start, end) of visual tokens in current sample
    "question_span": None,       # (start, end) of question text tokens
}


def configure(pruner=None, latency_tracker=None):
    """Set the active pruner and (optional) latency tracker for the patched model."""
    _STATE["pruner"] = pruner
    _STATE["latency_tracker"] = latency_tracker


# --------------------------------------------------------------------------- #
# Patch 1: prepare_inputs_labels_for_multimodal
# --------------------------------------------------------------------------- #

def _find_visual_span_in_input_ids(input_ids_1d):
    """Locate the IMAGE_TOKEN_INDEX in raw input_ids. Returns (start, end) in
    the post-embedding sequence, since one image token expands to N_VISUAL_TOKENS."""
    ids = input_ids_1d.tolist()
    try:
        img_pos = ids.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        return None
    return (img_pos, img_pos + N_VISUAL_TOKENS)


def _patch_prepare_inputs(model):
    """Patch the llava_arch mixin to record visual/question spans."""
    original = model.prepare_inputs_labels_for_multimodal

    @functools.wraps(original)
    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs):
        # Record where the visual tokens will land before calling original
        # (since the original consumes input_ids and we need the layout info).
        # We assume batch_size=1.
        if input_ids is not None and input_ids.shape[0] >= 1:
            visual_span = _find_visual_span_in_input_ids(input_ids[0])
        else:
            visual_span = None

        # The question text in the prompt sits between the image token and the
        # end-of-input. After embedding, this corresponds to: from end-of-visual
        # to end-of-sequence. We'll resolve the exact end from the post-embedding
        # length inside the forward patch (since we don't know the final length
        # until after the original runs).
        _STATE["visual_span"] = visual_span
        # We'll set question_span inside forward, once we know final seq_length.

        return original(input_ids, position_ids, attention_mask, past_key_values, labels, images, *args, **kwargs)

    model.prepare_inputs_labels_for_multimodal = wrapped


# --------------------------------------------------------------------------- #
# Patch 2: Qwen2Model.forward - copy of transformers 4.37.2, with pruning
# inserted after layer 0.
# --------------------------------------------------------------------------- #

def _slice_4d_mask(mask, keep_indices_1d):
    """Slice a 4D causal attention mask along q and kv dimensions.

    mask shape: [batch, 1, q_len, kv_len]. Both q_len and kv_len equal the
    original prefill seq_length. We slice both to the new pruned length so
    causality is preserved for the kept positions.
    """
    if mask is None:
        return None
    out = mask.index_select(2, keep_indices_1d).index_select(3, keep_indices_1d)
    return out


def _slice_2d_mask(mask, keep_indices_1d):
    """Slice a 2D attention mask (used by flash-attention path). Shape [batch, seq]."""
    if mask is None:
        return None
    return mask.index_select(1, keep_indices_1d)


def _build_pruned_forward():
    """Returns a forward function that closes over the patcher logic.

    The body is structurally identical to transformers 4.37.2's
    Qwen2Model.forward, with the prune-after-layer-0 block inserted.
    """

    def pruned_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError("padding_right not supported here")

        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        pruner = _STATE["pruner"]
        latency_tracker = _STATE["latency_tracker"]
        do_prune = (
            pruner is not None
            and seq_length > 1
            and _STATE["visual_span"] is not None
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, attention_mask, position_ids, past_key_values,
                    output_attentions, use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # === Pruning point: after layer 0, prefill only ===
            if layer_idx == 0 and do_prune:
                hidden_states, attention_mask, position_ids = _apply_pruning_step(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_length=seq_length,
                    attn_impl=self._attn_implementation,
                    pruner=pruner,
                    latency_tracker=latency_tracker,
                    past_key_values=past_key_values if use_cache else None,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return pruned_forward


def _apply_pruning_step(hidden_states, attention_mask, position_ids, seq_length, attn_impl, pruner, latency_tracker, past_key_values=None):
    """Run the pruner and slice the hidden states, mask, position_ids, and layer-0 KV cache.

    The KV cache slicing is critical: after layer 0 ran on the full sequence, its cache
    has all original tokens. If we don't slice it to match the post-pruning sequence,
    subsequent decode steps will mismatch position_ids vs cache length and fail.
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device

    visual_span = _STATE["visual_span"]
    v_start, v_end = visual_span
    v_end = min(v_end, seq_length)
    q_start, q_end = v_end, seq_length

    visual_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    question_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    visual_mask[:, v_start:v_end] = True
    question_mask[:, q_start:q_end] = True

    keep_visual = pruner.select_indices(hidden_states, visual_mask, question_mask)

    n_visual = v_end - v_start
    full_keep = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    full_keep[:, v_start:v_end] = keep_visual[:, :n_visual]

    if latency_tracker is not None:
        n_keep = int(keep_visual[0].sum().item())
        latency_tracker.set_visual_counts(pre=n_visual, post=n_keep)

    keep_indices = full_keep[0].nonzero(as_tuple=True)[0]
    new_seq_len = keep_indices.shape[0]

    new_hidden = hidden_states.index_select(1, keep_indices)

    if attn_impl in {"sdpa", "eager"}:
        new_mask = _slice_4d_mask(attention_mask, keep_indices)
    else:  # flash_attention_2
        new_mask = _slice_2d_mask(attention_mask, keep_indices)

    new_position_ids = torch.arange(new_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # CRITICAL: slice layer 0's KV cache so decode steps work.
    if past_key_values is not None:
        _slice_layer0_cache(past_key_values, keep_indices)

    return new_hidden, new_mask, new_position_ids


def _slice_layer0_cache(past_key_values, keep_indices):
    """Slice layer 0's K and V cache along the seq dimension."""
    if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
        past_key_values.key_cache[0] = past_key_values.key_cache[0].index_select(2, keep_indices)
        past_key_values.value_cache[0] = past_key_values.value_cache[0].index_select(2, keep_indices)
        if hasattr(past_key_values, "_seen_tokens"):
            past_key_values._seen_tokens = past_key_values.key_cache[0].shape[2]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def patch_model(model):
    """Apply both patches to a loaded HuatuoChatbot's underlying model.

    Args:
        model: the HuggingFace model that has .prepare_inputs_labels_for_multimodal
               and .model (the Qwen2Model). For HuatuoGPT-Vision-7B this is the
               LlavaQwenForCausalLM-equivalent module.
    """
    _patch_prepare_inputs(model)
    qwen_model = model.model
    new_forward = _build_pruned_forward()
    qwen_model.forward = new_forward.__get__(qwen_model, qwen_model.__class__)
