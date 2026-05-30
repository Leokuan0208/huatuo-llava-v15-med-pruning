"""ScoredHuatuoChatbot: subclass of HuatuoChatbot that exposes answer
confidence (first-token logprob distribution) and self-consistency samples,
WITHOUT modifying the upstream cli.py.

Two inference modes:
  - inference_scored(text, images): greedy decode + first-token logprob dist.
    Use for the canonical answer, confidence/entropy, and clean latency.
  - inference_sampled(text, images, k): k temperature samples for
    self-consistency vote agreement. Latency is NOT meaningful here
    (num_return_sequences inflates the patcher's per-forward timer).

The first-token distribution is captured RAW (top-k token ids + logprobs).
We do not assume the answer is a clean letter; downstream code maps the
distribution onto the option letters and flags samples where the argmax
token is not a clean A-K letter.
"""
import os
import sys

# --- make this module self-sufficient regardless of caller's cwd/path ---
# cli.py lives in <project>/HuatuoGPT-Vision/. Add it so `from cli import ...`
# works whether imported from the project root, the pruning/ dir, or a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../pruning
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                       # project root
_HUATUO_ROOT = os.path.join(_PROJECT_ROOT, "HuatuoGPT-Vision")
if _HUATUO_ROOT not in sys.path:
    sys.path.insert(0, _HUATUO_ROOT)

import torch
import torch.nn.functional as F

from cli import HuatuoChatbot


# Letters used by the eval prompt (llava_prompt uses A..K).
_OPTION_LETTERS = list("ABCDEFGHIJK")


class ScoredHuatuoChatbot(HuatuoChatbot):

    # ---- shared input prep (mirrors HuatuoChatbot.inference up to generate) ----
    def _prepare(self, text, images):
        if images is None:
            images = []
        if isinstance(images, str):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    from PIL import Image
                    Image.open(img).convert("RGB")
                valid_images.append(img)
            except Exception:
                print(f"{img} This image is wrong.")
                continue
        images = valid_images[: self.max_image_num]

        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)
        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess(conv, return_tensors="pt").unsqueeze(0).to(self.device)

        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None
        return input_ids, image_tensors

    # ---- precompute token ids for option letters, both bare and space-prefixed ----
    def _letter_token_ids(self):
        if hasattr(self, "_letter_id_cache"):
            return self._letter_id_cache
        cache = {}
        for L in _OPTION_LETTERS:
            ids = set()
            for form in (L, " " + L):
                enc = self.tokenizer(form, add_special_tokens=False).input_ids
                if len(enc) == 1:            # only single-token forms are usable
                    ids.add(enc[0])
            cache[L] = ids
        self._letter_id_cache = cache
        return cache

    # ---- GREEDY: canonical answer + first-token logprob distribution ----
    @torch.no_grad()
    def inference_scored(self, text, images=None, topk=5):
        input_ids, image_tensors = self._prepare(text, images)

        gen_kwargs = dict(self.gen_kwargs)
        gen_kwargs["do_sample"] = False           # greedy: clean argmax + clean latency
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("repetition_penalty", None)

        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )

        seqs = out.sequences                      # (1, gen_len) — only new tokens (inputs_embeds path)
        answer = self.tokenizer.decode(seqs[0], skip_special_tokens=True).strip()

        # First-token distribution
        first_logits = out.scores[0][0].float()   # (vocab,)
        first_logprobs = F.log_softmax(first_logits, dim=-1)
        top_lp, top_ids = torch.topk(first_logprobs, k=topk)
        top = [
            {"token": self.tokenizer.decode([tid]), "id": int(tid), "logprob": float(lp)}
            for lp, tid in zip(top_lp.tolist(), top_ids.tolist())
        ]

        # Map distribution onto option letters (sum bare + space-prefixed forms)
        letter_ids = self._letter_token_ids()
        option_logprob = {}
        for L, ids in letter_ids.items():
            if not ids:
                continue
            lps = [first_logprobs[i] for i in ids]
            # combine probability mass across forms, back to logprob
            option_logprob[L] = float(torch.logsumexp(torch.stack(lps), dim=0))

        argmax_id = int(top_ids[0])
        all_letter_ids = set().union(*letter_ids.values()) if letter_ids else set()
        letter_clean = argmax_id in all_letter_ids

        # entropy over the option letters only (router confidence signal)
        if option_logprob:
            ol = torch.tensor(list(option_logprob.values()))
            ol = ol - torch.logsumexp(ol, dim=0)          # renormalize over options
            entropy = float(-(ol.exp() * ol).sum())
        else:
            entropy = None

        return {
            "answer": answer,
            "first_token_topk": top,
            "option_logprob": option_logprob,
            "option_entropy": entropy,
            "letter_clean": letter_clean,
        }

    # ---- SAMPLING: k samples for self-consistency ----
    @torch.no_grad()
    def inference_sampled(self, text, images=None, k=5):
        input_ids, image_tensors = self._prepare(text, images)

        gen_kwargs = dict(self.gen_kwargs)
        gen_kwargs["do_sample"] = True
        gen_kwargs.setdefault("temperature", 0.2)
        gen_kwargs["num_return_sequences"] = k

        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                **gen_kwargs,
            )

        samples = [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in out]
        return {"samples": samples}
