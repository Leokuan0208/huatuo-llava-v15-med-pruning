"""Scored eval driver: greedy+logprobs OR k-sample self-consistency,
on top of the pruning patcher. Mirrors patch_and_eval.py.

Two modes (one per invocation):
  --mode scored   greedy decode; records canonical answer + first-token
                  logprob distribution + option entropy. Latency tracked.
  --mode sampled  k temperature samples (T=0.7) for self-consistency.
                  Latency NOT tracked (num_return_sequences inflates timers).

Usage (single run):
    cd ~/huatuo-llava-v15-med-pruning
    torchrun --nproc_per_node=1 scripts/scored_sweep.py \\
        --mode scored --pruner random --keep_ratio 0.5 \\
        --model_path /data/dan/weights/HuatuoGPT-Vision-7B \\
        --data_path  /data/dan/dataset/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json \\
        --output_dir /home/jamesyang/huatuo-llava-v15-med-pruning/results/scored_sweep
"""
import os
import sys
import argparse
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
HUATUO_ROOT = os.path.join(PROJECT_ROOT, "HuatuoGPT-Vision")
sys.path.insert(0, HUATUO_ROOT)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import set_seed

from pruning import RandomPruner, QSimPruner, patcher
from pruning.gridprune_pruner import GridPrunePruner
from pruning.fasp_gridprune_pruner import FASPGridPrunePruner
from pruning.latency import LatencyTracker
from pruning.scored_chatbot import ScoredHuatuoChatbot
from pruning.nested_random_pruner import NestedRandomPruner

from scorer import score_mix_llava
from eval import TestDataset

from collections import Counter

SAMPLE_TEMPERATURE = 0.7   # locked from smoke-test
SAMPLE_K = 5


def build_pruner(args):
    if args.pruner == "random":
        return RandomPruner(args.keep_ratio, seed=args.seed)
    elif args.pruner == "nested_random":
        return NestedRandomPruner(args.keep_ratio, seed=args.seed)
    elif args.pruner == "qsim":
        return QSimPruner(args.keep_ratio, reduction=args.qsim_reduction)
    elif args.pruner == "gridprune":
        return GridPrunePruner(keep_ratio=args.keep_ratio,
                               block_size=args.block_size, alpha=args.alpha)
    elif args.pruner == "fasp_gridprune":
        return FASPGridPrunePruner(keep_ratio=args.keep_ratio,
                                   block_size=args.block_size, alpha=args.alpha,
                                   bg_fraction=args.bg_fraction)
    elif args.pruner == "none":
        return None
    raise ValueError(f"Unknown pruner: {args.pruner}")


def first_letter(s):
    s = s.strip()
    return s[:1].upper() if s else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scored", "sampled"], required=True)
    parser.add_argument("--pruner",
                        choices=["random", "qsim", "gridprune", "fasp_gridprune", "none"],
                        required=True)
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--qsim_reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--bg_fraction", type=float, default=0.30)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    accelerator.print(f"args:\n{args}")

    bot = ScoredHuatuoChatbot(args.model_path)
    accelerator.print("load_finish")
    bot.gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.mode == "sampled":
        bot.gen_kwargs["temperature"] = SAMPLE_TEMPERATURE

    pruner = build_pruner(args)
    if pruner is not None:
        patcher.patch_model(bot.model)
        accelerator.print(f"patcher applied: pruner={pruner.name}")
    else:
        accelerator.print("pruner=none: no patcher applied (baseline mode)")

    # Latency only meaningful in scored (single-sequence) mode.
    latency_tracker = LatencyTracker() if args.mode == "scored" else None
    patcher.configure(pruner=pruner, latency_tracker=latency_tracker)

    os.makedirs(args.output_dir, exist_ok=True)
    pruner_tag = pruner.name if pruner else "baseline"
    if args.tag:
        pruner_tag = f"{pruner_tag}_{args.tag}"
    base = f"{os.path.basename(args.model_path)}__{pruner_tag}__{args.mode}"
    out_preds = os.path.join(args.output_dir, f"{base}__predictions.json")
    out_scores = os.path.join(args.output_dir, f"{base}__scores.json")
    out_latency_summary = os.path.join(args.output_dir, f"{base}__latency_summary.json")
    out_latency_jsonl = os.path.join(args.output_dir, f"{base}__latency.jsonl")
    checkpoint_path = os.path.join(args.output_dir, f"{base}__checkpoint_partial.json")

    dataset = TestDataset(args, args.data_path)
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                drop_last=False, collate_fn=dataset.collate_fn)
    val_dataloader = accelerator.prepare(val_dataloader)
    accelerator.wait_for_everyone()

    cache_data = []
    with torch.no_grad():
        it = (tqdm(val_dataloader, total=len(val_dataloader))
              if accelerator.is_main_process else val_dataloader)
        for i, batch in enumerate(it):
            for da, query, image in zip(batch["data"], batch["query"], batch["image"]):
                image_paths = [os.path.join(os.path.dirname(args.data_path), x) for x in image]
                for img in image_paths:
                    assert os.path.exists(img), f"{img} not exists"

                if args.mode == "scored":
                    sample_id = str(da.get("test_id", i))
                    with latency_tracker.time_sample(sample_id) as timer:
                        r = bot.inference_scored(query, image_paths)
                    try:
                        timer.set_n_generated(len(bot.tokenizer.encode(r["answer"])))
                    except Exception:
                        pass
                    da["model_output"] = r["answer"]
                    da["first_token_topk"] = r["first_token_topk"]
                    da["option_logprob"] = r["option_logprob"]
                    da["option_entropy"] = r["option_entropy"]
                    da["letter_clean"] = r["letter_clean"]
                else:  # sampled
                    r = bot.inference_sampled(query, image_paths, k=SAMPLE_K)
                    letters = [first_letter(s) for s in r["samples"]]
                    vote = Counter(letters).most_common(1)[0] if letters else ("", 0)
                    da["samples"] = r["samples"]
                    da["sample_letters"] = letters
                    da["vote_majority"] = vote[0]
                    da["vote_agreement"] = vote[1] / max(len(letters), 1)
                    # keep a canonical-ish answer field for compatibility
                    da["model_output"] = vote[0]

                cache_data.append(da)

            if (i + 1) % 500 == 0:
                with open(checkpoint_path, "w") as fw:
                    json.dump(cache_data, fw, ensure_ascii=False, indent=2)
                if latency_tracker is not None:
                    latency_tracker.save_jsonl(out_latency_jsonl)

        with open(checkpoint_path, "w") as fw:
            json.dump(cache_data, fw, ensure_ascii=False, indent=2)
        if latency_tracker is not None:
            latency_tracker.save_jsonl(out_latency_jsonl)
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        all_data = [None] * dist.get_world_size()
        dist.all_gather_object(all_data, cache_data)
        all_data = [item for sublist in all_data for item in sublist]

        if accelerator.is_main_process:
            with open(out_preds, "w") as fw:
                json.dump(all_data, fw, ensure_ascii=False, indent=2)
            print(f"predictions: {out_preds}  (n={len(all_data)})")

            val_res = score_mix_llava(all_data)
            print(json.dumps(val_res, ensure_ascii=False, indent=2))
            with open(out_scores, "w") as fw:
                json.dump(val_res, fw, ensure_ascii=False, indent=2)
            print(f"scores: {out_scores}")

            if latency_tracker is not None:
                lat = latency_tracker.summary()
                lat["config"] = {
                    "mode": args.mode, "pruner": args.pruner,
                    "keep_ratio": args.keep_ratio, "block_size": args.block_size,
                    "alpha": args.alpha, "bg_fraction": args.bg_fraction,
                }
                with open(out_latency_summary, "w") as fw:
                    json.dump(lat, fw, indent=2)
                print(f"latency summary: {out_latency_summary}")


if __name__ == "__main__":
    main()
