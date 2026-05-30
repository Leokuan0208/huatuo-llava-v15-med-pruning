"""
Probe: does answer-stability under visual-token pruning predict correctness?

Joins Random-pruning predictions across 5 budgets (kr=1.0 baseline + 0.75/0.5/0.25/0.1)
by the composite key (dataset, subset, test_id), then relates per-sample answer
stability to baseline correctness. Read-only; no GPU.
"""
import json, os
from collections import defaultdict

# ---- file paths: baseline lives under a NON-STANDARD name in archive/ ----
BASE = "results/v2_pre_llm"
ARCH = "results/archive/2026-05-25_baseline_paper_repro"
FILES = {
    "1.00": f"{ARCH}/HuatuoGPT-Vision-7B_medical_multimodel_evaluation_data.json",
    "0.75": f"{BASE}/2026-05-28_random_kr0.75/HuatuoGPT-Vision-7B__RandomPruner_kr0.75__predictions.json",
    "0.50": f"{BASE}/2026-05-28_random_kr0.5/HuatuoGPT-Vision-7B__RandomPruner_kr0.50__predictions.json",
    "0.25": f"{BASE}/2026-05-28_random_kr0.25/HuatuoGPT-Vision-7B__RandomPruner_kr0.25__predictions.json",
    "0.10": f"{BASE}/2026-05-28_random_kr0.1/HuatuoGPT-Vision-7B__RandomPruner_kr0.10__predictions.json",
}
LADDER = ["1.00", "0.75", "0.50", "0.25", "0.10"]  # high -> low evidence

def key(r):
    return (r["dataset"], r.get("subset"), r["test_id"])

# letter index -> option text, so we can score model_output ('C') against answer (text)
def is_correct(rec):
    out = (rec.get("model_output") or "").strip()
    opts = rec.get("options") or []
    gold = (rec.get("answer") or "").strip()
    if out[:1].upper() in "ABCDEFGH" and opts:
        idx = ord(out[:1].upper()) - ord("A")
        if 0 <= idx < len(opts):
            return opts[idx].strip() == gold
    return out == gold  # fallback: direct string match

# ---- load all five budgets keyed by composite id ----
data = {}
for kr, path in FILES.items():
    recs = json.load(open(path))
    data[kr] = {key(r): r for r in recs}
    print(f"loaded kr={kr}: {len(recs)} records")

common = set.intersection(*[set(d.keys()) for d in data.values()])
print(f"\ncommon samples across all 5 budgets: {len(common)}\n")

# ---- per-sample: baseline correctness, flip count, stable-but-wrong flag ----
n_flips_hist = defaultdict(int)
xtab = defaultdict(lambda: defaultdict(int))   # (baseline_correct) -> (n_flips) -> count
stable_wrong = 0
stable_total = 0
per_dataset = defaultdict(lambda: {"stable_wrong": 0, "n": 0})

for k in common:
    seq = [data[kr][k]["model_output"].strip()[:1].upper() for kr in LADDER]
    flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    base_correct = is_correct(data["1.00"][k])

    n_flips_hist[flips] += 1
    xtab[base_correct][flips] += 1

    if flips == 0:                       # answer never changed across all budgets
        stable_total += 1
        if not base_correct:
            stable_wrong += 1
            per_dataset[data["1.00"][k]["dataset"]]["stable_wrong"] += 1
    per_dataset[data["1.00"][k]["dataset"]]["n"] += 1

# ---- report ----
print("=" * 64)
print("Flip-count distribution (0 = answer identical across all budgets)")
print("=" * 64)
tot = sum(n_flips_hist.values())
for f in sorted(n_flips_hist):
    c = n_flips_hist[f]
    print(f"  {f} flips: {c:6d}  ({100*c/tot:5.1f}%)")

print("\n" + "=" * 64)
print("Correctness vs stability  (does stability track correctness?)")
print("=" * 64)
for bc in (True, False):
    row = xtab[bc]
    n = sum(row.values())
    stable = row.get(0, 0)
    print(f"  baseline {'CORRECT' if bc else 'WRONG  '}: n={n:6d} | "
          f"{100*stable/n:5.1f}% perfectly stable (0 flips)")

print("\n" + "=" * 64)
print("The language-prior signature: STABLE-BUT-WRONG")
print("(answer never moved even as 90% of visual tokens were removed,")
print(" yet the answer is wrong -> candidate prior-driven guess)")
print("=" * 64)
print(f"  stable (0-flip) samples: {stable_total}")
print(f"  of those, WRONG:         {stable_wrong}  ({100*stable_wrong/max(stable_total,1):.1f}% of stable)")
print(f"  stable-wrong as % of ALL: {100*stable_wrong/len(common):.1f}%")

print("\n  per-dataset stable-wrong rate:")
for ds in sorted(per_dataset):
    d = per_dataset[ds]
    print(f"    {ds:<28} {d['stable_wrong']:5d} / {d['n']:6d}  ({100*d['stable_wrong']/max(d['n'],1):4.1f}%)")
