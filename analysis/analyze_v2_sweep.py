#!/usr/bin/env python3
"""
Day 18 analysis: v2 pre-LLM sweep comparison table + Pareto plots.

Reads:
  results/v2_pre_llm/2026-05-26_{qsim,random}_kr{0.1,0.25,0.5,0.75}/
    {model}__*__scores.json
    {model}__*__latency_summary.json

Writes (into analysis/):
  2026-05-26_v2_sweep_keys.txt          first scores file's structure
  2026-05-26_v2_sweep_table.md          markdown comparison table
  2026-05-26_v2_sweep_table.csv         same numbers, machine-readable
  2026-05-26_v2_sweep_pareto_total.png  total accuracy vs kr
  2026-05-26_v2_sweep_pareto_panels.png 6-benchmark grid
  2026-05-26_v2_sweep_latency.png       accuracy vs latency
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "v2_pre_llm"
OUT_DIR = REPO / "analysis"
OUT_DIR.mkdir(exist_ok=True)

# Paper-reproduction baseline (kr=1.0) recorded May 25
# These are the numbers we compute deltas against.
BASELINE = {
    "total":   0.6787,
    "VQA-RAD": 0.6135,
    "SLAKE":   0.7644,
    "PathVQA": 0.5767,
    "PMC-VQA": 0.5420,
    "OmniMed": 0.7346,
    "MMMU":    0.5034,
}

# The scores JSON likely uses slightly different key names than our
# display labels. This maps each display label to a list of candidate
# JSON keys to try in order. If none match, we warn and leave the cell blank.
KEY_ALIASES = {
    "total":   ["The total score for multiple-choice questions", "total", "overall"],
    "VQA-RAD": ["VQA-RAD_test", "VQA-RAD", "vqa_rad"],
    "SLAKE":   ["SLAKE_test", "SLAKE", "slake"],
    "PathVQA": ["PathVQA_test", "PathVQA", "path_vqa"],
    "PMC-VQA": ["PMC-VQA_test", "PMC-VQA", "pmc_vqa"],
    "OmniMed": ["OmniMedVQA", "OmniMed", "OmniMedVQA_test"],
    "MMMU":    ["MMMU_Medical_Validation", "MMMU", "MMMU_test"],
}

# Method colors for plots (consistent across all figures)
COLORS = {
    "random":    "#1f77b4",  # blue — sanity floor
    "qsim_mean": "#d62728",  # red — yesterday's QSim
    "qsim_max":  "#2ca02c",  # green — today's max-reduction QSim
}
METHODS_IN_PLOT_ORDER = ["random", "qsim_mean", "qsim_max"]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def parse_run_name(folder_name):
    """
    2026-05-26_qsim_kr0.1     -> {'method': 'qsim_mean', 'kr': 0.1}
    2026-05-26_random_kr0.1   -> {'method': 'random',    'kr': 0.1}
    2026-05-27_qsim_max_kr0.1 -> {'method': 'qsim_max',  'kr': 0.1}
    """
    m = re.match(r"\d{4}-\d{2}-\d{2}_(qsim_max|qsim|random)_kr([\d.]+)$", folder_name)
    if not m:
        return None
    method = m.group(1)
    # Rename plain 'qsim' to 'qsim_mean' so the three methods are
    # explicitly distinguished. Yesterday's qsim sweep was mean-reduction.
    if method == "qsim":
        method = "qsim_mean"
    return {"method": method, "kr": float(m.group(2))}

def lookup_score(scores_dict, display_label):
    """Find a score by trying each alias key in order. Returns float or None."""
    for key in KEY_ALIASES[display_label]:
        if key in scores_dict:
            return float(scores_dict[key])
    return None

# ---------------------------------------------------------------------------
# Discover and load all runs
# ---------------------------------------------------------------------------

print(f"Looking for runs under: {RESULTS_DIR}")
runs = []
for folder in sorted(RESULTS_DIR.glob("2026-05-2[67]_*")):
    if not folder.is_dir():
        continue
    info = parse_run_name(folder.name)
    if info is None:
        continue

    scores_files = list(folder.glob("*__scores.json"))
    lat_files = list(folder.glob("*__latency_summary.json"))
    if not scores_files:
        print(f"  [skip] no scores file in {folder.name}")
        continue
    if not lat_files:
        print(f"  [warn] no latency summary in {folder.name}")

    with open(scores_files[0]) as f:
        scores = json.load(f)
    latency = {}
    if lat_files:
        with open(lat_files[0]) as f:
            latency = json.load(f)

    runs.append({
        "folder": folder.name,
        "method": info["method"],
        "kr": info["kr"],
        "scores_path": scores_files[0],
        "scores": scores,
        "latency": latency,
    })
    print(f"  found: {folder.name}")

if not runs:
    print("ERROR: No runs found. Check RESULTS_DIR path.", file=sys.stderr)
    sys.exit(1)

print(f"\nLoaded {len(runs)} runs.\n")

# ---------------------------------------------------------------------------
# Dump the structure of one scores file (so key-name mismatches are visible)
# ---------------------------------------------------------------------------

keys_path = OUT_DIR / "2026-05-26_v2_sweep_keys.txt"
with open(keys_path, "w") as f:
    f.write(f"Source file: {runs[0]['scores_path']}\n\n")
    f.write("Top-level keys and values:\n")
    f.write(json.dumps(runs[0]["scores"], indent=2))
print(f"Wrote scores-schema dump:  {keys_path}")

# Sanity check: which display labels resolved, which didn't, on run 0
first_scores = runs[0]["scores"]
resolved = {}
missing = []
for label in KEY_ALIASES:
    val = lookup_score(first_scores, label)
    if val is None:
        missing.append(label)
    else:
        resolved[label] = val
print(f"\nKey resolution on first run ({runs[0]['folder']}):")
for label, val in resolved.items():
    print(f"  {label}: {val:.4f}")
if missing:
    print(f"  MISSING (keys not found): {missing}")
    print(f"  -> Check {keys_path} and update KEY_ALIASES, then re-run.")

# ---------------------------------------------------------------------------
# Build the comparison table
# ---------------------------------------------------------------------------

BENCHMARKS = list(BASELINE.keys())  # total + 6 benchmarks

# Sort: baseline-like ordering, qsim then random within each kr (descending kr)
runs_sorted = sorted(runs, key=lambda r: (-r["kr"], r["method"]))

md_lines = []
md_lines.append("| K | Method | " + " | ".join(BENCHMARKS) + " | Latency (ms) |")
md_lines.append("|---:|:------|" + "|".join(["------:"] * len(BENCHMARKS)) + "|----:|")

# Baseline row
base_cells = [f"**{BASELINE[b]:.4f}**" for b in BENCHMARKS]
md_lines.append(f"| 1.00 | baseline | " + " | ".join(base_cells) + " | — |")

csv_lines = ["kr,method," + ",".join(BENCHMARKS) + ",mean_latency_ms,p95_latency_ms,visual_post_prune"]
csv_lines.append(f"1.00,baseline,"
                 + ",".join(f"{BASELINE[b]:.4f}" for b in BENCHMARKS)
                 + ",,,576")

for r in runs_sorted:
    row_md = [f"| {r['kr']:.2f} | {r['method']}"]
    row_csv = [f"{r['kr']:.2f},{r['method']}"]
    for b in BENCHMARKS:
        val = lookup_score(r["scores"], b)
        if val is None:
            row_md.append(" | — ")
            row_csv.append("")
        else:
            delta = val - BASELINE[b]
            sign = "+" if delta >= 0 else ""
            row_md.append(f" | {val:.4f} ({sign}{delta:.4f})")
            row_csv.append(f"{val:.4f}")

    lat = r["latency"]
    mean_ms = lat.get("mean_time_s", 0) * 1000
    p95_ms = lat.get("p95_time_s", 0) * 1000
    vis = lat.get("mean_visual_post_prune", "")
    row_md.append(f" | {mean_ms:.1f} |")
    row_csv.append(f"{mean_ms:.2f},{p95_ms:.2f},{vis}")

    md_lines.append("".join(row_md))
    csv_lines.append(",".join(row_csv))

table_md_path = OUT_DIR / "2026-05-26_v2_sweep_table.md"
table_md_path.write_text("\n".join(md_lines) + "\n")
csv_path = OUT_DIR / "2026-05-26_v2_sweep_table.csv"
csv_path.write_text("\n".join(csv_lines) + "\n")
print(f"\nWrote table (markdown): {table_md_path}")
print(f"Wrote table (csv):      {csv_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Plot 1: total accuracy vs keep-ratio, both methods
fig, ax = plt.subplots(figsize=(7, 5))
for method in METHODS_IN_PLOT_ORDER:
    method_runs = sorted([r for r in runs if r["method"] == method], key=lambda r: r["kr"])
    xs = [r["kr"] for r in method_runs]
    ys = [lookup_score(r["scores"], "total") for r in method_runs]
    ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, linewidth=2)
# Baseline reference line
ax.axhline(BASELINE["total"], color="gray", linestyle="--", linewidth=1, label=f"baseline (kr=1.0) = {BASELINE['total']:.4f}")
ax.set_xlabel("Keep ratio")
ax.set_ylabel("Total accuracy (mean across 6 benchmarks)")
ax.set_title("v2 sweep — total accuracy vs keep-ratio")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 1.0)
fig.tight_layout()
plot1_path = OUT_DIR / "2026-05-26_v2_sweep_pareto_total.png"
fig.savefig(plot1_path, dpi=120)
plt.close(fig)
print(f"Wrote plot:             {plot1_path}")

# Plot 2: per-benchmark grid (2 rows x 3 cols)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for ax, b in zip(axes.flat, [k for k in BENCHMARKS if k != "total"]):
    for method in METHODS_IN_PLOT_ORDER:
        method_runs = sorted([r for r in runs if r["method"] == method], key=lambda r: r["kr"])
        xs = [r["kr"] for r in method_runs]
        ys = [lookup_score(r["scores"], b) for r in method_runs]
        ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, linewidth=2)
    ax.axhline(BASELINE[b], color="gray", linestyle="--", linewidth=1)
    ax.set_title(b)
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
for ax in axes[1]:
    ax.set_xlabel("keep ratio")
axes[0, 0].legend(loc="lower right")
fig.suptitle("v2 sweep — per-benchmark accuracy vs keep-ratio", y=1.02)
fig.tight_layout()
plot2_path = OUT_DIR / "2026-05-26_v2_sweep_pareto_panels.png"
fig.savefig(plot2_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote plot:             {plot2_path}")

# Plot 3: accuracy vs latency (the speed-accuracy Pareto)
fig, ax = plt.subplots(figsize=(8, 5))
for method in METHODS_IN_PLOT_ORDER:
    method_runs = sorted([r for r in runs if r["method"] == method], key=lambda r: r["kr"])
    xs = [r["latency"].get("mean_time_s", 0) * 1000 for r in method_runs]
    ys = [lookup_score(r["scores"], "total") for r in method_runs]
    krs = [r["kr"] for r in method_runs]
    ax.plot(xs, ys, marker="o", color=COLORS[method], label=method, linewidth=2)
    for x, y, kr in zip(xs, ys, krs):
        ax.annotate(f"kr={kr}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.axhline(BASELINE["total"], color="gray", linestyle="--", linewidth=1, label=f"baseline accuracy")
ax.set_xlabel("Mean latency per sample (ms)")
ax.set_ylabel("Total accuracy")
ax.set_title("v2 sweep — accuracy vs latency Pareto\n(up-and-left is better)")
ax.legend()
ax.grid(alpha=0.3)
ax.invert_xaxis()  # so "left" is "faster", matching the convention
fig.tight_layout()
plot3_path = OUT_DIR / "2026-05-26_v2_sweep_latency.png"
fig.savefig(plot3_path, dpi=120)
plt.close(fig)
print(f"Wrote plot:             {plot3_path}")

print("\nDone.")
