# huatuo-llava-v15-med-pruning

Visual-token-pruning experiments on **HuatuoGPT-Vision-7B** (LLaVA-v1.5
architecture, Qwen2-7B LLM backbone), evaluated on the standard medical-VQA
benchmark suite: VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, and
MMMU-Medical-Tracks.

## Project context

This is the active repository for the project **Question-Aware Visual Token
Pruning for Medical VLMs**. It is the third (and intended-to-be-final) base
model the project has used:

| Phase | Base model | Repo | Status |
|---|---|---|---|
| 1 (May 10–20) | LLaVA-Med v1.0 | [llava-med-pruning-v1](https://github.com/Leokuan0208/llava-med-pruning-v1) | Frozen — scorer bugs, 0/11 MCQ compliance failure |
| 2 (May 21–25) | Qwen2.5-VL-7B-Instruct | [Qwen-v25-vl-med-pruning](https://github.com/Leokuan0208/Qwen-v25-vl-med-pruning) | Frozen — smoke test passed 20/20 but no published reproducibility target |
| **3 (May 25–)** | **HuatuoGPT-Vision-7B (LLaVA-v1.5)** | **this repo** | **Active** |

## Why HuatuoGPT-Vision-7B

Reproducibility-first. The authors of HuatuoGPT-Vision (Chen et al. 2024,
[arXiv:2406.19280](https://arxiv.org/abs/2406.19280)) publish:

- Merged weights on HuggingFace ([HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B)).
- Bundled evaluation data ([Medical_Multimodal_Evaluation_Data](https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data)).
- A one-command eval pipeline (`accelerate launch eval.py`).
- A Table of headline numbers across six benchmarks.

This means our first deliverable on this stack is a **paper reproduction**
rather than a from-scratch eval pipeline build. The published targets:

| | VQA-RAD | SLAKE | PathVQA | PMC-VQA | OmniMedVQA | MMMU H&M |
|---|---|---|---|---|---|---|
| **HuatuoGPT-Vision-7B** | **63.7** | **76.2** | **57.9** | **54.3** | **74.0** | **50.6** |

Successful reproduction (within ~2pp on each) validates our pipeline before
any pruning experiments.

## Repo layout
huatuo-llava-v15-med-pruning/
├── scripts/       # Entry points: download, eval driver, pruning sweeps
├── pruning/       # Pruning method implementations (qsim, random, future scorers)
└── eval/          # Evaluation harness adapters and scorer wrappers

The May 17 pruning hooks from `llava-med-pruning-v1` (qsim, random) will be
ported to this repo since HuatuoGPT-Vision-7B is architecturally similar to
LLaVA-Med v1.0 (both LLaVA-v1.5-family).

## Progress tracking

Day-by-day progress, experiment results, bug log, and related-work notes:
[Leokuan0208/question-aware-vtp-medvlm](https://github.com/Leokuan0208/question-aware-vtp-medvlm).
Each day-page links the relevant commit in this repo.
