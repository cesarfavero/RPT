# RPT

Research and engineering repository for two related tracks:

1. `BitNet / sparsity / QAT-STE` research that supported the early paper and methodology stage.
2. `gemma4_theory/`, which is the current applied product track: take a modern dense Gemma 2B-class model, prune it, ternarize the MLP path, recover function, and ship a usable Triton runtime bundle with our own method and runtime stack.

## What Matters Right Now

If you are coming here for the current practical outcome, start in [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md).

That track has already validated the following on the local 100-question benchmark:

- `google/gemma-2-2b-it` original HF reference: `0.90`
- our improved dense checkpoint `student_pruned`: `0.92`
- our packed ternary path after recovered adapter: `1.00`
- our Triton-integrated packed runtime: `1.00`

This is the clearest current claim of the repository:

- we improved the local benchmark result versus the original Gemma 2B baseline;
- we converted the entire MLP path to a packed ternary representation;
- we recovered functionality on top of that packed path;
- we produced a Linux/NVIDIA/Colab production-style runtime bundle.

## From Microsoft Guidance To Our Own Stack

BitNet/Microsoft material was useful at the beginning as a conceptual guide for ternary thinking, pruning, and deployment discipline.

But the current Gemma product track is not a Microsoft BitNet runtime clone. The practical stack now in this repository is our own:

- our own pruning/recovery path over Gemma 2B
- our own packed ternary MLP export
- our own packed-path recovery process
- our own structural and Triton runtime path
- our own production bundle for Linux/NVIDIA/Colab

The honest phrasing is:

- Microsoft/BitNet was a useful starting reference
- the current functional method and runtime are ours

## Product View

The product story is not “we made a toy notebook”.

The product story is:

- start from a model that already answers useful everyday prompts;
- improve its local benchmark score through controlled pruning/recovery;
- compress the MLP path into a packed ternary artifact;
- recover the packed path until it remains usable;
- ship a runtime bundle that a user can download and run in a compatible environment.

What is sellable now:

- a technically usable packed ternary Gemma-derived runtime bundle for Linux + NVIDIA + Colab;
- a reproducible research-to-runtime pipeline built on our own method and runtime stack;
- a concrete benchmark story: the original HF Gemma 2B reference scored `0.90`, our improved dense checkpoint reached `0.92`, and our packed recovered runtime reached `1.00` on the local suite;
- a strong demonstration that ternary conversion plus recovery can preserve and improve local task behavior while making the MLP path materially more compact.

What is not honest to claim yet:

- native Apple Silicon production runtime;
- Ollama compatibility;
- broad external benchmark superiority;
- production SLA across many hardware classes.

## Current Status by Track

### 1. BitNet / Paper Track

This repository still contains the earlier BitNet-focused work:

- paper source: `paper/rpt_sparsity_paper.tex`
- deployment scripts: `deploy/`
- historical notebooks and tracking files

That remains important for the research history, but it is not the only active value in the repository anymore.

### 2. Gemma Production Track

The practical current path lives in `gemma4_theory/`.

Key validated milestones:

- baseline original Gemma 2B HF reference on local suite: `0.90`
- our improved dense checkpoint `student_pruned`: `0.92`
- full MLP ternary expansion recovered on local suite: `0.92`
- packed/materialized recovered path: `1.00`
- Triton integrated runtime: `1.00`
- post-prewarm production bundle verification on T4:
  - `avg_tokens_per_sec ~= 5.66`
  - `min_latency ~= 0.35s`
  - `max_latency ~= 1.02s`

That means the project is past “theory only”. It already has a working runtime bundle for the supported environment, and the current practical path is no longer dependent on Microsoft BitNet tooling.

## Where To Start

### If you want the research narrative

Open:

- [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md)

### If you want the packed artifact/runtime view

Open:

- [`gemma4_theory/deploy/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/deploy/README.md)
- [`gemma4_theory/runtime/packed_recovered_l78/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/runtime/packed_recovered_l78/README.md)

### If you already downloaded the production zip

The target flow is:

```bash
unzip production_triton_packed_recovered_l78.zip
cd production_triton_packed_recovered_l78
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

More detail is documented in the runtime and deploy READMEs.

## Repository Layout

- `paper/`: paper source and publication-related files.
- `deploy/`: legacy BitNet deploy track.
- `gemma4_theory/`: current dense-to-pruned-to-ternary-to-runtime track.
- `sessions/`: historical experiment logs.
- `results/`: JSON/CSV outputs for tracked experiments.

## Reproducibility

The rule of this repository is simple:

- notebooks save timestamped JSON artifacts;
- results are not edited by hand;
- every useful experimental stage should end with git commit + push.

## License

MIT License (`LICENSE`).
