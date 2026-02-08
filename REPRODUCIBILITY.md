# Reproducibility Guide

This document defines the minimal path to reproduce the key claims reported in this repository.

## Scope

Reproduces the BitNet 2B sparsity and deploy pipeline claims:

1. Raw pruning improves PPL at 10% sparsity (T4).
2. Progressive pruning + fine-tuning improves PPL across 5%-50% (H100, AdamW).
3. QAT/STE deploy pipeline produces coherent GGUF output with final PPL 16.39 (A100).
4. Criticality measurements (Lyapunov and branching-ratio trend).

## Expected Reference Metrics

- Raw pruning (T4):
  - 0% baseline PPL: 9.39
  - 10% pruning PPL: 6.94 (-26.1%)
- Progressive pruning + FT (H100):
  - 0% baseline PPL: 25.10
  - Best at 10%: PPL 14.97 (-40.4%)
- QAT/STE deploy (A100):
  - Baseline PPL: 25.13
  - Final post-snap PPL: 16.39 (-34.8%)
  - Final sparsity: 42.6%
  - Ternary ratio: 100%
- Criticality:
  - Lyapunov: -0.002 (approx)
  - Total amplification: 0.94x

## Environment Notes

- Baseline PPL differs by hardware/runtime settings.
- Compare only against baseline computed in the same environment.
- T4/H100/A100 values are not directly cross-comparable.

## Step-by-Step

## 1) Sanity / base model behavior

Notebook: `RPT_BitNet_Microsoft.ipynb`

Goal:
- Verify base model loads and generates coherent text.

## 2) Raw pruning (no fine-tuning)

Notebook: `RPT_BitNet_Sparsity_Test.ipynb`

Goal:
- Reproduce 10% pruning improvement on T4-scale setup.

## 3) Progressive pruning + AdamW fine-tune

Notebook: `RPT_BitNet_Progressive_Sparsity.ipynb`

Goal:
- Reproduce 5%-50% table and 10% best point.

## 4) Activation pruning negative result

Notebook: `RPT_BitNet_Predictive_Coding.ipynb`

Goal:
- Reproduce degradation when pruning activations by percentile.

## 5) Criticality analysis

Notebook: `RPT_BitNet_Criticality.ipynb`

Goal:
- Reproduce Lyapunov near 0 and branching-ratio depth trend.

## 6) Deploy pipeline (QAT/STE + snap + GGUF)

Script/Notebook:
- `deploy/rpt_deploy_a100.py`
- `deploy/RPT_GGUF_CPU.ipynb`

Reference log:
- `sessions/2026-02-08/tracking.md`

Goal:
- Reproduce final deploy metrics and coherent CPU generation.

## Known Reproducibility Limits

- Public release currently includes model and GGUF artifacts.
- Full clean-room automation script for all experiments is not yet consolidated in a single entrypoint.
- Some experiments rely on notebook workflows and cloud runtime specifics.
