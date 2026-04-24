# Gemma 2B Ternary Runtime Track

This folder is the current practical track of the repository.

Goal:

- start from a dense modern Gemma 2B-class model;
- prune it without losing function;
- ternarize the full MLP path progressively;
- recover the ternary path until it remains usable;
- export the packed ternary artifact;
- integrate a real runtime;
- ship a usable bundle for Linux/NVIDIA/Colab.

This is not only a theory folder anymore. It now contains a reproducible path from benchmarked model surgery to a production-style runtime bundle.

## Executive Summary

Local 100-question benchmark:

- original `google/gemma-2-2b-it`: `0.90`
- `student_pruned`: `0.92`
- packed ternary recovered path: `1.00`
- Triton integrated runtime: `1.00`

Operational result on T4 after production prewarm:

- `avg_tokens_per_sec ~= 5.66`
- `min_latency ~= 0.35s`
- `max_latency ~= 1.02s`

Honest reading:

- the local benchmark improved versus the original Gemma 2B reference;
- the MLP path was fully converted to packed ternary;
- the packed path is functionally recoverable;
- the current runtime is usable in Linux/NVIDIA/Colab;
- the remaining work is portability and broader external validation, not “does this pipeline work at all?”.

## What Was Proved

### 1. Pruning helped instead of hurting on the local suite

Reference evaluation:

- original HF Gemma 2B: `0.90`
- `student_pruned`: `0.92`

That matters because the whole ternary track is built on top of `student_pruned`. The base checkpoint was not degraded; it improved locally.

### 2. Direct ternary snap alone was not enough

The early ternary attempt degraded heavily. That established an important constraint:

- ternary conversion without guided recovery is not the product;
- ternary conversion plus recovery is the product path.

### 3. Progressive ternary expansion worked

The project did not jump directly from dense to fully ternary MLPs. It expanded in stages.

Important milestones:

- full MLP expansion at `78` ternary MLP projections reached post-recovery `0.92`
- that proved the clean checkpoint path was viable

### 4. Packed ternary export worked

The ternary MLPs were exported into:

- `deploy/manifest.json`
- `deploy/mlp_ternary_2bit.bin`

This is the compact artifact that represents the ternary MLP path.

### 5. Packed path recovery worked

The packed/materialized path is much harder than the clean checkpoint path.

Key transition:

- packed base without adapter: `0.01`
- older adapter on packed path: `0.63`
- recovered packed path (`17`/`18`): `1.00`

That is the point where the project stopped being “interesting only in notebooks” and became a real runtime candidate.

### 6. Triton runtime integration worked

The Triton runtime replaced all `78` MLP base layers and preserved the packed recovered behavior.

Validated facts:

- `runtime_kind = native_triton_prototype`
- `is_native_ternary_runtime = true`
- `is_optimized_kernel = true`
- `replaced_count = 78`
- local suite under Triton runtime: `1.00`

### 7. The latency tail was diagnosed and controlled

The first clean runtime benchmark showed a bad latency tail.

The diagnosis notebook proved that:

- steady-state inference was not the issue;
- the long tail came from cold compile / warmup per prompt shape;
- after prewarm, rep1 and rep>1 converge to a stable latency range.

That enabled the production bundle stage.

## Product Claim

The correct product claim today is:

- a Gemma-derived packed ternary runtime bundle for Linux/NVIDIA/Colab that preserves the recovered local benchmark behavior and is operational after prewarm.

The incorrect product claim would be:

- “universal drop-in Gemma replacement everywhere”
- “already optimized for M1/Ollama”
- “globally benchmark-superior across external suites”

## Folder Map

### Data

- `data/gemma4_100_questions.json`: local benchmark used throughout the track.

### Notebooks

Core notebook order:

1. `01_baseline_eval_gemma4.ipynb`
2. `02_progressive_pruning_gemma4.ipynb`
3. `03_pruned_checkpoint_eval_gemma4.ipynb`
4. `04_quantization_eval_gemma4.ipynb`
5. `05_ternary_next_step_gemma4.ipynb`
6. `06_ternary_recovery_gemma4.ipynb`
7. `07_progressive_ternary_expansion_gemma4.ipynb`
8. `08_consolidate_mlp_ternary_results.ipynb`
9. `09_ternary_deployment_audit.ipynb`
10. `10_export_packed_ternary_mlp.ipynb`
11. `11_validate_hybrid_ternary_package.ipynb`
12. `12_external_eval_l78_hybrid.ipynb`
13. `13_external_eval_scaled_l78.ipynb`
14. `14_validate_packed_ternary_reconstruction.ipynb`
15. `15_recover_packed_ternary_path.ipynb`
16. `16_reference_gemma2b_question_eval.ipynb`
17. `17_t4_resume_packed_recovery.ipynb`
18. `18_test_packed_recovered_l78.ipynb`
19. `20_measure_and_build_packed_recovered_l78.ipynb`
20. `21_build_distributable_packed_recovered_l78.ipynb`
21. `22_native_ternary_mlp_runtime.ipynb`
22. `23_triton_ternary_matmul_spike.ipynb`
23. `24_integrate_triton_ternary_runtime.ipynb`
24. `25_benchmark_triton_runtime.ipynb`
25. `26_diagnose_triton_latency_tail.ipynb`
26. `27_build_production_triton_bundle.ipynb`

### Runtime and Bundle

- `runtime/packed_recovered_l78/`: runtime source files and CLIs.
- `deploy/`: packed MLP artifact metadata and compact package files.
- `build/production_triton_packed_recovered_l78/`: generated production bundle directory.

### Results

- `results/`: timestamped JSON outputs that act as the canonical record.

## Key Results

### Reference vs Improved Base

- original Gemma 2B reference: `0.90`
- `student_pruned`: `0.92`

### Quantization Stage

- `pruned_reference = 0.92`
- `pruned_8bit = 0.92`
- `pruned_4bit = 0.91`

### Full MLP Ternary Expansion

- `78` ternary MLP projections
- post-recovery clean path: `0.92`

### Packed Recovered Path

- packed base: `0.01`
- old adapter on packed path: `0.63`
- recovered packed path: `1.00`

### Triton Runtime

- full local suite: `1.00`
- `78` MLP base layers replaced
- persistent CUDA allocated after load: `2.40 GB`
- persistent CUDA reserved after load: `5.28 GB`

### Runtime Performance Diagnosis

Cold benchmark without production prewarm:

- `p50 ~= 0.46s`
- `p95 ~= 22.67s`
- clear shape-compilation tail on first use

Post-prewarm diagnosis:

- `warm p50 ~= 0.48s`
- `warm p95 ~= 0.67s`
- average unweighted throughput `~= 7.50 tok/s`

Production bundle verification:

- `avg_tokens_per_sec ~= 5.66`
- `min_latency ~= 0.35s`
- `max_latency ~= 1.02s`

## Recommended Current Entry Points

### If you want the research proof

Read:

- `results/packed_recovery/24_integrate_triton_ternary_runtime_20260424_192203.json`
- `results/packed_recovery/runtime_benchmarks/25_benchmark_triton_runtime_20260424_202518.json`
- `results/packed_recovery/runtime_benchmarks/26_diagnose_triton_latency_tail_20260424_204619.json`
- `results/packed_recovery/production_bundle/27_build_production_triton_bundle_20260424_212734.json`

### If you want the usable package

Read:

- [`deploy/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/deploy/README.md)
- [`runtime/packed_recovered_l78/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/runtime/packed_recovered_l78/README.md)

## What The User Gets Today

Today the user gets:

- a production-style zip bundle for Linux/NVIDIA/Colab;
- a compact packed ternary MLP artifact plus recovered adapter;
- Triton runtime entrypoint;
- prewarm profile and production manifest;
- a model that is locally stronger than the original Gemma reference on the project suite.

## What Still Needs Work

Still open:

- broader external benchmark validation for the final production bundle itself;
- portability to consumer Windows/Linux GPU environments beyond the current tested path;
- Apple Silicon / Ollama-style runtime compatibility;
- cleaner dependency pinning around `peft` / `torchao` warnings;
- production API/server wrappers.

## Rules of This Track

- every notebook writes timestamped artifacts into `results/`;
- results are never edited by hand;
- the repository should close each useful stage with commit + push.
