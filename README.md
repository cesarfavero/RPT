# RPT

RPT is an open research and engineering project for packed ternary inference.

The current practical track takes a Gemma 2B-class model through pruning,
ternary MLP conversion, packed-path recovery, Triton runtime integration, and
production-style bundle packaging for Linux/NVIDIA/Colab environments.

Official project site:

- https://rpt.dog

## Public Repository Map

- [`RPT`](https://github.com/cesarfavero/RPT): research source of truth, notebooks, validation, papers, result artifacts, and reproducibility documentation.
- [`rpt-runtime`](https://github.com/cesarfavero/rpt-runtime): runtime and CLI layer for packaged RPT artifacts.
- [`rpt-model-bundles`](https://github.com/cesarfavero/rpt-model-bundles): bundle manifests, checksums, release metadata, and bundle distribution notes.

Use GitHub Issues in the relevant repository for bugs, reproduction reports,
documentation gaps, and integration questions.

## What Was Validated

The current validated track lives in [`gemma4_theory/`](gemma4_theory/).

Local 100-question benchmark:

- original HF `google/gemma-2-2b-it`: `0.90`
- improved dense checkpoint `student_pruned`: `0.92`
- packed ternary recovered path: `1.00`
- Triton-integrated packed runtime: `1.00`

Operational result on T4 after production prewarm:

- `avg_tokens_per_sec ~= 5.66`
- `min_latency ~= 0.35s`
- `max_latency ~= 1.02s`

Bounded claim:

RPT currently demonstrates a reproducible dense-to-packed-ternary pipeline and
runtime path for the validated Linux/NVIDIA/Colab target. It is not presented
as a universal drop-in replacement for every hardware/runtime environment.

## Where To Start

Read in this order:

1. [`gemma4_theory/README.md`](gemma4_theory/README.md): main technical narrative and current validated path.
2. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): reproducibility expectations and artifact policy.
3. [`MODEL_CARD.md`](MODEL_CARD.md): model scope, intended use, and limitations.
4. [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md): current risks and known gaps.
5. [`docs/`](docs/): reports, audits, checklists, and governance notes.

Runtime-specific work belongs in:

- [`rpt-runtime`](https://github.com/cesarfavero/rpt-runtime)

Bundle metadata belongs in:

- [`rpt-model-bundles`](https://github.com/cesarfavero/rpt-model-bundles)

## Repository Layout

- `gemma4_theory/`: current Gemma 2B-class dense-to-pruned-to-packed-ternary runtime track.
- `notebooks/bitnet/`: historical BitNet, sparsity, QAT, and predictive-coding notebooks.
- `docs/`: reports, technical notes, security snapshots, checklists, and governance.
- `paper/`: paper source and publication-related files.
- `ops/`: operational support material.
- `sessions/`: historical experiment logs.
- `model.py`, `bitnet.py`, `trainer.py`: compact source files from the earlier research track.

## Replication Path

The recommended public replication path is:

1. review [`gemma4_theory/README.md`](gemma4_theory/README.md);
2. inspect the notebook order in [`gemma4_theory/notebooks/`](gemma4_theory/notebooks/);
3. verify result artifacts under [`gemma4_theory/results/`](gemma4_theory/results/);
4. review runtime notes under [`gemma4_theory/runtime/packed_recovered_l78/`](gemma4_theory/runtime/packed_recovered_l78/);
5. use [`rpt-runtime`](https://github.com/cesarfavero/rpt-runtime) for CLI/runtime execution work;
6. use [`rpt-model-bundles`](https://github.com/cesarfavero/rpt-model-bundles) for release metadata and bundle references.

## Method Independence

Microsoft BitNet material was useful early conceptual guidance for ternary
thinking, pruning, and low-bit deployment discipline.

The current Gemma track is not a Microsoft BitNet runtime clone. The current
functional path is RPT's own pruning, recovery, packed ternary export,
packed-path recovery, and Triton runtime integration work.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Third-party model weights, datasets, checkpoints, and trademarks remain subject
to their own upstream licenses and terms. In particular, Gemma-family assets
remain subject to the applicable Gemma Terms of Use.
