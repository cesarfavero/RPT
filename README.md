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

## Evidence Status

The current validated track lives in [`gemma4_theory/`](gemma4_theory/).

Historical 100-question development suite, scored by the original evaluator:

- original HF `google/gemma-2-2b-it`: `0.90`
- pruned/recovered dense checkpoint `student_pruned`: `0.92`
- packed ternary recovered path: `1.00`
- Triton-integrated packed runtime: `1.00`

The recovery notebooks used these same questions and expected answers for
training and checkpoint selection. The scores demonstrate recovery on the
development suite and functional execution of packed weights. They do not
establish held-out generalization or superiority over the dense reference.

The saved answers have also been regraded with a conservative full-answer
evaluator: both packed paths still score 100/100. This is a regrade of saved
answers, not a new model run. See [the evidence review](docs/reports/EVIDENCE_STATUS_2026-11-02.md).

Historical eight-prompt T4 check after prewarm:

- `avg_tokens_per_sec ~= 5.66`
- `min_latency ~= 0.35s`
- `max_latency ~= 1.02s`

The 5.66 figure is an arithmetic mean of request-level rates, not aggregate
throughput, and does not establish acceleration over a matched dense baseline.
The earlier External500 evaluation used a dense/pruned base plus an older
adapter, not the final packed artifact. The current kernel decodes packed weights
inside tiles and uses a floating-point dense dot product; zero-skipping and a
whole-model 1.58-bit format are not implemented.

Bounded claim:

RPT currently demonstrates a reproducible dense-to-packed-ternary pipeline and
runtime path for the validated Linux/NVIDIA/Colab target. It is not presented
as a universal drop-in replacement for every hardware/runtime environment.

## Independent Validation and Current Research

The executable validation path now includes a versioned exact-match evaluator,
frozen dataset manifests with training-overlap checks, an exact-artifact external
evaluation runner, runtime backend evidence, and CPU regression tests.

```bash
python -m pip install -r validation/requirements-cpu.txt
python -m validation.run_validation --phase cpu
python -m validation.environment --output environment.json
```

Use [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the packed and dense controls,
[SCNET_VALIDATION.md](docs/SCNET_VALIDATION.md) for SCNet/DTK setup, and
[the experimental protocol](validation/protocol.json) for pending model runs.
Passing CPU tests does not certify GPU compatibility or model quality.

The [adaptive output reference experiment](research/adaptive_output/README.md)
tests whether incremental ternary corrections can stop after certifying the
greedy argmax of the same complete representation. Its published results are
synthetic CPU measurements, including slower cases. They are not an LLM benchmark,
an energy result, or a claim of scientific priority.

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

