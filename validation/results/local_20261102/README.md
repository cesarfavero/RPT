# Local validation, 2026-11-02

The RPT suite passed all 85 tests. The runtime suite discovered 48 tests:
42 passed and 6 were skipped because PyTorch is unavailable. One of the skipped
tests also requires Triton and two GPUs. Logs and tested source hashes are
included here. Python bytecode compilation and edited notebook syntax passed.

The combined CPU/kernel run correctly returned `blocked`, exit code 2, because
the accelerator dependencies are absent. This is not a failed CPU test suite
and is not a successful GPU validation. No language model or external benchmark
was executed. `model_quality_evaluated` remains false.

Local Git commits in the automatically generated report refer to partial source
snapshots assembled for this review. The public baseline commits and the exact
tested file hashes are recorded in `context.json` and `report.json`. CI records
will identify the published commit independently.

The historical regrade in the parent directory processes saved answers. The
adaptive-output experiment under `research/` is a separate synthetic CPU result.
Neither replaces final-bundle validation on SCNet/NVIDIA.
