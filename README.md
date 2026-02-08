# RPT: Thermodynamic Predictive Networks

Research repository for RPT experiments on ternary language models (BitNet b1.58), sparse pruning, and QAT/STE deployment.

## Current Status (Important)

- **Primary validated result (2026-02-08):** BitNet b1.58-2B-4T + sparsity + QAT/STE deploy pipeline.
- **Canonical outcome:** PPL **25.13 -> 16.39** (**-34.8%**), **42.6%** sparsity, coherent CPU generation via GGUF i2_s.
- **Scope note:** this repository also contains historical/auxiliary experiments. They provide context, but are **not** the main publication claim.

## Paper Status

- **arXiv:** submission in progress (**pending endorsement/moderation**).
- **Current source of truth:** `paper/rpt_sparsity_paper.tex` in this repository.
- The arXiv URL will be added here once it is publicly available.

## Main Entry Points

- `paper/rpt_sparsity_paper.tex`: publication source for the primary claim.
- `sessions/2026-02-08/tracking.md`: canonical metrics/log for the successful deploy run.
- `deploy/rpt_deploy_a100.py`: end-to-end deploy script for the successful pipeline.
- `RPT_BitNet_Progressive_Sparsity.ipynb`: progressive pruning experiment.
- `RPT_BitNet_Sparsity_Test.ipynb`: raw pruning experiment.

## Highlights

- Raw magnitude pruning on BitNet 2B improves PPL by **26.1%** at 10% sparsity (no fine-tuning).
- Progressive pruning + fine-tuning on H100 improves PPL by up to **40.4%**.
- Deploy pipeline (A100, QAT/STE, ternary snap, GGUF i2_s) reaches **PPL 16.39** from baseline **25.13** (**-34.8%**), with **42.6% sparsity** and coherent CPU generation.
- Criticality signal: Lyapunov exponent approximately **-0.002** (near edge-of-chaos regime).

## Public Artifacts

- Model weights: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned
- GGUF artifacts: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned-GGUF

## Repository Contents

- `paper/`: paper source (`rpt_sparsity_paper.tex`) and NeurIPS style file.
- `deploy/`: deployment scripts/notebooks (no large binaries committed).
- `sessions/2026-02-08/tracking.md`: deploy tracking and final metrics.
- Core code: `bitnet.py`, `model.py`, `trainer.py`.
- Experiment notebooks:
  - `RPT_BitNet_Microsoft.ipynb`
  - `RPT_BitNet_Sparsity_Test.ipynb`
  - `RPT_BitNet_Progressive_Sparsity.ipynb`
  - `RPT_BitNet_Predictive_Coding.ipynb`
  - `RPT_BitNet_Criticality.ipynb`
- Documentation:
  - `RPT_DOCUMENTO_COMPLETO.md`
  - `RPT_VALIDACAO_BITNET2B.md`
  - `REPRODUCIBILITY.md`
  - `MODEL_CARD.md`
  - `KNOWN_ISSUES.md`

## Historical Context

- Earlier architecture/conversion explorations are kept for transparency and reproducibility context.
- They should be interpreted as intermediate research artifacts, not as the final validated deliverable.

## Quick Start

```bash
pip install -r requirements.txt
```

To inspect main results, open and run notebooks listed above.

## Reproducibility Notes

- See `REPRODUCIBILITY.md` for expected numbers and experiment-by-experiment instructions.
- Environment-specific baseline PPL differs across T4/H100/A100 due precision/runtime settings; compare within the same environment.

## Citation

See `CITATION.cff`.

## License

MIT License (`LICENSE`).
