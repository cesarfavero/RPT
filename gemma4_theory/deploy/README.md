# RPT Hybrid Ternary MLP Package

This folder contains the lightweight metadata for the packed ternary MLP export.

Runtime package components:

- `manifest.json`: tensor offsets, shapes, scales, and package metadata.
- `mlp_ternary_2bit.bin`: packed 2-bit MLP tensor codes. This file is intentionally not tracked by normal git.
- `../checkpoints/student_ternary_recovered_l78/`: LoRA recovery adapter required to reproduce the functional `l78` result. Large model files are intentionally not tracked by normal git.

The functional `l78` model is hybrid:

```text
packed ternary MLP base + LoRA recovery adapter
```

The packed ternary MLP alone corresponds to the pre-recovery collapsed model and is not the final functional artifact.
