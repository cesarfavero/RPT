# Packed Recovered L78 Operational Build

Generated at: 2026-04-22 21:17:27

Load order:
1. `student_pruned`
2. materialize `mlp_ternary_2bit.bin` using `manifest.json`
3. apply `student_packed_ternary_recovered_l78_t4_resume`

Main files:
- `packed_recovered_l78_manifest.json`: measured build metadata
- `loader_packed_recovered_l78.py`: standalone loader

Validated packed SHA: `0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116`
Applied packed MLP tensors: `78`

This build keeps the benchmark claim separate from operational measurements. Use notebook 18 for reproduced accuracy and this notebook for runtime/build data.
