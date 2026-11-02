# RPT-9 Model Optimization Roadmap

## Priority update (2026-11-02)

Evaluate the final packed artifact on frozen external data before selecting
more checkpoints or changing recovery. This supersedes placing external
validation after optimization in the historical sequence below.

Portable evaluation and integrity checks, hardware preflight, a strict Triton
probe and a synthetic adaptive-output reference are now implemented. Actual
model evaluation, SCNet/NVIDIA measurements, a packed-first loader, optimized
kernels, energy measurements and matched-budget recovery controls are pending.
See [the evidence review](EVIDENCE_STATUS_2026-11-02.md) and
[the declared campaign](../../validation/protocol.json).


This file turns `RPT-9` into an execution sequence instead of a loose idea list.

Constraint:

- public install remains disabled until the runtime is shipped as a compiled distribution
- optimization work should improve the eventual product surface, not just notebook metrics

## Objective

Resume technical development with a clear order, concrete gates, and target hardware profiles.

Done for `RPT-9` means:

- hardware targets are explicit
- the next optimization sequence is fixed
- each stage has a pass/fail gate
- output artifacts and decision points are documented

## Product Hardware Targets

These are the target profiles to optimize against first:

1. `linux_nvidia_t4`
   Use:
   - current validated packed runtime path
   - continuity target for quality and runtime behavior

2. `consumer_nvidia_8gb_to_12gb`
   Use:
   - realistic first-user GPU target
   - validates whether the runtime can leave pure notebook territory

3. `cpu_8gb_ram`
   Use:
   - tests whether a lightweight mode is commercially meaningful
   - forces discipline around memory, context, and runtime overhead

Deferred:

- `apple_silicon_base`
- `ollama_style_distribution`

These should not block the next cycle.

## Sequence

### Phase 1. Hardware matrix and measurement contract

Goal:

- freeze what gets measured on every target

Required outputs:

- target profile table
- exact metric schema
- comparable run settings

Metrics:

- load time
- time to first token
- tokens/sec
- peak RAM
- peak VRAM
- model artifact size
- KV-cache footprint estimate
- local 100-question accuracy
- one external eval summary

Gate:

- no optimization branch starts without all metrics defined for all three target profiles

### Phase 2. Layer-wise ternary ablation

Goal:

- determine what must stay above ternary precision

Test slices:

- embeddings
- attention projections
- attention output
- MLP blocks by layer range
- final norm / lm_head

Deliverable:

- per-slice table with quality delta, memory delta, and runtime delta

Gate:

- identify a candidate mixed-precision layout that preserves at least the current local benchmark quality band while reducing memory or improving latency on at least one target profile

### Phase 3. Memory split accounting

Goal:

- stop treating model weight size as the only bottleneck

Breakdown required:

- weights
- packed artifact
- adapter
- KV cache
- activations
- tokenizer/runtime overhead
- compile/warmup overhead where applicable

Deliverable:

- one measurement note per target profile

Gate:

- identify the top two real memory/latency bottlenecks per target profile

### Phase 4. KV-cache reduction strategy

Goal:

- test whether the next real gain comes from context/runtime policy rather than more weight compression

Candidates:

- quantized KV cache
- shorter default context windows
- sliding window
- summary compaction
- prompt/profile presets by hardware tier

Deliverable:

- ranked recommendation with quality and latency tradeoff

Gate:

- choose one default strategy for lightweight mode and one for balanced mode

### Phase 5. Recovery/distillation versus runtime engineering decision

Goal:

- choose the highest-return next branch instead of doing both blindly

Compare:

- more recovery/distillation on the packed path
- runtime engineering on the current recovered path

Decision basis:

- quality delta per engineering week
- latency/memory gain per engineering week
- effect on compiled distribution readiness

Gate:

- written decision selecting one primary branch and one secondary branch

### Phase 6. External validation for the final packed runtime path

Goal:

- validate the actual final path, not just adjacent checkpoints

Minimum:

- one broader external suite on the packed recovered runtime story
- clear statement of what is validated functionally versus operationally

Gate:

- no broad performance claim goes public without this pass

## Suggested Work Packages

### Work package A

- define metric schema
- define target profile table
- create run template for all future measurements

### Work package B

- run layer-wise ternary ablation
- produce mixed-precision candidate layout

### Work package C

- measure memory split
- recommend KV-cache strategy

### Work package D

- choose primary next branch: recovery/distillation or runtime engineering
- prepare external validation scope

## Immediate next actions

- [ ] finalize the three target profiles above as the official working matrix
- [ ] create the measurement template for every run
- [ ] schedule layer-wise ternary ablation as the first optimization experiment
- [ ] measure memory split on `linux_nvidia_t4` before changing more model weights
- [ ] choose whether lightweight mode is GPU-first or CPU-first

## Notes

- current site and public install work should stay blocked from reopening until compiled distribution exists
- `RPT-9` should optimize toward the compiled product path, not toward another Python-distributed runtime bundle

