# Bounded adaptive output projection: reference experiment

This directory is a **synthetic CPU research prototype**. It implements a
testable starting point for incremental ternary output projection. It does not
load an LLM, recover a checkpoint, establish generalization, or demonstrate a
production speedup. It does not claim a new quantization method or scientific
priority.

## What is actually certified

For one supplied hidden vector, return the same greedy output index as evaluating
every plane of the **same completed ternary representation**, using the defined
ordered `float64` arithmetic. If a strict interval certificate cannot establish
the winner, evaluate all planes and return the full reference's first argmax.

This guarantee does not concern the original dense source matrix, the quality of
its hidden vector, the correctness of a generated answer, sampled distributions,
or a complete generated sequence. It also does not imply bitwise equivalence to
BLAS, Triton, CUDA, FP16/BF16, a fused reconstructed weight matrix, or a different
reduction order. A port must establish its own numerical error model.

The current codes are stored as **int8**, not packed ternary. Row scales and bias
are float64. Bound metadata stores one int64 nonzero count per row and plane.
All this persistent array storage is included in the result. Adding planes may
increase total storage substantially. No compression advantage is assumed.

## Representation and full reference

Let each code plane have entries in `{-1, 0, 1}`, with one scale per row:

\[
W = \sum_{p=0}^{P-1} \operatorname{diag}(s_p) T_p.
\]

Plane zero is the base; the remaining planes are corrections. The reference
starts with the bias. For each plane in increasing order, it accumulates the
signed hidden entries in increasing column order with float64 additions,
multiplies the row sum by the plane scale, and adds that result to the logits.
Explicit order makes the comparison unambiguous and avoids silently relying on
BLAS implementation details. It is intentionally not an optimized kernel.

`from_dense` supplies a simple greedy absmax residual ternarization convenience
constructor for experiments. Its output approximates its input matrix. The
certificate remains relative to its completed representation, not its input.

## Conservative numerical bound

In real arithmetic, the unexecuted contribution of a row obeys

\[
|s_{p,i} (T_{p,i} h)| \le |s_{p,i}|\,n_{p,i}\,\|h\|_\infty,
\]

where `n[p, i]` is the number of nonzero ternary codes in that row. This is a
coarse L1 row bound. It reads the small precomputed count and scale arrays, not
every skipped code. Its looseness is a deliberate limitation to investigate.

Using that expression without rounding protection would be insufficient for a
floating-point certificate. This reference constructs an upward-rounded table:

\[
B_0=0,\qquad
B_{k+1}=\operatorname{nextUp}(\operatorname{fl}(B_k+\|h\|_\infty)).
\]

Under IEEE-754 binary64 arithmetic with gradual underflow, `B[n]` bounds the
absolute value of an ordered sum of `n` signed entries. Multiplication by a
ternary code is an exact sign change or zero. Adding zeros cannot increase that
sum's magnitude. Induction and monotonicity of correctly rounded addition give
the enclosure even in the presence of cancellation. `nextUp` deliberately adds
one representable step of conservatism. Subnormal cases are tested.

For each unexecuted plane, an upward-rounded product bounds its scaled row sum.
Starting with the exactly computed floating prefix, update its interval by
outward-rounded subtraction/addition of that plane's bound. Repeating these
steps encloses the logits of the documented full floating reference, including
its future scale multiplication and prefix additions.

A row is certified only when its lower endpoint is **strictly greater** than all
other rows' upper endpoints. Exact or unresolved ties require the full fallback.
If the bounds overflow, defer the decision; a coarse infinite bound need not
mean the actual reference overflows. If actual reference arithmetic becomes
nonfinite, raise an error instead of returning a token with a false certificate.
Nonfinite inputs, malformed shapes and non-ternary codes are rejected.

The implementation assumes standard correctly rounded NumPy elementwise binary64
operations and `nextafter`. A small startup check rejects obvious nonstandard
subnormal behavior. It does not prove arbitrary hardware arithmetic correct.

## Run locally

Only NumPy and Python's standard library are required. From the repository root:

```bash
python -m unittest discover -s tests -p 'test_adaptive_output.py' -v
python -m research.adaptive_output.run_experiment
```

The second command writes `results/reference_cpu.json`. For a smaller smoke run:

```bash
python -m research.adaptive_output.run_experiment \
  --rows 16 --hidden-size 16 --samples 16 --repeats 1 \
  --output /tmp/rpt_adaptive_output_smoke.json
```

No cloud job, network call, dataset download or model download occurs.

## Experiment and accounting

The committed result uses seeded synthetic matrices and hidden vectors, with
four fixtures: small corrections, substantial corrections, identical output
rows causing exact ties, and a greedy approximation of a Gaussian dense matrix.
The fixture definitions intentionally include both early-exit-friendly and
unfriendly cases. They are not representative samples of a language model.

Each fixture reports:

- Exact agreement with the completed representation on all supplied vectors.
- How often a certificate stopped early, and how many planes were executed.
- Complete-call timings for both reference paths, including hidden validation,
  all projections, norm computation, bound-table creation, metadata reads,
  outward rounding, dominance checks, result construction and fallback.
- Separate model construction, correctness verification, warmup and full-run
  times. Timing rounds alternate method order.
- Every persistent ndarray byte, supplied input bytes, measured traced peak
  call allocations, and whole-process high-water RSS. Tracemalloc is not a GPU
  profiler or a guarantee that every native allocator is visible.
- Defined scalar work counters for projection and auxiliary operations. These
  are not FLOPs, physical DRAM traffic or an energy measurement. Python object
  and dispatch overhead remains included in the complete-call timing.
- Source hashes, baseline commit, seeds, input hashes, shapes and environment.

Any local timing ratio is restricted to this interpreted NumPy implementation.
It is not evidence of a faster language model. The implementation can skip
planes yet still lose time to its certificate, especially on small matrices or
when bounds are loose. Energy is not measured and is explicitly marked false.

The tests include all 729 code layouts of a 3-plane, 2-row, 1-column projection
under three hidden vectors, randomized all-prefix enclosures, exact ties,
reversal by later corrections, negative scales, ULP boundaries, subnormals,
cancellation, finite references with overflowing coarse bounds, and rejection
of actual nonfinite reference arithmetic.

## What would be needed for real-model validation

1. Supply hidden states and output weights from a pinned, licensed checkpoint,
   with held-out prompts distinct from any fitting or recovery data.
2. Evaluate complete-representation quality against the source checkpoint
   separately from certificate correctness. Matching a weak compressed
   reference does not recover its lost quality.
3. Compare to a strong optimized fixed-precision output kernel on the same
   hardware. Include all metadata, intermediate allocations, cache behavior,
   batch sizes, launch overhead and any full fallback.
4. Derive and verify arithmetic bounds for the actual deployed kernel before
   treating a NumPy certificate as portable.
5. For sampling or sequence-level statistical guarantees, define a different
   object of control and evaluate the complete deployed policy. Those features
   are not implemented here.

## Prior work and research position

Progressive representations and conditional computation already have substantial
prior work. Relevant primary sources include:

- [PTQTP](https://arxiv.org/abs/2509.16989): ternary trit-plane decomposition.
- [Matryoshka Quantization](https://proceedings.mlr.press/v267/nair25a.html) and
  [MatGPTQ](https://arxiv.org/abs/2602.03537): nested quantized representations.
- [BitStack](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1842fcfd74f3c2c2a7693994a4a7c37-Abstract-Conference.html):
  incremental residual blocks and variable-memory deployment.
- [DecDEC](https://arxiv.org/abs/2412.20185): dynamic residual compensation.
- [CALM, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html):
  adaptive computation with calibrated sequence-level consistency.
- [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html):
  statistical control of expected monotone losses. This prototype's
  deterministic interval check is not conformal risk control.
- [Runtime-Certified Bounded-Error Quantized Attention](https://arxiv.org/abs/2605.20868):
  local attention bounds and precision fallback.
- [Auditing Conformal Prediction under Inference-Time Model Shift](https://arxiv.org/abs/2610.28937):
  why calibration cannot silently transfer between inference configurations.

The open question is whether a more useful representation and tighter cheap
bounds can make certified output refinement worthwhile on real hardware. This
directory establishes a correctness-oriented baseline for that question.
