# Reproducibility Guide

## Evidence policy

The historical 100-question suite is development/regression data. Recovery used
its expected answers and selected checkpoints on that suite. A 100/100 score is
not held-out evidence. Historical results stay immutable; regrading creates a
new result with the source hash and evaluator version.

The old External500 reports describe student_pruned plus an older adapter. The
new runner must evaluate the actual final packed artifact and record observed
execution. See [the evidence review](docs/reports/EVIDENCE_STATUS_2026-11-02.md).

## 1. Portable tests

Use a checkout fixed to a full Git commit. The reference test environment is
Python 3.11/3.12 and NumPy 2.3.5.

```bash
python -m pip install -r validation/requirements-cpu.txt
python -m validation.run_validation --phase cpu
python -m validation.environment --output environment.json
```

A successful CPU report certifies only the tests it lists. It does not evaluate
a language model, a GPU kernel or energy use. The runtime repository has an
additional CPU tensor suite; its CI installs a pinned CPU PyTorch build.

For notebooks, start from the checkout or set RPT_SOURCE_ROOT to its absolute
path. Notebooks no longer create their own embedded evaluator copies. Saved
notebook outputs remain historical and should not be interpreted as outputs of
the edited source.

## 2. Identify the exact model

The packed bundle has this layout:

```text
bundle/
  student_pruned/
  adapter_recovered_l78/
  packed/manifest.json
  packed/mlp_ternary_2bit.bin
```

The dense model directory and adapter need their actual configuration, tokenizer
and weight files; empty directories are not a complete bundle. The current
loader still loads the dense base before replacing the MLPs. A packed-first
loader and removal of the duplicated dense MLP checkpoint remain pending.

```bash
python -m validation.lm_eval_packed --inspect-artifact --model-dir /path/to/bundle
python -m validation.lm_eval_packed --inspect-artifact \
  --backend hf-dense --model-source /path/to/dense-model
```

Each inspection returns artifact.sha256 over sorted file paths, sizes and
SHA256 values. Keep the returned identities with the experiment declaration.
A remote dense model additionally requires --model-revision with a complete
40-character Hugging Face commit. Never replace it with an unpinned main revision.

The final packed file historically had SHA256
0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116.
This hash identifies that file only, not the adapter, tokenizer or whole bundle.

## 3. Check the accelerator

Keep the accelerator framework supplied by the hardware provider. See the
[SCNet guide](docs/SCNET_VALIDATION.md) for BW-1/DTK. Generic CUDA wheels are not
a substitute for the vendor stack.

```bash
python -m validation.run_validation --phase kernel --full-shapes
```

This probe executes the actual packed operator on synthetic matrices, compares
it with a dense reference and records first-call and warmed times. No silent
fallback is permitted. A broken driver or unavailable backend produces a
failure/blocker report, not a score.

The probe's first-call time may reuse an existing disk compilation cache. It is
not a guaranteed cold-cache result. Full-model generation must be measured
separately, with identical prompts, actual token counts and a matched baseline.

## 4. Freeze external data before scoring

The external runner requires lm-evaluation-harness and the relevant model stack
in the selected environment. Record their exact installed versions. A complete
GPU/DTK dependency lock has not yet been validated; do not describe the CPU
dependency file as a lock for inference.

For a short infrastructure smoke run:

```bash
python -m validation.lm_eval_packed --freeze-only \
  --tasks arc_easy,arc_challenge,hellaswag,truthfulqa_mc1 \
  --limit 20 \
  --known-training gemma4_theory/data/gemma4_100_questions.json \
  --split-manifest runs/external-smoke-splits.json
```

This creates a new immutable manifest and returns manifest_sha256. It records
the selected documents, source fingerprints, task configuration, prompt policy
and training exposure. The smoke subset is not the final scientific evaluation.
For the final campaign, use a new manifest without --limit.

By default the classification is external_validation. Only set
--training-data-complete after supplying all recovery data and documenting that
inventory. That flag is a declaration, not an automated proof. Even a held_out
classification concerns RPT recovery; exposure during foundation pretraining
remains unknown. The overlap check covers normalized exact prompts and IDs, not
all semantic paraphrases.

For custom train/development/test files:

```bash
python -m validation.splits freeze --dataset-id rpt-campaign \
  --train train.json --development dev.json --test test.json \
  --output runs/splits.json
```

Renaming an example or changing its ID does not make historical training data
held out.

## 5. Run the declared variants

Use the exact hashes returned above. Keep task, limit, seed, few-shot, maximum
context and chat-template settings identical to the frozen declaration.

```bash
python -m validation.lm_eval_packed \
  --model-dir /path/to/bundle --limit 20 \
  --split-manifest runs/external-smoke-splits.json \
  --expected-split-sha256 SPLIT_SHA256 \
  --expected-artifact-sha256 PACKED_BUNDLE_SHA256 \
  --output runs/packed-smoke.json

python -m validation.lm_eval_packed \
  --backend hf-dense --model-source /path/to/dense-model --limit 20 \
  --split-manifest runs/external-smoke-splits.json \
  --expected-split-sha256 SPLIT_SHA256 \
  --expected-artifact-sha256 DENSE_MODEL_SHA256 \
  --output runs/dense-smoke.json
```

packed-triton loads the verified object before passing it to HFLM, checks that
the harness retains that object, requires all 78 packed layers to execute in
Triton, and verifies the actual scored documents and artifact hashes again.
hf-dense is an explicitly requested control, never an automatic fallback.

Use --offline after all models, datasets and task dependencies are cached.
This sets Hugging Face's offline controls before loading their packages.

A failed identity, dataset, backend or numerical check produces a failure
record with no published score. Existing result files are never overwritten.

The dense control above is the current checkpoint comparison. The causal
compression experiment additionally needs a dense model trained with the same
recovery data, tokens, optimizer budget and seeds. That training has not been
executed by the validation code alone.

## 6. Runtime measurement contract

The runtime CLI supports explicit prewarm and repeated requests in the same
process:

```bash
python runtime/run_triton_inference.py --model-dir /path/to/bundle \
  --prompt "Explain why seasons occur." --max-new-tokens 128 \
  --prewarm --repeat 5 --json
```

The output distinguishes loading, prewarm, first request and subsequent requests.
Throughput is sum(actual output tokens) / sum(synchronized generation seconds).
Mean request-level rates remain separate. Whole-request time is not TTFT or TPOT;
those metrics stay unmeasured until a dedicated instrumented path exists.

The kernel reads packed codes but computes floating-point dot products. A
verified Triton execution does not imply zero-skipping, a fully integer ternary
operator or acceleration. Measure against a dense implementation and competitive
2/3/4-bit variants using the same hardware, contexts, output lengths and batches.
Report persistent storage, remaining dense weights, adapters, load peak, runtime
peak and KV cache separately. Energy is not inferred from time.

## 7. Reference research experiment

```bash
python -m research.adaptive_output.run_experiment
```

Read [its mathematical scope](research/adaptive_output/README.md) before using
the result. It certifies only the greedy argmax of its complete representation,
given the same hidden vector and documented float64 arithmetic. It is synthetic
and includes negative performance results. Learned correction policies,
integration into an LLM, GPU kernels and energy results are pending.

## Historical BitNet work

The BitNet notebooks are under notebooks/bitnet/. The historical log
sessions/2026-02-08/tracking.md records PPL 25.13 to 16.39 and 42.6% total zeros.
Those observations need matching-data and matching-training controls before
causal conclusions about pruning. The model already had zeros before pruning.
The exact deploy script cited in that log is not currently present in the
public source tree, which limits clean reproduction.

Hardware-to-hardware PPL differences require controlled diagnosis. The old
Lyapunov/amplification observations do not establish self-organized criticality,
and lower PPL after pruning does not test Landauer's heat-of-erasure bound.

## Remaining gates

[validation/protocol.json](validation/protocol.json) records the pending
campaign. Real SCNet/NVIDIA execution, complete model generalization, a
packed-first bundle, optimized kernels, same-budget recovery controls and a
second model family remain explicit gates. Do not mark them complete based on
portable tests or a synthetic experiment.

