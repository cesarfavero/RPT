# RPT Technical Execution Checklist

This file is the working source of truth for next technical steps.

It serves two purposes:

1. keep product, research, runtime, website, and infrastructure work aligned;
2. make it explicit what is already validated in the repository versus what still needs execution.

Status rule:

- `[x]` done and evidenced in-repo
- `[ ]` not done yet or not yet evidenced well enough

## 1. Core Model and Runtime Validation

- [x] Baseline reference evaluation for the current Gemma track exists.
  Evidence: `gemma4_theory/notebooks/01_baseline_eval_gemma4.ipynb`, `gemma4_theory/results/01_baseline_eval_20260420_060326.json`
- [x] Improved dense checkpoint beats the original local Gemma reference.
  Evidence: `gemma4_theory/notebooks/03_pruned_checkpoint_eval_gemma4.ipynb`, `gemma4_theory/results/reference_eval/16_reference_gemma2b_eval_summary.json`, `gemma4_theory/README.md`
- [x] Quantization comparison for the pruned checkpoint was measured.
  Evidence: `gemma4_theory/notebooks/04_quantization_gemma4.ipynb`, `gemma4_theory/results/04_quantization_eval_20260420_073000.json`
- [x] Direct ternary conversion failure mode was documented before recovery.
  Evidence: `gemma4_theory/notebooks/05_ternary_qat_ste_gemma4.ipynb`, `gemma4_theory/notebooks/06_ternary_recovery_gemma4.ipynb`, `KNOWN_ISSUES.md`
- [x] Progressive ternary MLP expansion up to the full `78` projections was validated.
  Evidence: `gemma4_theory/notebooks/07_progressive_ternary_expansion_gemma4.ipynb`, `gemma4_theory/results/07_COM-TARGET[78]/07_ternary_expansion_summary_20260421_133727.json`
- [x] Packed ternary export exists as a real artifact layer.
  Evidence: `gemma4_theory/notebooks/10_export_packed_ternary_mlp.ipynb`, `gemma4_theory/deploy/README.md`, `gemma4_theory/deploy/manifest.json`
- [x] Packed-path recovery was validated on the local 100-question suite.
  Evidence: `gemma4_theory/notebooks/15_recover_packed_ternary_path.ipynb`, `gemma4_theory/notebooks/17_t4_resume_packed_recovery.ipynb`, `gemma4_theory/notebooks/18_test_packed_recovered_l78.ipynb`
- [x] Native structural runtime exists and was validated.
  Evidence: `gemma4_theory/notebooks/22_native_ternary_mlp_runtime.ipynb`, `gemma4_theory/results/packed_recovery/22_native_ternary_mlp_runtime_20260422_232225.json`
- [x] Triton runtime integration exists and preserves packed recovered behavior.
  Evidence: `gemma4_theory/notebooks/24_integrate_triton_ternary_runtime.ipynb`, `gemma4_theory/results/packed_recovery/24_integrate_triton_ternary_runtime_20260424_192203.json`
- [x] Runtime benchmark and latency-tail diagnosis were done.
  Evidence: `gemma4_theory/notebooks/25_benchmark_triton_runtime.ipynb`, `gemma4_theory/notebooks/26_diagnose_triton_latency_tail.ipynb`, `gemma4_theory/results/packed_recovery/runtime_benchmarks/`
- [x] Production-style Triton bundle was built and verified.
  Evidence: `gemma4_theory/notebooks/27_build_production_triton_bundle.ipynb`, `gemma4_theory/results/packed_recovery/production_bundle/27_build_production_triton_bundle_20260424_212734.json`, `gemma4_theory/runtime/packed_recovered_l78/README.md`

## 2. Next Model Optimization Work

Execution roadmap: `RPT_9_MODEL_OPTIMIZATION_ROADMAP.md`


- [ ] Define the official hardware target matrix for the product.
  Deliverable: explicit target profiles such as `CPU 8 GB`, `consumer notebook`, `Apple Silicon`, `Linux NVIDIA`.
- [ ] Run layer-wise ternary ablation to identify what must stay at higher precision.
  Goal: separate `embeddings`, `attention`, `lm_head`, and critical MLP blocks instead of treating ternary as one monolithic switch.
- [ ] Quantify real memory split between weights, KV cache, activations, and runtime overhead.
  Goal: avoid optimizing only checkpoint size while missing inference bottlenecks.
- [ ] Evaluate KV-cache reduction options.
  Candidates: quantized KV cache, shorter default context, sliding window, summary compaction.
- [ ] Decide whether the next gain comes from more recovery/distillation or from runtime engineering.
  Gate: compare quality delta per unit of engineering effort on the current benchmark plus one external suite.
- [ ] Broaden external validation for the final packed runtime path, not only the HF/adapter path.
  Reason: external-eval notes already state that functional quality does not yet prove packed-runtime efficiency claims.

## 3. Productization and Distribution

- [x] Closed-source commercialization direction is documented.
  Evidence: `COMMERCIALIZATION.md`, `LICENSE`, `NOTICE`
- [x] Product positioning already distinguishes Gemma-derived artifacts from fully owned base models.
  Evidence: `COMMERCIALIZATION.md`, `README.pt-BR.md`, `gemma4_theory/README.md`
- [ ] Finalize repository boundaries.
  Goal: keep `RPT` as private research source of truth and define what, if anything, remains public-facing.
- [ ] Define the shipping surface.
  Options to settle: compiled CLI, packaged runtime bundle, optional thin SDK, update channel, license enforcement.
- [ ] Create a release manifest policy.
  Minimum: version, supported hardware, benchmark snapshot, known limits, third-party notices, SHA256.
- [ ] Decide how model downloads are delivered.
  Options: bundled artifact, on-demand download, authenticated pull, or hybrid.
- [ ] Add explicit user-facing modes.
  Suggested modes: `ultra-light`, `balanced`, `quality`.

## 4. Site and Public Communication

- [ ] Update the public website to match the current claim set.
  Required message: closed-source product, Gemma-derived optimized runtime, supported environments, honest benchmark scope.
- [ ] Remove claims that imply universal portability if the runtime is still Linux/NVIDIA/Colab-centric.
- [ ] Publish a minimal product page with supported hardware, install path, and known limits.
- [ ] Publish benchmark summaries without exposing private pipeline details.
- [ ] Audit every public README and page for wording that leaks internal method or overstates ownership.

## 5. Security, Privacy, and Repo Hygiene

- [ ] Make sure core repositories are private where intended.
- [ ] Audit the current repo for secrets, tokens, private URLs, service credentials, or leaked internal paths.
- [ ] Audit generated artifacts before distribution.
  Check: bundled configs, logs, notebook outputs, manifests, cached credentials, machine metadata.
- [ ] Review notebook outputs for accidental environment leakage.
  Risk: some result files already store detailed environment info that may not belong in public artifacts.
- [ ] Separate distributable artifacts from research traces.
  Goal: the release bundle should not carry extra experiment residue.
- [ ] Add a pre-release scrub checklist.
  Minimum: secret scan, notice validation, artifact diff, benchmark snapshot, legal wording review.

## 6. VPS and Runtime Hardening

- [ ] Inventory every public-facing service on the VPS.
  Check: open ports, SSH exposure, web apps, notebooks, model endpoints, file servers.
- [ ] Remove or disable anything not required for shipping or internal ops.
- [ ] Rotate credentials that may have touched shared notebooks, CI, or deployment hosts.
- [ ] Enforce SSH hardening.
  Minimum: key-only auth, no password auth, limited users, fail2ban or equivalent, audited sudo access.
- [ ] Put the inference service behind the smallest possible exposed surface.
  Options: reverse proxy, VPN-only access, allowlist, internal-only worker.
- [ ] Add system patching and dependency patch cadence for the VPS.
- [ ] Add basic observability for security and runtime failures.
  Minimum: auth logs, process restart logs, disk usage, GPU usage, service health.
- [ ] Write an incident checklist for "what if something leaked".
  Minimum: revoke, rotate, invalidate, diff artifacts, verify access logs.

## 7. Documentation and Operability

- [x] Research and runtime documentation already describe the validated path.
  Evidence: `README.pt-BR.md`, `gemma4_theory/README.md`, `gemma4_theory/deploy/README.md`, `gemma4_theory/runtime/packed_recovered_l78/README.md`
- [x] Reproducibility and known-issue notes already exist.
  Evidence: `REPRODUCIBILITY.md`, `KNOWN_ISSUES.md`
- [ ] Add one operator-facing runbook for release assembly.
  Goal: take the validated notebook path and turn it into a short deterministic packaging checklist.
- [ ] Add one operator-facing runbook for infra bring-up.
  Goal: fresh VPS to secured runtime service without relying on memory.
- [ ] Add one operator-facing runbook for rollback.
  Goal: revert runtime release, model artifact, or site claim safely.

## 8. Linear Operating Model

- [ ] Create one Linear project for productization and hardening.
  Suggested name: `RPT Productization`
- [ ] Create one Linear document mirroring this checklist.
- [ ] Create issues for each execution stream:
  Streams: `model optimization`, `runtime portability`, `site update`, `repo/privacy`, `VPS hardening`, `release process`
- [ ] Use the repo file as the detailed source of truth and Linear as the execution board.
  Rule: long-form evidence stays in Git; task ownership and status live in Linear.

## 9. Immediate Recommended Order

- [ ] 1. Freeze the product boundary.
  Decide what is private, what is downloadable, and what the supported hardware claim is.
- [ ] 2. Audit repo and VPS for leaks before expanding distribution.
- [ ] 3. Update public site and docs so the external story matches the actual supported runtime.
- [ ] 4. Create the Linear project/doc/issues so execution stops living only in chat.
- [ ] 5. Resume technical optimization on the next bottleneck after the audit.
  Preferred starting points: layer-wise ternary ablation, KV-cache plan, runtime portability matrix.
