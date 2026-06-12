# RPT Security Audit

Date: `2026-04-27`

Scope covered in this pass:

- repository text scan for obvious secrets and credentials
- notebook and runtime scan for token handling
- artifact scan for environment leakage and machine metadata
- local repository structure review for release-surface risks

Scope not covered in this pass:

- live VPS inspection
- open-port validation
- host firewall / SSH / sudo policy review
- CI/CD secret store review outside this repository

## Executive Summary

No clear plaintext credential was found in the repository during this pass.

The main current risk is not an exposed token committed in source. The main current risk is overexposed operational metadata inside versioned result artifacts, especially environment snapshots that reveal:

- OS and kernel details
- GPU model and driver version
- CPU model and cache layout
- library versions
- hypervisor/vendor details
- hardware vulnerability status
- absolute library paths under `/usr/lib`

That is not the same as a leaked credential, but it is still unnecessary exposure for distributable or public-facing artifacts.

## Findings

### 1. No obvious committed secrets found in plaintext

Status: low immediate risk

What was checked:

- common API key patterns
- HF tokens
- GitHub tokens
- private key blocks
- connection strings
- generic secret/token markers in text files

Result:

- no direct plaintext token or private key block was found in the repository text scan

Important caveat:

- this does not prove secrets never existed in notebook outputs or binary artifacts;
- it only means the current text-visible pass did not find them.

### 2. `HF_TOKEN` is widely referenced in notebooks and runtime

Status: expected, but operationally sensitive

Files include:

- `gemma4_theory/notebooks/01_baseline_eval_gemma4.ipynb`
- `gemma4_theory/notebooks/02_progressive_pruning_gemma4.ipynb`
- `gemma4_theory/notebooks/03_recovery_distillation_gemma4.ipynb`
- `gemma4_theory/notebooks/06_ternary_recovery_gemma4.ipynb`
- `gemma4_theory/notebooks/14_validate_packed_ternary_reconstruction.ipynb`
- `gemma4_theory/notebooks/15_recover_packed_ternary_path.ipynb`
- `gemma4_theory/notebooks/17_t4_resume_packed_recovery.ipynb`
- `gemma4_theory/notebooks/18_test_packed_recovered_l78.ipynb`
- `gemma4_theory/notebooks/24_integrate_triton_ternary_runtime.ipynb`
- `gemma4_theory/notebooks/27_build_production_triton_bundle.ipynb`
- `gemma4_theory/runtime/packed_recovered_l78/run_inference.py`
- `gemma4_theory/runtime/packed_recovered_l78/run_native_inference.py`
- `gemma4_theory/runtime/packed_recovered_l78/run_triton_inference.py`

Interpretation:

- this is acceptable as long as the token stays in environment or Colab secrets;
- before distributing notebooks or bundles, verify that no executed output ever printed the token value.

### 3. Versioned result artifacts expose detailed machine and vulnerability metadata

Status: medium risk for public or customer-facing distribution

Examples:

- `gemma4_theory/results/external_eval/l78_adapter_fast_2026-04-21T16-24-01.236123.json`
- `gemma4_theory/results/external_eval/student_pruned_base_fast_2026-04-21T16-20-08.517165.json`
- `gemma4_theory/results/external_eval/student_pruned_base_full_2026-04-21T16-33-25.217198.json`
- `gemma4_theory/results/external_eval_scaled/l78_adapter_scaled500_2026-04-22T14-33-54.783777.json`

Observed fields:

- `pretty_env_info`
- `Nvidia driver version`
- `CUDA runtime version`
- `Hypervisor vendor`
- vulnerability lines such as `Spectre`, `Meltdown`, `Retbleed`, `Tsx async abort`

Interpretation:

- these files are useful for internal reproducibility;
- they should not be part of public release bundles or public benchmark packs without scrubbing.

### 4. Runtime bundle builder references public raw GitHub fallback

Status: medium operational risk

File:

- `gemma4_theory/notebooks/27_build_production_triton_bundle.ipynb`

Observed behavior:

- fallback path references `https://raw.githubusercontent.com/cesarfavero/RPT/main/gemma4_theory/runtime/packed_recovered_l78`

Interpretation:

- if the repository is private, this fallback may fail unexpectedly in some environments;
- if the repository becomes public temporarily or partially, this path may widen what gets fetched externally;
- release assembly should not depend on opportunistic runtime auto-download.

### 5. Repository includes many research traces that should not ride along with product release artifacts

Status: medium productization risk

Current shape:

- `98` JSON result files under `gemma4_theory/results`
- multiple notebook outputs with operational detail
- historical experiment traces under `sessions/`

Interpretation:

- internal repo is fine;
- distribution bundle needs a stricter separation between product artifact and research residue.

### 6. No VPS hardening evidence exists in this repository

Status: unknown / not yet validated

This pass found no host-level evidence for:

- SSH hardening
- firewall rules
- reverse proxy setup
- fail2ban
- system patch cadence
- service supervision
- auth or access logs

Interpretation:

- this is not proof the VPS is weak;
- it only means the VPS portion cannot be closed from repo evidence alone.

## Recommended Actions

### Immediate

- create a scrub rule for any distributable bundle: exclude raw `results/`, `sessions/`, notebook outputs, and environment dumps
- identify every artifact containing `pretty_env_info` and classify it as internal-only
- make release assembly independent from runtime auto-download from GitHub

### Short term

- add a release-scrub checklist before packaging
- create a clean export path for public benchmark summaries that strips environment metadata
- add a repo-level note distinguishing internal reproducibility artifacts from distributable artifacts

### VPS follow-up

- inspect open ports with `ss -tulpn` or equivalent
- inspect SSH config and auth mode
- inspect reverse proxy / exposed services
- inspect systemd services and restart policies
- inspect logs and credential locations
- rotate any token that may have lived in notebooks, Colab, or deployment shells if exposure is uncertain

## Closure Criteria For `RPT-6`

This audit issue should only be considered done when:

- obvious secrets scan is clean
- environment-heavy artifacts are classified and excluded from release bundles
- a repeatable pre-release scrub checklist exists
- release packaging is separated from research traces
- any uncertain credentials have been rotated or explicitly cleared

## Closure Criteria For `RPT-7`

The VPS hardening issue still requires host access. It should only be considered done when:

- exposed services are inventoried
- unnecessary services are removed
- SSH is key-only and restricted
- host patching state is reviewed
- runtime exposure is minimized
- basic security and health observability is in place
