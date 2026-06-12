# go.rpt.dog installer hardening

This folder contains the current public installer response for:

- `https://go.rpt.dog/install.sh`

Current policy:

- public install is disabled until the runtime is distributed as a compiled artifact
- the current Python/Triton bundle must not be distributed from the public installer endpoint because it exposes source-visible runtime files

## Publish checklist

- upload `install.sh` to the origin behind `go.rpt.dog`
- verify `curl -fsSL https://go.rpt.dog/install.sh | sed -n '1,220p'` matches this file
- verify no bundle, manifest, `.git`, or directory listing is publicly reachable
- verify release bundles do not include raw `results/`, `sessions/`, or environment-heavy notebook outputs
- verify the public installer stays disabled until a compiled distribution exists

## Why this is blocked

- the current bundle layout explicitly includes `runtime/*.py`
- the documented entrypoint is `python runtime/run_triton_inference.py`
- that means the public installer is distributing source-visible execution logic, not a compiled product surface

## Re-enable condition

- runtime distribution is compiled or otherwise source-hidden
- public install flow no longer exposes private implementation files
- release packaging has a scrubbed, intentional artifact boundary
