# Packed Recovered L78 Runtime

This runtime is the execution layer for the packed ternary Gemma bundle. It is part of our own runtime path for this project, not a Microsoft BitNet runtime wrapper.

It exists in three modes:

1. materialized hybrid runtime
2. native structural runtime
3. Triton runtime

The current recommended production path is the Triton runtime.

## Current Recommended Use

If you downloaded the production bundle zip, the normal entrypoint is:

```bash
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

This assumes you are inside the extracted bundle directory:

```text
production_triton_packed_recovered_l78/
```

## Expected Bundle Layout

```text
MODEL_DIR/
  student_pruned/
  adapter_recovered_l78/
  packed/
    manifest.json
    mlp_ternary_2bit.bin
  runtime/
    loader.py
    native_loader.py
    triton_loader.py
    run_inference.py
    run_native_inference.py
    run_triton_inference.py
    requirements.txt
  production_manifest.json
  prewarm_profile.json
  PRODUCTION_USAGE.txt
```

## Runtime Modes

### 1. Materialized Hybrid

Command:

```bash
python runtime/run_inference.py --model-dir /path/to/MODEL_DIR --prompt "Qual a capital da França?"
```

Behavior:

- loads `student_pruned`
- materializes packed ternary MLPs into dense weights
- applies the recovered adapter
- runs as a hybrid dense execution path

Use this when:

- you want the simplest fallback path;
- you do not need the Triton path.

### 2. Native Structural Prototype

Command:

```bash
python runtime/run_native_inference.py --model-dir /path/to/MODEL_DIR --prompt "Qual a capital da França?"
```

Behavior:

- replaces the MLP base layers with `PackedTernaryLinear`
- keeps packed tensors as persistent buffers
- still uses a PyTorch unpack path in `forward`

Use this when:

- you want to validate packed structural execution;
- performance is not the main goal.

### 3. Triton Runtime

Command:

```bash
python runtime/run_triton_inference.py --model-dir /path/to/MODEL_DIR --prompt "Qual a capital da França?"
```

Behavior:

- replaces the MLP base layers with `TritonPackedTernaryLinear`
- uses the Triton kernel for the packed MLP path
- is the current best runtime path in this project

Use this when:

- you want the current production-style path;
- you are on Linux/NVIDIA/Colab-compatible environment.

## What Was Validated

### Functional validation

The Triton runtime preserved the packed recovered model behavior on the local suite:

- `78` MLP base layers replaced
- `packed_sha256` validated
- local 100-question suite: `1.00`

### Runtime diagnosis

Initial benchmark:

- good `p50`
- very bad `p95`
- cause: cold compile / shape warmup, not steady-state inference weakness

After production prewarm:

- latency stabilized to roughly `0.35s` to `1.02s` in the production verification set
- average throughput around `5.66 tok/s`

That is why the production bundle includes `prewarm_profile.json`.

## Installation

Inside the extracted bundle root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
```

If your environment needs Hugging Face authentication for base model access:

```bash
export HF_TOKEN=your_token_here
```

## First Run

From the bundle root:

```bash
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

If you want a second quick check:

```bash
python runtime/run_triton_inference.py --model-dir . --prompt "Quanto é 12 x 8?"
```

## What The User Actually Downloaded

When you download the production zip, you are not downloading just a checkpoint.

You are downloading:

- dense base checkpoint
- recovered adapter
- packed ternary artifact
- runtime code
- production manifest
- prewarm profile

That is why the bundle is usable. It is the complete execution package for the supported environment.

## Limitations

Current limitations are still important:

- target environment is Linux/NVIDIA/Colab, not Apple Silicon or Ollama;
- external benchmark coverage for the final bundle is still narrower than the local benchmark story;
- some dependency versions may emit warnings (`peft`, `torchao`) even when execution succeeds;
- this is a production-style engineering bundle, not a universal runtime distribution yet.

## Honest Positioning

Correct positioning:

- usable ternary packed Gemma-derived runtime bundle for Linux/NVIDIA/Colab
- strong local benchmark and functional runtime validation
- real packed artifact plus real execution path
- a runtime path that already beats the original HF Gemma local score in this project benchmark when used with the recovered packed model

Incorrect positioning:

- universal consumer installer
- native M1/Ollama package
- fully hardware-agnostic runtime
