# Known Issues

## 1) AdamW without STE breaks deploy quality

Symptom:
- PyTorch perplexity can improve, but GGUF output becomes incoherent.

Cause:
- Weights drift to continuous values during fine-tune and fail after ternary re-quantization.

Mitigation:
- Use QAT/STE so forward pass remains ternary-constrained during learning.

## 2) GGUF architecture name mismatch

Symptom:
- Model loads but generates incoherent text without explicit runtime error.

Cause:
- Converter metadata uses `bitnet` instead of `bitnet-b1.58`, selecting wrong runtime graph path.

Mitigation:
- Ensure architecture metadata is `bitnet-b1.58` in conversion path.

## 3) I2_S format support is not universal

Symptom:
- Some local `llama.cpp` builds cannot run I2_S models.

Cause:
- BitNet-specific support differs across forks/builds.

Mitigation:
- Use the BitNet-compatible runtime/toolchain used in the documented pipeline.

## 4) Baseline PPL differs across environments

Symptom:
- Different baseline values across T4/H100/A100.

Cause:
- Precision/runtime differences (e.g., TF32/compile settings).

Mitigation:
- Compare metrics only against baseline measured in the same environment.

## 5) Benchmark coverage is incomplete

Symptom:
- Strong WikiText-2 improvements may not transfer to broader tasks.

Mitigation:
- Validate on MMLU, HellaSwag, ARC, and additional benchmarks before broad claims.
