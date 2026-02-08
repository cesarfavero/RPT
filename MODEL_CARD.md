# Model Card: rpt-bitnet-2b-pruned

## Model Details

- Base model: `microsoft/bitnet-b1.58-2B-4T-bf16`
- Family: Ternary LLM (BitNet b1.58)
- Parameters: ~2.4B
- Weight domain: {-1, 0, +1} with group scales
- Quantization target: GGUF i2_s for bitnet.cpp runtime

## Training / Adaptation Summary

- Progressive magnitude pruning
- QAT/STE fine-tuning
- Ternary snap after fine-tune
- GGUF conversion for CPU inference

## Reported Metrics (WikiText-2)

- Baseline PPL (A100 setup): 25.13
- Final post-snap PPL: 16.39 (-34.8%)
- Final sparsity: 42.6%
- Ternary ratio: 100%
- CPU generation (Colab CPU): coherent text, ~0.26 tok/s (hardware dependent)

## Intended Use

- Research on efficient LLM inference with ternary + sparsity.
- Reproducibility studies on pruning and quantization-aware training.

## Out-of-Scope Use

- High-stakes production decisions without additional validation.
- Claims of broad capability improvement beyond tested benchmarks.

## Limitations

- Main quality claims are based on WikiText-2.
- Broader benchmark validation (MMLU, HellaSwag, ARC) is pending.
- Throughput values are runtime/hardware dependent.

## Artifacts

- Model: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned
- GGUF: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned-GGUF

## Safety / Ethics

This model can generate incorrect, biased, or repetitive text. Human review is required before downstream usage in sensitive domains.
