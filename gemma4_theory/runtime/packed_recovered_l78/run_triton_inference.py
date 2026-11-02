from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

try:
    from .triton_loader import load_triton_packed_recovered_l78, runtime_execution_report, reset_runtime_execution
    from .artifacts import read_json
    from .metrics import summarize_generation
except ImportError:
    from triton_loader import load_triton_packed_recovered_l78, runtime_execution_report, reset_runtime_execution
    from artifacts import read_json
    from metrics import summarize_generation


PROMPT_TEMPLATE = "Responda de forma curta e direta.\nPergunta: {prompt}\nResposta:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Triton packed ternary RPT L78 inference.")
    parser.add_argument("--model-dir", required=True, help="Diretorio distribuivel do modelo.")
    parser.add_argument("--prompt", required=True, help="Pergunta ou prompt de entrada.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--raw-prompt", action="store_true", help="Nao aplicar template pt-BR de pergunta curta.")
    parser.add_argument("--no-verify-sha", action="store_true", help="Nao validar SHA do packed bin.")
    parser.add_argument("--json", action="store_true", help="Imprimir resposta estruturada em JSON.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow PyTorch fallback; cannot certify Triton execution.")
    parser.add_argument("--require-full-checksums", action="store_true", help="Require checksums for all bundle files.")
    parser.add_argument("--prewarm", action="store_true", help="Execute the bundled warmup prompts in this process.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the same request; report first and subsequent calls separately.")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.repeat < 1:
        parser.error("--max-new-tokens and --repeat must be positive")
    if args.no_verify_sha and not args.allow_fallback:
        parser.error("Strict Triton runs require SHA256 verification")
    return args


def synchronize_model_devices(model):
    devices = {item.device for item in list(model.parameters()) + list(model.buffers())
               if item.device.type == 'cuda'}
    for device in sorted(devices, key=str):
        torch.cuda.synchronize(device)


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    synchronize_model_devices(model)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    synchronize_model_devices(model)
    elapsed = time.perf_counter() - started
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated_tokens = max(0, int(output.shape[-1]) - prompt_tokens)
    return {"answer": tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip(),
            "generation_elapsed_sec": elapsed, "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens, "tokens_per_sec": generated_tokens / elapsed}


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    started_load = time.perf_counter()
    model, tokenizer, report = load_triton_packed_recovered_l78(
        args.model_dir,
        token=token,
        device_map=args.device_map,
        dtype=args.dtype,
        verify_sha=not args.no_verify_sha,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        require_triton=not args.allow_fallback,
        fallback_to_pytorch=args.allow_fallback,
        require_full_checksums=args.require_full_checksums,
    )
    load_elapsed = time.perf_counter() - started_load

    warmup_rows = []
    if args.prewarm:
        profile = read_json(Path(args.model_dir).expanduser() / "prewarm_profile.json")
        prompts = list(dict.fromkeys(row.get("question", row.get("prompt"))
                       for row in profile.get("prewarm_rows", []) if isinstance(row, dict)))
        if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("Prewarm profile must contain prewarm_rows with question/prompt strings")
        for question in prompts:
            warmup_rows.append(generate_once(model, tokenizer, PROMPT_TEMPLATE.format(prompt=question), args.max_new_tokens))
        runtime_execution_report(model, require_triton=not args.allow_fallback)
    reset_runtime_execution(model)
    prompt = args.prompt if args.raw_prompt else PROMPT_TEMPLATE.format(prompt=args.prompt)
    rows = []
    for index in range(args.repeat):
        row = generate_once(model, tokenizer, prompt, args.max_new_tokens)
        row["phase"] = "after_explicit_prewarm" if args.prewarm else "first_request" if index == 0 else "subsequent_request"
        row["repeat"] = index + 1
        rows.append(row)
    report.update(runtime_execution_report(model, require_triton=not args.allow_fallback))
    answer = rows[-1]["answer"]
    payload = {
        **rows[-1],
        "load_elapsed_sec": load_elapsed,
        "loader_report": report,
        "prewarm_executed_here": bool(warmup_rows),
        "prewarm_summary": summarize_generation(warmup_rows),
        "generation_rows": rows,
        "generation_summary": summarize_generation(rows),
        "phase_summaries": {phase: summarize_generation([row for row in rows if row["phase"] == phase])
                            for phase in sorted({row["phase"] for row in rows})},
        "cold_cache_guaranteed": False,
        "prompt_mode": "raw" if args.raw_prompt else "historical_short_answer_template",
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print()
        print(json.dumps({k: v for k, v in payload.items() if k != "answer"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

