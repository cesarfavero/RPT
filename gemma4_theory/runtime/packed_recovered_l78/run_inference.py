from __future__ import annotations

import argparse
import json
import os
import time

import torch

from loader import load_packed_recovered_l78


PROMPT_TEMPLATE = "Responda de forma curta e direta.\nPergunta: {prompt}\nResposta:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RPT packed recovered L78 inference.")
    parser.add_argument("--model-dir", required=True, help="Diretorio distribuivel do modelo.")
    parser.add_argument("--prompt", required=True, help="Pergunta ou prompt de entrada.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--raw-prompt", action="store_true", help="Nao aplicar template pt-BR de pergunta curta.")
    parser.add_argument("--no-verify-sha", action="store_true", help="Nao validar SHA do packed bin.")
    parser.add_argument("--json", action="store_true", help="Imprimir resposta estruturada em JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    started_load = time.perf_counter()
    model, tokenizer, report = load_packed_recovered_l78(
        args.model_dir,
        token=token,
        device_map=args.device_map,
        dtype=args.dtype,
        verify_sha=not args.no_verify_sha,
    )
    load_elapsed = time.perf_counter() - started_load

    prompt = args.prompt if args.raw_prompt else PROMPT_TEMPLATE.format(prompt=args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    started_generation = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generation_elapsed = time.perf_counter() - started_generation

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    total_tokens = int(output.shape[-1])
    generated_tokens = max(0, total_tokens - prompt_tokens)
    answer = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
    payload = {
        "answer": answer,
        "load_elapsed_sec": load_elapsed,
        "generation_elapsed_sec": generation_elapsed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / generation_elapsed if generation_elapsed > 0 else None,
        "loader_report": report,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(answer)
        print()
        print(json.dumps({k: v for k, v in payload.items() if k != "answer"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
