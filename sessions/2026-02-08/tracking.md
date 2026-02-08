# Sessao 8 Feb 2026

## Resumo
Deploy completo do pipeline RPT: QAT/STE fine-tune na VPS A100 + conversao GGUF no Colab CPU.
**PRIMEIRO DEPLOY FUNCIONAL** - modelo gera texto coerente em bitnet.cpp.

## O que foi feito

### 1. Bug #10 encontrado e corrigido (architecture name)
- llama.cpp tem `build_bitnet()` e `build_bitnet_158()` (grafos diferentes)
- Converter escrevia `"bitnet"` (errado), precisava `"bitnet-b1.58"` (correto)
- Fix: patch constants.py + pip force-reinstall
- Todos notebooks GGUF atualizados com o fix

### 2. Script deploy unificado (rpt_deploy_a100.py)
- Script .py unico que faz tudo: load → prune → QAT/STE → snap → save → GGUF
- 6 bugs encontrados e corrigidos pelo code reviewer antes de rodar
- Adaptado pra VPS (sem dependencias Colab)

### 3. Deploy na VPS A100 (Lightning AI)
- GPU: A100 (VRAM ~40 GB)
- Batch size: 8
- Tempo total GPU: ~7 min (2x 300 steps a 1.5 it/s)

**Resultados treino:**
```
Baseline PPL: 25.13
Texto baseline: "The capital of France is Paris. Paris is a city in the north of France..."

Nivel 5%:
  Pruned: 6.4%
  PPL antes FT: 24.93
  Loss: 3.541 → 3.030 (300 steps)
  PPL apos FT: 25.05 (-0.3% vs baseline)

Nivel 10%:
  Pruned: 15.2%
  PPL antes FT: 27.55
  Loss: 3.044 → 2.992 (300 steps)
  PPL apos FT: 33.07 (+31.6% vs baseline)

Snap ternario:
  0 pesos re-zerados (drift do FT)
  1,766,678,734 pesos ajustados
  Esparsidade final: 42.6%
  Ternario: 100.0%
  PPL final: 16.39 (-34.8% vs baseline!)
```

**Textos gerados (pos-snap, PyTorch):**
- "The capital of France is Paris. The city is known for its iconic landmarks, such as the Eiffel Tower and the Louvre."
- "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
- "The largest planet in the solar system is Jupiter. It is a gas giant planet that is about 318 Earths in size."

### 4. Modelo salvo e transferido
- `bitnet_2b_pruned/` (4.84 GB) salvo na VPS
- Comprimido: `bitnet_2b_pruned.tar.gz`
- Baixado via HTTP server (porta 8080)

### 5. Conversao GGUF no Colab CPU
- Notebook RPT_GGUF_CPU.ipynb atualizado pra pegar tar.gz do Drive
- Todos 10 fixes aplicados (architecture, weight_quant, vocab, etc)
- GGUF i2_s gerado (~1.1 GB)

**Teste inferencia (Colab CPU):**
```
Prompt: "The capital of France is"
Output: "Paris . There are also some cities that can be considered as their main cities,
such as the city that has been capital of France since the 17th century."

Velocidade: 0.26 tokens/s (CPU generico Colab)
```

### 6. Teste local Windows (FALHOU - esperado)
- llama-cpp-python nao suporta I2_S
- llama.cpp pre-compilado (b7972) nao suporta type 36 do BitNet fork
- I2_S precisa do binario do BitNet (fork especifico)

## Numeros finais
| Metrica | Valor |
|---------|-------|
| PPL baseline | 25.13 |
| PPL final (pos-snap) | **16.39 (-34.8%)** |
| Ternario | **100%** |
| Esparsidade | **42.6%** |
| Modelo HF | 4.84 GB |
| GGUF i2_s | ~1.1 GB |
| Texto GGUF | Coerente |

## Observacoes
- PPL melhorou 34.8% apos snap ternario - resultado incomum, pode ser favorecido pelo wikitext-2
- STE funcionou perfeitamente - modelo aprendeu a operar com restricao ternaria
- 42.6% esparsidade (pedimos 10% progressivo, mas modelo ja tinha ~33% zeros nativos)
- 0 pesos re-zerados = STE manteve mascaras de pruning intactas

## Proximo passo
- Benchmark mais rigoroso (MMLU, HellaSwag, ARC) pra confirmar melhoria
- Testar esparsidades mais altas (20%, 30%, 40%) com QAT/STE
- Publicar modelo no HuggingFace Hub
