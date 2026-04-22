# Packed Recovery L78

Este diretório registra o fechamento do recovery T4 para o caminho packed/materialized ternary MLP L78.

## Resultado oficial

- notebook: `17_t4_resume_packed_recovery`
- status: `target_met`
- fase final: `phase_1200`
- steps concluídos: `1200`
- acurácia final: `1.00` (`100/100`)
- target mínimo: `0.90`
- referência limpa `student_pruned + l78`: `0.92`
- packed base sem adapter: `0.01`
- adapter antigo `l78` no packed: `0.63`
- validação sem retreino (`18_test_packed_recovered_l78`): `1.00` (`100/100`)

## Medição operacional

- notebook: `20_measure_and_build_packed_recovered_l78`
- arquivo: `20_measure_and_build_packed_recovered_l78_20260422_211727.json`
- ambiente: Colab/T4 com `14.56 GB` VRAM
- tensores packed aplicados: `78`
- SHA do packed bin: `0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116`
- tamanho bruto:
  - `student_pruned`: `4.90 GB`
  - adapter recovered L78: `1016.65 MB`
  - packed bin: `394.88 MB`
  - manifest: `29.05 KB`
- tempo de carga:
  - tokenizer: `7.44s`
  - base `student_pruned`: `103.56s`
  - materialização packed MLP: `26.93s`
  - adapter recovered: `14.44s`
  - total aproximado: `152.38s`
- VRAM:
  - após base: `4.87 GB`
  - após adapter: `5.10 GB`
  - pico: `5.33 GB`
- geração smoke test: `5` prompts, `13` tokens gerados, média `7.75 tokens/s`

Leitura: o artefato é utilizável em T4, mas ainda não é um modelo autônomo nem um runtime ternário real. O `student_pruned` continua necessário como base densa; o `packed_bin` troca somente os MLPs, e o adapter recovered atua como correção funcional sobre essa composição.

## Distribuição executável

O runtime versionado fica em `../../runtime/packed_recovered_l78/` e o pacote distribuível deve ser montado com `21_build_distributable_packed_recovered_l78.ipynb`.

Layout esperado:

```text
build/distributable_packed_recovered_l78/
  student_pruned/
  adapter_recovered_l78/
  packed/
  runtime/
  README.md
  package_manifest.json
```

Comando de inferência no pacote:

```bash
python runtime/run_inference.py --model-dir . --prompt "Qual a capital da França?"
```

Este formato resolve a experiência de execução para terceiros, mas ainda não resolve a leveza plena. A leveza real exige remover MLPs densos duplicados ou executar os MLPs packed diretamente.

## Runtime nativo estrutural

O protótipo nativo estrutural fica em:

```text
../../runtime/packed_recovered_l78/native_loader.py
../../runtime/packed_recovered_l78/run_native_inference.py
../../notebooks/22_native_ternary_mlp_runtime.ipynb
```

Ele substitui os 78 MLPs por `PackedTernaryLinear`, mantendo os bytes 2-bit como buffer persistente. O objetivo é validar que o modelo + adapter recovered continuam funcionais sem manter os MLPs como parâmetros densos persistentes.

Primeiro smoke test reportado:

- load: `102.06s`
- `replaced_count`: `78`
- `runtime_kind`: `native_structural_prototype`
- VRAM alocada após load: `2.40 GB`
- VRAM reservada após load: `5.28 GB`
- pico CUDA: `5.33 GB`
- smoke test: `5` prompts coerentes
- tokens gerados: `15`
- tempo total de geração: `7.12s`
- média: `2.22 tokens/s`
- avaliação 100 perguntas: ainda não rodada

Validação completa posterior:

- arquivo: `22_native_ternary_mlp_runtime_20260422_232225.json`
- `replaced_count`: `78`
- `runtime_kind`: `native_structural_prototype`
- `is_native_ternary_runtime`: `true`
- `is_optimized_kernel`: `false`
- load: `115.06s`
- VRAM alocada após load: `2.40 GB`
- VRAM reservada após load: `5.28 GB`
- pico CUDA: `5.33 GB`
- smoke test: `5` prompts coerentes
- tokens gerados no smoke: `15`
- média do smoke: `2.29 tokens/s`
- avaliação 100 perguntas: `100/100` (`1.00`)
- falhas: `0`

Comparação com o notebook 20:

- materialized hybrid após adapter: `5.10 GB` alocados
- native structural após load: `2.40 GB` alocados
- ganho observado em memória persistente alocada: aproximadamente `2.70 GB`
- trade-off: geração caiu de `~7.75 tokens/s` para `~2.29 tokens/s`, esperado porque o protótipo ainda desempacota pesos no `forward`

Comando esperado no pacote distribuível:

```bash
python runtime/run_native_inference.py --model-dir . --prompt "Qual a capital da França?"
```

Limitação: ainda não é kernel otimizado. O protótipo desempacota no `forward`, então é lento. Como preservou `100/100`, o próximo marco é CUDA/Triton para matmul ternária.

O notebook `22_native_ternary_mlp_runtime.ipynb` roda a avaliação das 100 perguntas por padrão. Para depuração rápida, definir `RPT_NATIVE_RUN_FULL_100=0` antes da célula de avaliação.
O avaliador usa `expected_answer` e `accepted_answers`, que é o formato real de `data/gemma4_100_questions.json`.

Se o ambiente falhar com `Found an incompatible version of torchao`, atualizar para `torchao>=0.16.0`. O bootstrap do notebook 22 foi ajustado para corrigir automaticamente `torchao 0.10.0` quando detectado.

## Spike Triton

O próximo notebook é `23_triton_ternary_matmul_spike.ipynb`.

Escopo:

- validar matmul ternária packed fora do modelo completo
- testar shapes reais:
  - `gate/up`: `[9216, 2304]`
  - `down`: `[2304, 9216]`
- comparar:
  - `F.linear` com peso reconstruído
  - unpack PyTorch a cada forward
  - kernel Triton lendo packed

Critério para avançar ao notebook 24: erro numérico aceitável e ganho real contra o caminho `unpack_then_linear`.

Resultado do spike:

- arquivo: `23_triton_ternary_matmul_spike_20260422_234112.json`
- ambiente: Tesla T4, PyTorch `2.10.0+cu128`, Triton `3.6.0`
- `success`: `true`
- casos: `4`
- erro máximo: `0.00137`
- speedup médio contra `unpack_then_linear`: `187.43x`
- speedup médio contra `F.linear` denso materializado: `0.109x`

Interpretação: o kernel Triton é correto o suficiente para o spike e elimina o gargalo absurdo de reconstruir o peso em PyTorch a cada forward. Porém ele ainda é mais lento que o caminho dense materializado. O próximo passo é integrar no runtime nativo (`24`) para medir o efeito real em tokens/s, VRAM e `100/100`.

## Runtime Triton integrado

Arquivos:

```text
../../runtime/packed_recovered_l78/triton_loader.py
../../runtime/packed_recovered_l78/run_triton_inference.py
../../notebooks/24_integrate_triton_ternary_runtime.ipynb
```

O notebook 24 integra o kernel Triton no modelo completo e roda a avaliação das 100 perguntas por padrão. Ele deve ser comparado contra:

- notebook 20: materialized hybrid, `~7.75 tokens/s`, `~5.10 GB` alocados após load
- notebook 22: native structural, `~2.29 tokens/s`, `~2.40 GB` alocados após load
- notebook 23: kernel isolado, `187.43x` contra unpack PyTorch, mas `0.109x` contra dense

Critério: preservar `100/100`, manter VRAM baixa e superar claramente o notebook 22 em tokens/s.

## Importância

Este run mostra que o artefato packed/materialized (`mlp_ternary_2bit.bin`) pode ser recuperado funcionalmente. O packed puro praticamente colapsa no benchmark local, mas o adapter treinado especificamente para esse caminho recupera o desempenho e supera a referência local de `0.92` nessa bateria.

A validação `18_test_packed_recovered_l78_20260422_205445.json` reproduziu o `100/100` sem retreinar, aplicando novamente os `78` tensores MLP packed e o adapter recuperado. Isso torna o marco mais forte: o resultado não é apenas o log final do treinamento, mas também um teste de carregamento/reprodução do artefato final.

## Ordem obrigatória de carregamento

```text
load student_pruned -> materialize mlp_ternary_2bit.bin using manifest.json -> apply this adapter with PeftModel.from_pretrained
```

## Artefatos esperados no Drive

```text
adapter:
/content/drive/MyDrive/RPT/gemma4_theory/checkpoints/student_packed_ternary_recovered_l78_t4_resume

manifest:
/content/drive/MyDrive/RPT/gemma4_theory/exports/ternary_mlp_l78/manifest.json

packed bin:
/content/drive/MyDrive/RPT/gemma4_theory/exports/ternary_mlp_l78/mlp_ternary_2bit.bin
```

## Arquivos principais

- `17_t4_resume_report.json`: resumo oficial do run e comparações.
- `17_t4_resume_progress.json`: estado consolidado resumable.
- `17_t4_resume_phase_1200_20260422_203242.json`: avaliação detalhada da fase final.
- `packed_base_required.json`: cópia do estado/base obrigatório para reprodução.
- `18_test_packed_recovered_l78_20260422_205445.json`: reprodução pós-marco sem retreino.
- `20_measure_and_build_packed_recovered_l78_20260422_211727.json`: medição operacional de carga, tamanho, VRAM e throughput.
- `../../build/packed_recovered_l78/`: loader standalone e manifesto operacional gerados pelo notebook 20.
- `../../runtime/packed_recovered_l78/`: runtime CLI versionado para distribuição.
