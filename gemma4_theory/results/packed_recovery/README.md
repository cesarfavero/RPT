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
