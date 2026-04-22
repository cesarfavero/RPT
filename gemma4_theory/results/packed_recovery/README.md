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

## Importância

Este run mostra que o artefato packed/materialized (`mlp_ternary_2bit.bin`) pode ser recuperado funcionalmente. O packed puro praticamente colapsa no benchmark local, mas o adapter treinado especificamente para esse caminho recupera o desempenho e supera a referência local de `0.92` nessa bateria.

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
