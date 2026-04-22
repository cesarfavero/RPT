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
