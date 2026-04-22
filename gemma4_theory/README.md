# Gemma 4 Theory Track

Esta pasta separa a nova trilha experimental para validar a teoria incrementalmente em um modelo denso moderno.

## Estrutura

- `data/gemma4_100_questions.json`: 100 perguntas com gabarito e categorias.
- `notebooks/01_baseline_eval_gemma4.ipynb`: baseline do modelo e métrica inicial.
- `notebooks/02_progressive_pruning_gemma4.ipynb`: pruning progressivo em modelo denso.
- `notebooks/03_recovery_distillation_gemma4.ipynb`: recuperação guiada pelo teacher original.
- `notebooks/04_quantization_gemma4.ipynb`: PTQ/QAT intermediário.
- `notebooks/05_ternary_qat_ste_gemma4.ipynb`: tentativa de ternarização estilo BitNet.
- `notebooks/06_ternary_recovery_gemma4.ipynb`: recovery leve após snap ternário parcial.
- `notebooks/07_progressive_ternary_expansion_gemma4.ipynb`: expansão curricular do piloto ternário com recovery por estágios.
- `notebooks/08_consolidate_mlp_ternary_results.ipynb`: consolidação dos resultados da fase MLP ternária.
- `notebooks/09_ternary_deployment_audit.ipynb`: auditoria de deploy, estimativa de packing ternário e próximos passos de runtime.
- `notebooks/10_export_packed_ternary_mlp.ipynb`: exporta os MLPs ternários em artefato 2-bit empacotado e testa reconstrução.
- `notebooks/11_validate_hybrid_ternary_package.ipynb`: valida o pacote híbrido exportado e o adapter `l78`.
- `notebooks/12_external_eval_l78_hybrid.ipynb`: prepara avaliação externa com `lm-evaluation-harness`.
- `notebooks/13_external_eval_scaled_l78.ipynb`: avaliação externa ampliada do adapter `l78`.
- `notebooks/14_validate_packed_ternary_reconstruction.ipynb`: valida reconstrução packed/materialized.
- `notebooks/15_recover_packed_ternary_path.ipynb`: recovery inicial do caminho packed/materialized.
- `notebooks/16_reference_gemma2b_question_eval.ipynb`: avaliação de referência HF original e `student_pruned`.
- `notebooks/17_t4_resume_packed_recovery.ipynb`: recovery T4 retomável do caminho packed/materialized.
- `notebooks/18_test_packed_recovered_l78.ipynb`: validação pós-marco do checkpoint packed recuperado, sem retreino.
- `notebooks/20_measure_and_build_packed_recovered_l78.ipynb`: mede custo operacional do checkpoint packed recuperado e gera pacote de build reproduzível.
- `notebooks/21_build_distributable_packed_recovered_l78.ipynb`: monta uma pasta distribuível executável do modelo packed recovered L78.
- `runtime/packed_recovered_l78/`: runtime CLI versionado para carregar e executar o pacote distribuível.
- `notebooks/06_result_analysis_gemma4.ipynb`: consolidação dos resultados que você vai me devolver.
- `utils/eval_utils.py`: utilitários compartilhados de avaliação e logging.
- `results/`: saídas JSON/CSV de cada execução.

## Fluxo correto

1. Rodar `01_baseline_eval_gemma4.ipynb`.
2. Me mandar os resultados salvos em `results/`.
3. Rodar `02_progressive_pruning_gemma4.ipynb` com cortes pequenos.
4. Se degradar, usar `03_recovery_distillation_gemma4.ipynb`.
5. Só depois entrar em `04` e `05`.
6. Validar `06_ternary_recovery_gemma4.ipynb`.
7. Escalar em `07_progressive_ternary_expansion_gemma4.ipynb`.
8. Consolidar a fase MLP em `08_consolidate_mlp_ternary_results.ipynb`.
9. Auditar deploy/packing em `09_ternary_deployment_audit.ipynb`.
10. Exportar artefato MLP ternário empacotado em `10_export_packed_ternary_mlp.ipynb`.
11. Validar pacote híbrido em `11_validate_hybrid_ternary_package.ipynb`.
12. Preparar benchmark externo em `12_external_eval_l78_hybrid.ipynb`.
13. Validar/recover o caminho packed/materialized em `14`, `15` e `17_t4_resume_packed_recovery.ipynb`.
14. Reproduzir o checkpoint packed recuperado sem retreino em `18_test_packed_recovered_l78.ipynb`.
15. Medir carga, tamanho, RAM/VRAM, throughput e gerar build operacional em `20_measure_and_build_packed_recovered_l78.ipynb`.
16. Montar pacote distribuível executável em `21_build_distributable_packed_recovered_l78.ipynb`.
17. Consolidar tudo em `06_result_analysis_gemma4.ipynb`.

## Estado atual da hipótese

- `01` baseline: `0.90`
- `02` pruning progressivo: até `~15.96%` de sparsity global sem degradar, chegando a `0.92`
- `03` checkpoint podado reconstruído: `0.92`
- `04` quantização:
  - `pruned_reference = 0.92`
  - `pruned_8bit = 0.92`
  - `pruned_4bit = 0.91`
- `05` ternary snap direto em 8 camadas MLP: `0.76`
- `06` ternary recovery após o snap parcial: `0.91`
- `07` expansão para 12 projeções MLP:
  - pré-recovery: `0.46`
  - pós-recovery: `0.92`
- `07` expansão para 16 projeções MLP:
  - pré-recovery: `0.16`
  - pós-recovery: `0.92`
- `07` expansão para 18 projeções MLP:
  - pré-recovery: `0.12`
  - pós-recovery: `0.92`
- `07` expansão para 21 projeções MLP:
  - pré-recovery: `0.02`
  - pós-recovery: `0.91`
- `07` expansão para 21 projeções MLP com recovery forte:
  - pré-recovery: `0.02`
  - pós-recovery: `0.92`
- `07` expansão para 24 projeções MLP com recovery forte:
  - pré-recovery: `0.05`
  - pós-recovery: `0.92`
- `07` expansão para 36 projeções MLP com recovery forte:
  - pré-recovery: `0.07`
  - pós-recovery: `0.91`
- `07` repetição de 36 projeções MLP com recovery mais forte:
  - pré-recovery: `0.07`
  - pós-recovery: `0.91`
- `07` expansão para 48 projeções MLP com recovery forte:
  - pré-recovery: `0.03`
  - pós-recovery: `0.91`

Leitura honesta:
- a teoria está forte para `pruning + 8bit`
- ternário direto sem adaptação falhou
- ternário com adaptação ficou sustentado até `24` projeções MLP no platô `0.92`, usando recovery forte
- em `36` projeções MLP, dois recoveries preservaram função em `0.91`, mas saíram do platô; antes de saltar para `48`, o próximo teste deve medir o intermediário `30`

## Estratégia de expansão

Não vamos varrer de `3 em 3` até o fim. Depois de validar `21` com recovery forte, a expansão passa a ser por saltos:

```text
24 -> 36 -> 48 -> 60 -> 78
```

Critério:
- se o estágio mantiver `0.92`, avançar para o próximo salto
- se cair para `0.91`, repetir o mesmo estágio com recovery mais forte antes de avançar
- se cair abaixo de `0.90`, testar o ponto intermediário entre o último estágio bom e o estágio que falhou
- `78` significa todos os MLPs ternarizados em um modelo com 26 blocos e 3 projeções MLP por bloco

Depois de `36` repetir `0.91` mesmo com recovery mais forte, existem duas rotas:
- mapear o limite fino entre `24` e `36` usando `30`/`33`
- ou avançar para o próximo regime com `48` usando recovery mais forte

Decisão operacional atual: testar `48` para observar se o regime colapsa, estabiliza em `0.91`, ou ainda consegue voltar a `0.92`.

Resultado do `48`: estabilizou em `0.91`, sem colapsar. Próximo passo operacional: repetir `48` com recovery mais longo antes de tentar `60`.

Decisão revisada: `0.91` é aceitável para a fase exploratória de expansão. A partir de `48`, o claim deixa de ser "sem perda contra o melhor checkpoint" e passa a ser "preservação alta com perda de 1 ponto". O próximo teste operacional é `60`, mantendo recovery forte.

Resultado do `60`: pré-recovery `0.00`, pós-recovery `0.92`. Isso mostra recuperação total contra o melhor checkpoint mesmo após colapso bruto completo do benchmark. Próximo teste operacional: `78`, todos os MLPs ternarizados.

Resultado do `78`: todos os MLPs ternarizados.
- pré-recovery: `0.01`
- pós-recovery: `0.92`

Conclusão inicial da fase MLP: concluída com sucesso no benchmark local. O próximo passo não deve ser ternarizar attention imediatamente; deve ser consolidar resultados, validar em benchmarks maiores e medir custo/tamanho/latência.

## Marco packed/materialized L78

O teste `17_t4_resume_packed_recovery` fecha a recuperação do caminho realmente empacotado/materializado (`mlp_ternary_2bit.bin`) para `78` projeções MLP ternarizadas.

Resultado oficial do run:
- fase final: `phase_1200`
- passos concluídos: `1200`
- acurácia final: `1.00` (`100/100`)
- target mínimo: `0.90`
- referência limpa `student_pruned + l78`: `0.92`
- packed base sem adapter: `0.01`
- adapter antigo `l78` aplicado no caminho packed: `0.63`
- delta contra adapter antigo no packed: `+0.37`
- delta contra referência limpa: `+0.08`

Validação pós-marco:
- notebook: `18_test_packed_recovered_l78`
- sem retreino: sim
- tensores packed MLP aplicados: `78`
- SHA do packed bin confirmado: `0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116`
- resultado reproduzido: `1.00` (`100/100`)
- arquivo: `results/packed_recovery/18_test_packed_recovered_l78_20260422_205445.json`

Medição operacional:
- notebook: `20_measure_and_build_packed_recovered_l78`
- ambiente medido: Colab/T4 (`14.56 GB` VRAM)
- tamanho bruto dos componentes:
  - `student_pruned`: `4.90 GB`
  - adapter recovered L78: `1016.65 MB`
  - `mlp_ternary_2bit.bin`: `394.88 MB`
  - `manifest.json`: `29.05 KB`
- carga medida:
  - tokenizer: `7.44s`
  - `student_pruned`: `103.56s`
  - materialização packed MLP: `26.93s`
  - adapter recovered: `14.44s`
  - total aproximado: `152.38s`
- VRAM medida:
  - após base: `4.87 GB`
  - após adapter: `5.10 GB`
  - pico: `5.33 GB`
- smoke test de geração: `5` prompts, `13` tokens gerados, média `7.75 tokens/s`
- arquivo: `results/packed_recovery/20_measure_and_build_packed_recovered_l78_20260422_211727.json`
- build operacional gerado: `build/packed_recovered_l78/`

Leitura operacional:
- o caminho atual é funcional e cabe em T4, mas ainda depende do `student_pruned` como base densa
- o `packed_bin` substitui somente os `78` MLPs; ele ainda não é um runtime ternário autônomo
- o adapter recovered corrige o comportamento sobre a base + MLPs packed, mas não substitui o modelo base
- o próximo marco de runtime deve remover duplicação dos MLPs densos ou criar execução nativa ternária, antes de afirmar redução real de memória/inferência

Distribuição executável:
- runtime versionado: `runtime/packed_recovered_l78/`
- notebook de empacotamento: `21_build_distributable_packed_recovered_l78`
- layout gerado: `build/distributable_packed_recovered_l78/`
- comando esperado após montar/baixar o pacote:

```bash
python runtime/run_inference.py --model-dir . --prompt "Qual a capital da França?"
```

Este é o formato "baixei e executei". Ele ainda é `materialized_hybrid`: executável e reproduzível, mas não deve ser vendido como runtime ternário nativo até existir execução packed sem materializar os MLPs em float.

Importância:
- prova que o caminho packed/materialized não é apenas estruturalmente reconstruível; ele também é recuperável funcionalmente com adapter específico
- separa o problema de runtime/packing do problema de capacidade: o packed puro colapsa (`0.01`), mas o recovery T4 devolve desempenho acima da referência local
- fixa a ordem reprodutível de carregamento: carregar `student_pruned`, materializar `mlp_ternary_2bit.bin` via `manifest.json`, depois aplicar o adapter `student_packed_ternary_recovered_l78_t4_resume` com `PeftModel.from_pretrained`
- estabelece o checkpoint packed recuperado como o melhor artefato local para validações seguintes, sujeito a confirmação em benchmarks maiores

Arquivos registrados:
- `results/packed_recovery/17_t4_resume_report.json`
- `results/packed_recovery/17_t4_resume_progress.json`
- `results/packed_recovery/17_t4_resume_phase_1200_20260422_203242.json`
- `results/packed_recovery/packed_base_required.json`
- `results/packed_recovery/18_test_packed_recovered_l78_20260422_205445.json`

Conclusão atualizada da fase MLP: concluída com sucesso no benchmark local e no caminho packed/materialized. O claim honesto é que o resultado atingiu e reproduziu `1.00` no benchmark local de 100 perguntas, acima da referência `0.92`; próximos passos exigem benchmark maior/externo, medição de custo/tamanho/latência e teste de robustez fora dessa bateria.

Próximos passos após o marco:
- criar uma bateria maior e inédita para medir generalização fora das 100 perguntas locais
- comparar em protocolo único: HF Gemma original, `student_pruned`, clean `l78`, old `l78` no packed e recovered packed `l78`
- executar `20_measure_and_build_packed_recovered_l78.ipynb` para medir tamanho, latência, VRAM/RAM, throughput e gerar um build operacional reproduzível
- executar `21_build_distributable_packed_recovered_l78.ipynb` para gerar o pacote baixável/executável
- só depois decidir entre avançar attention, runtime/export, ou uma fase de robustez do adapter packed

## Regra do projeto

Cada notebook salva um artefato em `results/` com timestamp. Não edite resultados manualmente; rode novamente.

## Regra de git

Sempre fechar cada etapa útil com `commit` e `push` para `origin`, para não perder estado experimental.
