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
8. Consolidar tudo em `06_result_analysis_gemma4.ipynb`.

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

Leitura honesta:
- a teoria está forte para `pruning + 8bit`
- ternário direto sem adaptação falhou
- ternário com adaptação ficou sustentado até `21` projeções MLP no platô `0.92`, desde que o recovery seja fortalecido
- o próximo teste deve expandir para `24` projeções MLP usando o recovery forte

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

## Regra do projeto

Cada notebook salva um artefato em `results/` com timestamp. Não edite resultados manualmente; rode novamente.

## Regra de git

Sempre fechar cada etapa útil com `commit` e `push` para `origin`, para não perder estado experimental.
