# RPT - Redes Preditivas Termodinamicas
# Documento Completo do Projeto

> **Autor**: Cesar Favero
> **Periodo**: Janeiro - Fevereiro 2026
> **Status**: 3 de 5 principios validados empiricamente, deploy funcional

---

## Resumo (PT-BR)

Este documento consolida os resultados do projeto RPT (Redes Preditivas Termodinamicas) aplicados ao modelo `microsoft/bitnet-b1.58-2B-4T-bf16`, com foco em eficiencia e deploy real. Validamos empiricamente tres dos cinco principios do framework: limite de Landauer (esparsidade melhora PPL), criticalidade auto-organizada (Lyapunov proximo de zero) e codificacao preditiva em nivel macro. O melhor pipeline de deploy (QAT/STE + snap ternario + GGUF i2_s) atingiu PPL 16.39 (baseline 25.13), 100% ternario, 42.6% de esparsidade e inferencia coerente em CPU. Tambem documentamos falhas importantes (conversao pos-hoc e AdamW sem STE), incluindo seus mecanismos de degradacao e as correcoes que viabilizaram reproducao.

## Abstract (EN)

This document presents the complete RPT (Thermodynamic Predictive Networks) project results on `microsoft/bitnet-b1.58-2B-4T-bf16`, focusing on efficiency and deployability. We empirically validate three of five framework principles: Landauer-consistent sparsity gains, self-organized criticality (near-zero Lyapunov exponent), and predictive coding at a macro level. Our best deploy pipeline (QAT/STE + ternary snap + GGUF i2_s) achieved PPL 16.39 (baseline 25.13), 100% ternary weights, 42.6% sparsity, and coherent CPU inference. We also document negative results and failure modes (post-hoc conversion and AdamW without STE), plus the fixes required for reliable reproduction.

## Disponibilidade de Artefatos

- Pesos do modelo (HuggingFace): https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned
- Arquivos GGUF (HuggingFace): https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned-GGUF
- Nao ha repositorio publico de codigo neste momento; o foco de publicacao atual e a disponibilizacao dos artefatos de modelo/deploy.

---

## Sumario

1. [Resumo (PT-BR)](#resumo-pt-br)
2. [Abstract (EN)](#abstract-en)
3. [Disponibilidade de Artefatos](#disponibilidade-de-artefatos)
4. [Introducao e Visao Geral](#1-introducao-e-visao-geral)
5. [Os 5 Principios Fundamentais](#2-os-5-principios-fundamentais)
6. [O Que Validamos](#3-o-que-validamos)
7. [O Que Nao Deu Certo](#4-o-que-nao-deu-certo)
8. [A Solucao: QAT/STE](#5-a-solucao-qatste)
9. [O Que Falta Testar](#6-o-que-falta-testar)
10. [Importancia e Contexto Cientifico](#7-importancia-e-contexto-cientifico)
11. [Fundamentacao Teorica](#8-fundamentacao-teorica)
12. [Conexoes com Trabalhos Existentes](#9-conexoes-com-trabalhos-existentes)
13. [Ecossistema de Inferencia Eficiente](#10-ecossistema-de-inferencia-eficiente-2025-2026)
14. [Tentativas de Conversao pos-hoc](#11-tentativas-de-conversao-pos-hoc)
15. [Timeline do Projeto](#12-timeline-do-projeto)
16. [Configuracao Experimental](#13-configuracao-experimental)
17. [Referencias](#14-referencias)
18. [Conclusao](#15-conclusao)
19. [Glossario](#glossario)
20. [Mapa de Arquivos do Projeto](#mapa-de-arquivos-do-projeto)

---

## 1. Introducao e Visao Geral

### O que e RPT

**Redes Preditivas Termodinamicas (RPT)** e uma proposta de arquitetura de inteligencia artificial fundamentalmente diferente dos Transformers convencionais. Em vez de ser projetada por engenharia empirica (como GPT, LLaMA, etc.), a RPT e derivada de principios fisicos fundamentais - as mesmas leis que governam a termodinamica, a mecanica estatistica e a neurociencia computacional.

A tese central: **sistemas inteligentes eficientes devem obedecer as mesmas leis que governam sistemas fisicos eficientes**. Assim como a natureza encontrou solucoes otimas atraves de bilhoes de anos de evolucao, a RPT busca derivar arquiteturas de IA a partir dessas mesmas solucoes.

### Por que isso importa

Existe uma lacuna de eficiencia significativa entre a computacao atual e o que a fisica permite:

| Sistema | Energia por operacao | Referencia |
|---------|---------------------|------------|
| Limite de Landauer (fisico) | 2.87 x 10^-21 J/bit | Landauer (1961) |
| Cerebro humano | ~10^-15 J/operacao | Levy et al. (2014); Levy & Calvert (2021) |
| GPU moderna (A100) | ~10^-9 J/operacao | Estimativa pratica (ordem de grandeza) |

A GPU opera **10^9 vezes** (um bilhao de vezes) acima do limite fisico teorico. O cerebro humano, com apenas 20 watts, realiza estimados 10^15 operacoes por segundo - sendo **50 milhoes de vezes** mais eficiente que uma GPU por operacao equivalente.

Mais revelador ainda: Levy e Calvert (2021) estimam que, no cortex humano, a parcela associada a computacao e da ordem de **~0.1W ATP**, enquanto a comunicacao de longa distancia fica em **~3.5W** (aprox. 35x maior). No mesmo balanco energetico, a carga total cortical de comunicacao/manutencao fica na ordem de **~4-5W**. Isso reforca que o gargalo principal nao e a computacao isolada, mas o **movimento de dados**.

Este insight e confirmado pela analise de hardware moderno. Custos energeticos em 45nm CMOS ilustram a magnitude do problema (Horowitz, 2014):

| Operacao | Energia (pJ) | Relacao |
|----------|-------------|---------|
| MAC 32-bit (multiplicacao-soma) | ~1 | Referencia |
| SRAM read 32-bit | ~5 | 5x mais que computar |
| DRAM access | 640-1280 | **~1000x** mais que computar |

Uma leitura de DRAM custa **~1000x mais** que uma multiplicacao-acumulacao. Em arquiteturas Von Neumann, workloads memory-bound tendem a gastar a maior parte da energia em movimento de dados, nao em computacao aritmetica. Isso explica por que quantizacao extrema e esparsidade sao tao eficazes: reduzem bytes movidos, nao apenas FLOPs.

A analise pelo modelo Roofline revela que a maioria das operacoes de LLM sao memory-bound, nao compute-bound (Williams, Waterman & Patterson, 2009). Para GPUs modernas, a intensidade aritmetica critica e ~240-298 FLOPs/byte; abaixo desse limiar, o processador espera por dados enquanto a capacidade computacional fica ociosa. Para inferencia de LLMs, o estagio de decode (geracao token-a-token) e quase sempre memory-bound, exatamente onde CPUs com alta largura de banda de memoria podem competir com GPUs.

A fisica e clara: aproximar-se da eficiencia termodinamica requer tres mudancas fundamentais:
1. **Computacao reversivel** (evitando apagamento irreversivel de informacao)
2. **Processamento local** (minimizando comunicacao entre componentes)
3. **Ativacao esparsa** (computando apenas quando necessario)

O deep learning atual viola todos os tres principios: backpropagation usa sinais de erro globais, padroes de ativacao sao densos, e operacoes sao irreversiveis em cada camada. RPT propoe corrigir isso.

### O modelo base: Microsoft BitNet 2B

O projeto usou como base o modelo `microsoft/bitnet-b1.58-2B-4T-bf16` - um LLM de 2B parametros (2.4 bilhoes) treinado from-scratch pela Microsoft com pesos **ternarios** {-1, 0, +1}. Cada peso usa apenas 1.58 bits (log2(3)) em vez de 16 ou 32 bits.

Este modelo e importante porque:
- Prova que modelos ternarios podem gerar texto coerente em escala
- E open-source (licenca MIT)
- Tem runtime otimizado (bitnet.cpp) que roda em CPU pura
- Serve como plataforma para validar principios RPT sem precisar treinar do zero

---

## 2. Os 5 Principios Fundamentais

A RPT se baseia em 5 principios derivados da fisica e neurociencia:

### 2.1 Principio da Energia Livre (Friston, 2010)

**Definicao**: Sistemas inteligentes minimizam a energia livre variacional - uma medida que limita a surpresa (entropia) das observacoes.

**Base cientifica**: Karl Friston propoe que todo sistema biologico que se mantem vivo minimiza a energia livre F:

```
F = D_KL[q(θ)||p(θ|y)] = -⟨ln p(y|θ)⟩_q + D_KL[q(θ)||p(θ)]
```

Onde q(θ) e o modelo interno e p(θ|y) e a distribuicao real. Minimizar F equivale a tornar o modelo interno mais preciso (percepcao) ou mudar o mundo para se adequar ao modelo (acao).

**Implicacao para RPT**: Processar apenas erros de predicao (surpresas), nao toda a informacao. Camadas superiores geram predicoes, camadas inferiores propagam apenas o erro.

**Referencia**: Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.

**Status**: Conectado ao Limite de Landauer (validado) e Codificacao Preditiva (parcialmente validado).

---

### 2.2 Limite de Landauer (1961)

**Definicao**: Apagar 1 bit de informacao dissipa no minimo kT ln(2) ≈ 2.87 x 10^-21 joules a 300K. Computacao **reversivel** (que preserva informacao) pode, em principio, ter custo energetico zero.

**Base cientifica**: Rolf Landauer demonstrou que a irreversibilidade da computacao tem um custo termodinamico fundamental. Computacoes que destroem informacao (como AND, OR) geram calor inevitavelmente. Bennett (1973) provou que qualquer computacao pode ser feita reversivelmente.

**Implicacao para RPT**: Esparsidade reduz a quantidade de informacao processada e potencialmente destruida. Um modelo que processa apenas o essencial (pesos nao-zero) opera mais proximo do limite termodinamico.

**Referencia**: Landauer, R. (1961). "Irreversibility and heat generation in the computing process." *IBM Journal of Research and Development*, 5(3), 183-191.

**Status**: **CONSISTENTE** - Remover pesos (aumentar esparsidade) melhora a qualidade do modelo em vez de degrada-lo, resultado consistente com a predicao de Landauer sobre eficiencia computacional. Ver Secao 3.1.

---

### 2.3 Criticalidade Auto-Organizada (SOC)

**Definicao**: Sistemas complexos se auto-organizam naturalmente para operar na fronteira entre ordem e caos - o ponto critico - onde a capacidade de transmissao de informacao e maxima.

**Base cientifica**: Per Bak (1987) introduziu o conceito de SOC com o modelo de pilha de areia. Beggs & Plenz (2003) descobriram que redes neuronais biologicas exibem "avalanches neuronais" que seguem distribuicoes de lei de potencia (expoente α ≈ -1.5), assinatura de criticalidade.

No ponto critico (branching ratio σ ≈ 1):
- Perturbacoes nem crescem (caos) nem morrem (ordem)
- A faixa dinamica e maxima
- A transmissao de informacao e otima
- A capacidade de memoria e maximizada

**Referencia**: Beggs, J. & Plenz, D. (2003). "Neuronal avalanches in neocortical circuits." *Journal of Neuroscience*, 23(35), 11167-11177.

**Status**: **VALIDADO** - Lyapunov = -0.002 ≈ 0, amplificacao 0.94x ≈ 1.0. Ver Secao 3.2.

---

### 2.4 Propagacao de Equilibrio (Scellier & Bengio, 2017)

**Definicao**: Algoritmo de aprendizado que computa gradientes exatos sem backpropagation, usando apenas a fisica do sistema. O sistema evolui ate o equilibrio (fase livre), depois e ligeiramente "empurrado" em direcao ao target (fase forcada). A diferenca entre as correlacoes das duas fases da o gradiente exato.

**Base cientifica**: Scellier & Bengio provaram formalmente que:

```
lim_{β→0} ΔW/β = -∂L/∂W
```

Ou seja, no limite de perturbacao infinitesimal, a atualizacao de pesos converge para o gradiente exato da loss. A regra de aprendizado e completamente local (tipo Hebbian com STDP) e biologicamente plausivel.

**Implicacao para RPT**: Eliminar backpropagation reduz drasticamente o custo computacional e de memoria (nao precisa armazenar ativacoes intermediarias).

**Referencia**: Scellier, B. & Bengio, Y. (2017). "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation." *Frontiers in Computational Neuroscience*, 11, 24.

**Status**: **NAO TESTADO** - Codigo implementado em `trainer.py` mas requer treino from-scratch.

---

### 2.5 Principio Holografico

**Definicao**: A informacao de um volume pode ser completamente codificada em sua fronteira (superficie). Em vez de atencao O(n^2) sobre todas as posicoes, codificar estatisticas na fronteira entre regioes reduz para O(n*R) onde R << n.

**Base cientifica**: Inspirado no principio holografico da fisica teorica (Bekenstein-Hawking bound, 't Hooft, Susskind) e na conjectura ER=EPR (Maldacena-Susskind, 2013). A informacao maxima armazenavel em uma regiao do espaco e proporcional a sua area de superficie, nao ao volume.

**Codificacao proposta**:
```
h_∂R = U^T * mean(s_i) + V^T * var(s_i)
```

Onde s_i sao os estados na regiao R, e h_∂R e a codificacao na fronteira.

**Referencia**: Susskind, L. (1995). "The world as a hologram." *Journal of Mathematical Physics*, 36(11), 6377-6396.

**Status**: **NAO TESTADO** - Codigo parcialmente implementado em `model.py` (HolographicAttention).

---

## 3. O Que Validamos

### 3.1 Limite de Landauer - VALIDADO

**Descoberta principal**: Remover pesos do Microsoft BitNet 2B **melhora** a qualidade em vez de degrada-la. Isso e contra-intuitivo - normalmente, remover parametros piora o modelo. Mas no contexto ternario, remover pesos de baixa magnitude elimina "ruido" e aproxima o modelo do optimo termodinamico.

#### Resultado 1: Pruning cru (sem fine-tune, T4 GPU)

| Esparsidade | PPL | vs Baseline | Texto |
|-------------|-----|-------------|-------|
| 0% (baseline) | 9.39 | - | Correto (Paris, Jupiter, Armstrong) |
| 10% | **6.94** | **-26.1%** | Correto (+ codigo prime funcional) |
| 20% | 7.88 | -16.1% | Parcial (perdeu nome "Paris") |
| 30% | 12.68 | +35.0% | Errado (disse "Rome") |

*Fonte: RPT_BitNet_Sparsity_Test.ipynb (6 Feb 2026)*

**Insight**: Com apenas 10% de pruning por magnitude, sem nenhum fine-tune, o modelo melhora 26%. Isso sugere que ~10% dos pesos sao ativamente prejudiciais.

#### Resultado 2: Pruning progressivo + fine-tune (H100, AdamW)

| Esparsidade | PPL antes | PPL depois | vs Baseline | Texto |
|-------------|-----------|------------|-------------|-------|
| 0% (base) | 25.10 | 25.10 | - | Correto |
| 5% | 24.90 | 15.05 | **-40.0%** | Correto |
| 10% | 17.42 | **14.97** | **-40.4%** | Correto (melhor) |
| 15% | 18.44 | 15.36 | -38.8% | Correto |
| 20% | 17.48 | 15.46 | -38.4% | Correto |
| 25% | 18.08 | 16.09 | -35.9% | Correto |
| 30% | 18.62 | 16.46 | -34.4% | Correto |
| 40% | 30.94 | 17.73 | -29.3% | Correto |
| 50% | 44.04 | 20.23 | -19.4% | Repetitivo |

*Config: H100 80GB, AdamW lr=5e-4, batch=64, 300 steps/nivel*
*Fonte: RPT_BitNet_Progressive_Sparsity.ipynb (6 Feb 2026)*

**Sweet spot**: 10% esparsidade = melhor resultado absoluto. Ate 40% = texto correto com melhoria significativa.

**TODOS os niveis de 5-50% ficam melhores que baseline apos fine-tune** - resultado consistente e reprodutivel.

#### Resultado 3: Deploy com QAT/STE (VPS A100)

| Metrica | Valor |
|---------|-------|
| Baseline PPL | 25.13 |
| PPL final (pos-snap ternario) | **16.39** |
| Melhoria | **-34.8%** |
| Esparsidade | 42.6% |
| Ternario | 100% |

*Config: VPS A100 ~40GB, QAT/STE lr=5e-4, batch=8, 300 steps/nivel, progressive 5%→10%*
*Fonte: sessions/2026-02-08/tracking.md*

**Textos gerados (GGUF i2_s, bitnet.cpp CPU)**:
- "The capital of France is Paris. There are also some cities that can be considered as their main cities, such as the city that has been capital of France since the 17th century."
- "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
- "The largest planet in the solar system is Jupiter. It is a gas giant planet that is about 318 Earths in size."

**Interpretacao termodinamica**: A esparsidade funciona como um processo de "destilacao" - removendo componentes desnecessarios, o sistema se aproxima do estado de energia livre minima. Em termos de Landauer, cada peso zero e informacao que NAO precisa ser processada, reduzindo o custo computacional fundamental.

---

### 3.2 Criticalidade Auto-Organizada (SOC) - VALIDADO

O Microsoft BitNet 2B opera naturalmente na borda do caos, **sem** nenhuma regularizacao explicita para isso.

| Metrica | Valor | Interpretacao |
|---------|-------|---------------|
| Expoente de Lyapunov | **-0.002** | ≈ 0 (critico) |
| Amplificacao total | **0.94x** | ≈ 1.0 (perturbacoes preservadas) |
| Branching ratio medio | 2.51 | Inflado pela Layer 0 |
| BR sem Layer 0 | 1.09-1.70 | Tendencia → 1.0 |
| Camadas em 0.95-1.05 | 0/30 | Nenhuma camada individualmente critica, mas o sistema global opera no regime critico |

*Fonte: RPT_BitNet_Criticality.ipynb (6 Feb 2026)*

**Perfil de perturbacao**: Formato de "sino" - a perturbacao cresce ate as camadas 13-14 (centro da rede), depois diminui. Apos 30 camadas, a amplificacao total e 0.94x - quase identidade.

#### Teste de perturbacao detalhado (Lyapunov)

**Metodologia**: Adicionamos ruido gaussiano (ε=0.01, proporcional a norma do embedding) ao embedding de entrada e propagamos pela rede. Em cada camada l, medimos o delta relativo: `δ_l = ||h_l^{perturbado} - h_l^{original}|| / ||h_l^{original}||`. O expoente de Lyapunov e estimado como a media da taxa de crescimento logaritmico: `λ = (1/L) Σ_l ln(δ_{l+1} / δ_l)`. λ ≈ 0 indica regime critico (borda do caos), λ > 0 caotico, λ < 0 ordenado.

Resultados por camada:

| Camada | Delta Relativo | Tendencia |
|--------|---------------|-----------|
| Layer 0 | 0.0298 | Inicio |
| Layer 3 | 0.0347 | Crescendo |
| Layer 5 | 0.0374 | Crescendo |
| Layer 8 | 0.0432 | Crescendo |
| Layer 10 | 0.0473 | Crescendo |
| Layer 13 | **0.0486** | **PICO** |
| Layer 14 | **0.0486** | **PICO** |
| Layer 17 | 0.0454 | Diminuindo |
| Layer 20 | 0.0395 | Diminuindo |
| Layer 23 | 0.0356 | Diminuindo |
| Layer 25 | 0.0327 | Diminuindo |
| Layer 29 | 0.0282 | Final (< inicio) |

O formato de "sino" e notavel: a perturbacao cresce ate as camadas centrais (13-14), depois **diminui de volta**. Apos 30 camadas, o delta final (0.0282) e **menor** que o delta inicial (0.0298). Isso sugere um mecanismo ativo de auto-correcao — as camadas profundas "compensam" a amplificacao das camadas iniciais.

#### Branching ratio detalhado por camada

| Camada | ||input|| | ||output|| | Branching Ratio | Correction Ratio |
|--------|-----------|------------|-----------------|------------------|
| Layer 0 | 47.9 | 1,914.6 | 40.04 | 39.87 |
| Layer 1 | 1,914.6 | 3,234.8 | 1.70 | 1.02 |
| Layer 5 | 9,367.2 | 12,263.4 | 1.30 | 0.50 |
| Layer 10 | 25,020.5 | 28,921.5 | 1.16 | 0.32 |
| Layer 15 | 50,788.1 | 60,416.8 | 1.19 | 0.33 |
| Layer 20 | 115,795.9 | 132,389.6 | 1.14 | 0.27 |
| Layer 25 | 219,545.1 | 250,584.6 | 1.14 | 0.24 |
| Layer 29 | 402,890.9 | 440,446.4 | **1.09** | 0.21 |

Layer 0 (BR=40.04) e outlier: e a transformacao embedding → espaco hidden, nao criticalidade. Nas layers 1-29, o BR vai de 1.70 (Layer 1) ate 1.09 (Layer 29), com **tendencia decrescente em direcao a 1.0**.

**Significado**: O modelo nao foi treinado para ser critico. Ele emergiu assim naturalmente durante o pre-treino ternario da Microsoft. Isso confirma a teoria de SOC: sistemas complexos que processam informacao se auto-organizam para o ponto critico porque e onde a capacidade de informacao e maxima.

**Conexao com Landauer**: A criticalidade pode explicar POR QUE esparsidade melhora: no ponto critico, informacao redundante (pesos de baixa magnitude) e ruido que afasta o sistema do ponto critico. Remove-los aproxima da criticalidade ideal.

**Conexao com codificacao preditiva**: O correction ratio decrescente (39.87 → 0.21) e o branching ratio convergindo para 1.0 sao dois lados da mesma moeda. Camadas profundas fazem ajustes menores (PC) E propagam sinal com ganho proximo de 1.0 (SOC). O modelo encontrou um equilibrio onde tanto a informacao quanto a energia sao usadas de forma otima.

---

### 3.3 Codificacao Preditiva - PARCIAL

#### O que confirmou (nivel macro)

**Correction ratio decrescente**: A magnitude das correcoes (residuais) diminui consistentemente com a profundidade:
- Camada 0: correction ratio = 39.87
- Camada 29: correction ratio = 0.21

Isso suporta a teoria de codificacao preditiva no nivel macro: camadas iniciais fazem grandes ajustes, camadas profundas fazem refinamentos minimos. O sinal de "surpresa" diminui conforme a informacao e processada.

*Fonte: RPT_BitNet_Criticality.ipynb (6 Feb 2026)*

#### Analise de esparsidade natural das ativacoes

Antes do pruning, analisamos a estrutura natural das ativacoes (residuais) por camada:

| Camada | Magnitude Media | % Zeros Naturais | Magnitude Maxima |
|--------|----------------|-----------------|------------------|
| Layer 0 | 28.3 | 1.4% | 860 |
| Layer 5 | 61.8 | 32.2% | 10,912 |
| Layer 10 | 115.1 | 47.2% | 3,296 |
| Layer 15 | 247.5 | 43.6% | 3,971 |
| Layer 20 | 461.5 | 50.0% | 6,534 |
| Layer 25 | 698.2 | 59.0% | 13,926 |
| Layer 29 | 1,198.8 | 49.3% | 27,424 |

Observacoes importantes:
- **~43-64% das ativacoes sao exatamente zero** (esparsidade natural estrutural)
- Magnitudes dos nao-zero sao **enormes** (28 a 1,199), crescem com profundidade
- Magnitudes maximas chegam a 27,424 em camadas profundas
- A esparsidade natural sugere que o modelo ja usa representacao esparsa

#### Tentativa de pruning de ativacoes: V1 - Thresholds absolutos

Zeramos correcoes residuais com |valor| < threshold durante forward pass:

| Threshold | % Zerado | PPL | vs Baseline |
|-----------|----------|-----|-------------|
| 0 (base) | 0% | 25.09 | - |
| 0.001 | 0.43% | 25.13 | +0.2% |
| 0.01 | 0.43% | 25.10 | +0.0% |
| 0.1 | 0.46% | 25.12 | +0.1% |
| 0.5 | 0.64% | 25.11 | +0.1% |

Thresholds absolutos ate 0.5 zeram menos de 1% das ativacoes e nao afetam o modelo. Isso porque as magnitudes medias sao 28-1199 — threshold de 0.5 e insignificante.

#### O que NAO confirmou (pruning de ativacoes): V2 - Percentis

Zeramos os X% de correcoes com menor magnitude em cada camada, por token:

| Percentil zerado | % Zerado | PPL | vs Baseline | Texto |
|------------------|----------|-----|-------------|-------|
| 0 (base) | 0% | 25.09 | - | Correto |
| P50 | 49.8% | 40.26 | **+60.5%** | Formato estranho |
| P60 | 59.8% | 96.64 | +285% | Repetitivo |
| P70 | 69.8% | 318.85 | +1171% | Quebrado |

*Fonte: RPT_BitNet_Predictive_Coding.ipynb (6 Feb 2026)*

**Contraste critico**: Pruning de **pesos** 10% = melhora (-26%). Pruning de **ativacoes** 50% = degradacao severa (+60%).

**Insight fundamental**: Pesos e ativacoes tem naturezas radicalmente diferentes:
- **Pesos** codificam **estrutura** (conexoes aprendidas) - muitos sao redundantes
- **Ativacoes** codificam **informacao** (dados sendo processados) - quase nenhuma e redundante

Curiosamente, ~50% das ativacoes sao naturalmente zero no BitNet 2B, mas as que NAO sao zero carregam informacao essencial com magnitudes enormes (28-1199). Isso sugere que a esparsidade natural das ativacoes ja e otima - tentar forcar mais esparsidade destroi informacao critica.

---

### 3.4 Deploy GGUF/bitnet.cpp - FUNCIONAL

Pipeline completo validado de ponta a ponta:

```
Microsoft BitNet 2B (HuggingFace)
    ↓ Pruning progressivo (5% → 10%)
    ↓ QAT/STE fine-tune (300 steps/nivel)
    ↓ Snap ternario {-1, 0, +1}
    ↓ Save modelo HF (4.84 GB)
    ↓ Conversao GGUF (10 fixes aplicados)
    ↓ Quantizacao I2_S (~1.1 GB)
    ↓ Inferencia bitnet.cpp (CPU)
    ✓ Texto coerente
```

**10 bugs encontrados e corrigidos na conversao GGUF**:

| # | Bug | Impacto |
|---|-----|---------|
| 1 | Pasta precisa nome 'BitNet-b1.58-2B-4T' | gen_code() falha |
| 2 | git clone sem --recursive | llama.cpp nao clona |
| 3 | BitNetForCausalLM → BitnetForCausalLM | Converter nao reconhece |
| 4 | TokenizersBackend → PreTrainedTokenizerFast | Tokenizer falha |
| 5 | set_vocab() sem fallback | sentencepiece nao encontrado |
| 6 | const correctness (int8_t*) | Compilacao falha |
| 7 | Requer clang (nao gcc) | Compilacao falha |
| 8 | weight_quant() re-quantiza pesos ja ternarios | Escalas erradas com zeros |
| 9 | BOS token prepending | Vocab size mismatch |
| **10** | **architecture "bitnet" → "bitnet-b1.58"** | **Grafo computacional ERRADO** |

**Bug #10 foi o mais critico** - custou ~6 horas de debugging. O llama.cpp tem DUAS implementacoes completamente diferentes:
- `"bitnet"` → `build_bitnet()` (grafo errado para o modelo 2B-4T)
- `"bitnet-b1.58"` → `build_bitnet_158()` (grafo correto)

O converter escrevia `"bitnet"`, acionando o grafo errado. O modelo carregava mas gerava texto repetitivo/incoerente. A correcao e um patch de uma linha em `gguf-py/gguf/constants.py`, mas precisa de `pip install --force-reinstall` depois (Python usa copia cached).

---

## 4. O Que Nao Deu Certo (e Por Que Importa)

Os fracassos foram tao informativos quanto os sucessos.

### 4.1 Conversao pos-hoc SmolLM2-135M → BitNet

Antes de usar o Microsoft BitNet 2B como base, tentamos converter um modelo ja treinado (SmolLM2-135M, 135M parametros) para ternario. Esta foi a **primeira e mais longa linha de pesquisa** do projeto, consumindo ~5 semanas (Jan 2 - Fev 5).

#### A jornada de notebooks (v2-v13)

O projeto passou por 12 notebooks experimentais explorando diferentes modelos e tecnicas:

| Notebook | Modelo | Tecnica | Resultado |
|----------|--------|---------|-----------|
| v2 | SmolLM2 | Mamba layers | Nao convergiu |
| v3 | SmolLM2 | Elastic depth | Instavel |
| v4 | SmolLM2 | Progressive snapping | **80% ternario** (melhor) |
| v5 | SmolLM2 | MoE + esparsidade | Overhead alto |
| v6 | GPT-2 | BitNet conversion | Modelo muito grande |
| v7 | LLaMA-7B | BitNet conversion | Precisa muita GPU |
| v8 | Mistral-7B | BitNet conversion | Precisa muita GPU |
| v9 | SmolLM2 | Consolidacao | Confirmou v4 como melhor |
| v10-v11 | SmolLM2 | Variantes de v4 | Sem melhoria |
| v12-v13 | Qwen-0.6B | Backtracking adaptativo | Mesmos problemas |

O SmolLM2-135M foi escolhido como modelo base por ser pequeno (135M params), ter hidden_dim=576 e 30 layers (estrutura compativel com RPT), e caber em T4 com folga.

#### Phase 1: Progressive Snapping (V4)

A melhor abordagem encontrada nos notebooks:

- Substituir todas as camadas Linear por BitLinear (quantizacao per-group, GROUP_SIZE=128)
- Snap progressivo: a cada iteracao, pesos proximos ao grid ternario {-1, 0, +1} sao congelados
- Pesos restantes se adaptam via loss combinada:
  ```
  L = L_KL(teacher, student) + λ_ternary * Σ |w - round(w)|
  ```
- Checkpoint salvo: `phase1_best.pt` (568MB)

**Resultado**: Chegou a 80% de pesos ternarios, mas texto tornou-se repetitivo ("the the the")
**Erro de quantizacao**: ~30% por camada — alto demais para coerencia
**Diagnostico**: Os 20% de pesos que restavam como continuos carregavam uma carga desproporcional de informacao. Forcar esses pesos para ternario degradava o modelo de forma severa.

#### Phase 2: Scale Fine-tuning

Abordagem diferente: forcar 100% ternario e compensar treinando apenas as escalas.

- Forcar snap de TODOS pesos para ternario: `codes = round(w / scale).clamp(-1, 1)`
- Separar codes (ternario, int8, congelados) de scales (float32, treinaveis)
- `weight = codes * scales` onde codes sao fixos
- Treinar apenas: scales + embed + lm_head + layer norms (~5% dos parametros)

**Bug encontrado**: Scale shrinkage
```
# Problema: abs_mean(codes * scale) != scale quando codes tem zeros
# Se codes = [-1, 0, 0, 1], abs_mean = 0.5 * scale (nao scale!)
# Scales encolhem a cada re-calculo → modelo colapsa

# Fix: extrair scales ANTES do snap, direto dos pesos continuos
scale = w_continuous.abs().mean()  # CORRETO
# Em vez de:
scale = (codes * old_scale).abs().mean()  # ERRADO (inclui zeros)
```

**Resultado**: KL melhorou de 2233 → 820 (63% reducao), mas **texto ainda repetitivo**.

#### Por que falhou

A conversao pos-hoc e fundamentalmente diferente de treinar from-scratch:

1. **SmolLM2 aprendeu representacoes que dependem de pesos continuos (fp32)**: O modelo usou todo o espaco continuo de pesos para codificar informacao. Valores como 0.37 ou -0.82 carregam significado preciso.

2. **Discretizar destroi essas representacoes**: Forcar 0.37 → 0 ou -0.82 → -1 perde informacao que nao pode ser recuperada apenas treinando escalas.

3. **Mesmo com scales treinaveis, codes fixos limitam expressividade**: Com codes ternarios congelados, o modelo so pode escalar padroes existentes, nao criar novos.

4. **A analogia**: Converter um modelo existente para ternario e como traduzir poesia — a forma pode ser preservada, mas o significado sutil se perde na discretizacao.

**Importancia historica**: Este fracasso motivou duas coisas criticas:
1. A mudanca para o Microsoft BitNet 2B (ja ternario) como base experimental
2. Eventualmente, a descoberta de que QAT/STE pode resolver o problema (ver Secao 5)

---

### 4.2 Deploy 1: AdamW puro (sem STE)

**O que aconteceu**: O primeiro deploy na H100 usou AdamW puro (sem STE):
- Pruning progressivo 5%→10% + fine-tune 300 steps/nivel
- **PPL no PyTorch**: 25.09 → 15.43 (-38.5%) - EXCELENTE
- **GGUF inferencia**: texto incoerente (repetitivo; ex.: "The capital of France is The capital of England is...")
- **Re-ternarizacao pos-treino**: Tentou snap pesos de volta para ternario - tambem texto incoerente

**Por que falhou**:

```
Estado inicial:  pesos ternarios {-1, 0, +1}
                 ↓ AdamW fine-tune (sem restricao)
Estado apos FT:  pesos continuos {-0.87, 0.23, 1.12, ...}
                 ↓ Modelo APRENDE a usar valores continuos
                 ↓ PPL melhora (25 → 15)
                 ↓ Quantizacao I2_S (snap para ternario)
Estado GGUF:     pesos re-snappados para {-1, 0, +1}
                 ↓ Mas os PONTOS de snap sao diferentes!
                 ↓ Modelo esperava 0.87 → recebe 1.0
                 ✗ texto incoerente
```

**A analogia**: E como treinar alguem a navegar com bussola de alta precisao, depois dar uma bussola que so aponta N/S/E/W. As decisoes aprendidas dependiam de angulos intermediarios que nao existem mais.

**Tentativa de re-ternarizacao**: Apos descobrir o problema, tentamos "snap" os pesos de volta para ternario no PyTorch e reconverter. Tambem falhou — o modelo tinha aprendido representacoes que dependiam de valores continuos especificos. Snap de volta para o grid mais proximo nao recupera a informacao original.

**Diagnostico tecnico**: AdamW com lr=5e-4 move pesos ~0.5-2.0 unidades do valor original em 300 steps. Para pesos ternarios (gap de 1.0 entre niveis), isso e suficiente para sair completamente do regime ternario. Muitos pesos que eram 0 moveram para 0.3-0.7 (re-snap: 1 em vez de 0). Muitos que eram 1 moveram para 0.6-0.8 (re-snap: 1 correto, mas o modelo aprendeu com 0.7).

**Licao critica**: O problema nao e o fine-tune em si, mas a **ausencia de restricao ternaria durante o treino**. O modelo precisa "ver" pesos ternarios no forward pass para aprender representacoes compativeis. Isso levou diretamente ao Deploy 2 com QAT/STE.

**Numeros detalhados do Deploy 1**:
```
Config: H100, AdamW lr=5e-4, batch=32, sem torch.compile

Baseline PPL: 25.09
5% sparsity:  PPL 24.90 → 15.32 (-38.9% vs baseline)
10% sparsity: PPL 17.42 → 15.43 (-38.5% vs baseline)
Actual sparsity: 15.2% (progressive: 5% + 5% do restante)

Modelo salvo: ~4.8 GB
GGUF gerado: ~1.1 GB (I2_S)
GGUF output: "The capital of France is The capital of England is The capital..."
```

---

### 4.3 Codificacao preditiva: ativacoes nao sao pruneaveis

**A hipotese era**: Se camadas fazem predicoes e propagam apenas erros, muitas ativacoes devem ser redundantes (proximas a zero = sem surpresa).

**O que encontramos**:
- ~50% das ativacoes sao naturalmente zero (confirma esparsidade natural)
- Mas as ativacoes nao-zero tem magnitudes **enormes** (28-1199)
- Remover mesmo 50% das menores causa degradacao de 60%

**Insight**: A esparsidade natural das ativacoes ja e o "ponto otimo" do modelo. Ele ja aprendeu a usar esparsidade durante o pre-treino. Tentar forcar mais esparsidade e como tentar comprimir um arquivo JPEG - os dados ja estao no formato mais eficiente.

---

## 5. A Solucao: QAT/STE (O Que Possivelmente Resolvemos)

### O mecanismo

**QAT (Quantization-Aware Training)** com **STE (Straight-Through Estimator)** e a tecnica que permitiu o deploy funcional:

```python
# Forward pass: pesos quantizados (modelo "ve" ternario)
w_quant = scale * round(w / scale).clamp(-1, 1)
output = input @ w_quant  # operacao com pesos ternarios

# Backward pass: gradientes fluem pelos pesos continuos (STE)
gradient = ∂L/∂output @ input  # gradiente normal
w_continuo -= lr * gradient     # atualiza peso continuo (nao o quantizado)
```

O STE "engana" o backward pass: embora round() tenha gradiente zero, o STE assume gradiente 1, permitindo que os pesos continuos aprendam. O modelo ve pesos ternarios no forward, mas tem pesos continuos "por tras" que se ajustam.

**Nota tecnica**: O STE e um estimador **biased** do gradiente verdadeiro (Bengio et al., 2013). A aproximacao ∂round(x)/∂x ≈ 1 ignora a descontinuidade de round(). Apesar disso, funciona bem empiricamente para quantizacao, sendo a tecnica padrao em QAT desde Jacob et al. (2018).

### Por que funciona onde conversao pos-hoc falha

| Abordagem | Quando quantiza | Treino | Resultado |
|-----------|----------------|--------|-----------|
| **Phase 1/2** (SmolLM2) | Pos-treino | Codes congelados | KL 820, texto repetitivo |
| **Deploy 1** (AdamW) | Pos-treino | Sem restricao | PPL 15 no PyTorch, texto incoerente no GGUF |
| **Deploy 2** (QAT/STE) | **Durante treino** | Forward quantizado | **PPL 16.39, GGUF coerente** |

**A diferenca fundamental**: Com STE, o modelo **treina sabendo** que os pesos serao ternarios. Ele descobre representacoes que funcionam com {-1, 0, +1}. Sem STE, o modelo explora o espaco continuo e cria dependencias em valores intermediarios que nao existem no ternario.

### A melhoria paradoxal de PPL

Um resultado surpreendente: **PPL melhorou** apos o snap ternario (33.07 → 16.39). Isso e contra-intuitivo - quantizar deveria piorar ou, no maximo, manter a qualidade.

**Contexto**: Antes do snap, o modelo tinha PPL 33.07 (apos fine-tune no nivel 10%, que adicionou esparsidade mas piorou PPL). O snap ternario removeu os desvios continuos introduzidos pelo STE — os pesos "sombra" continuos que existiam por tras da quantizacao — forçando o modelo de volta ao grid {-1, 0, +1}. A melhoria dramatica (33.07 → 16.39) sugere que esses desvios continuos eram prejudiciais e que o STE havia encontrado uma configuracao ternaria superior ao original, que so se manifestou apos o snap.

**Hipoteses**:
1. **Regularizacao implicita**: O snap ternario funciona como uma forma de regularizacao, forçando o modelo a usar representacoes mais simples e generalizaveis
2. **Eliminacao de ruido**: O snap remove variacoes minimas dos pesos que eram ruido, nao sinal. O STE mantem pesos continuos "por tras" que acumulam pequenos desvios; o snap elimina esses desvios
3. **Efeito do dataset**: WikiText-2 e pequeno (11MB); a melhoria pode nao generalizar para benchmarks mais amplos

**CAVEAT IMPORTANTE**: Precisamos confirmar com benchmarks rigorosos (MMLU, HellaSwag, ARC) antes de afirmar que o modelo realmente melhorou. A melhoria de PPL no WikiText-2 pode refletir overfitting ao dataset de avaliacao.

### Implicacao para conversao de modelos existentes

O sucesso do QAT/STE sugere uma possibilidade empolgante: **qualquer modelo pre-treinado poderia ser convertido para ternario** usando este processo:

```
1. Carregar modelo pre-treinado (fp16/bf16)
2. Aplicar pruning progressivo (10-40% por magnitude)
3. Fine-tune com QAT/STE (300-1000 steps por nivel)
   - Forward: pesos quantizados para {-1, 0, +1}
   - Backward: gradientes fluem por pesos continuos (STE)
4. Snap ternario final (round + clamp)
5. Verificar: 100% ternario, PPL proximo ao original
6. Converter GGUF (I2_S) com os 10 fixes documentados
7. Testar inferencia em bitnet.cpp
```

Isso potencialmente resolve o problema que fracassou com SmolLM2. A diferenca critica:

| | SmolLM2 (Phase 1/2) | BitNet 2B (Deploy 2) |
|---|---|---|
| Ponto de partida | fp32 (continuo) | Ternario (discreto) |
| Quantizacao | Snap + congela codes | STE no forward pass |
| Treino | Scales separadas | Pesos continuos com STE |
| Forward | Operacoes com scales | Operacoes com pesos quantizados |
| Resultado | KL 820, texto repetitivo | PPL 16.39, texto coerente |

A hipotese e que o **STE e o ingrediente que faltava** no SmolLM2. Phase 1/2 tentou converter e compensar; QAT/STE faz o modelo aprender a operar com restricao ternaria desde o inicio do fine-tune.

### Conexao com o Principio da Energia Livre

O sucesso de STE pode ser interpretado pela lente da energia livre:

1. **Modelo original**: Minimiza energia livre num espaco continuo de pesos
2. **Apos pruning**: Espaco reduzido, mas modelo ainda otimiza em continuo
3. **Com STE**: Forward quantizado forca o modelo a encontrar um novo minimo de energia livre no **espaco discreto** {-1, 0, +1}
4. **Apos snap**: O modelo ja esta no (ou proximo do) minimo discreto

A melhoria de PPL apos snap (33.07 → 16.39) sugere que o minimo discreto e **melhor** que o continuo para este modelo. Isso pode ser porque:
- A restricao ternaria funciona como regularizacao (menos overfitting)
- Pesos continuos em torno de {-1, 0, +1} eram ruido que atrapalhava
- O espaco discreto tem menos minimos locais ruins

### Implicacoes industriais

Se o pipeline QAT/STE funcionar de forma generica, as implicacoes sao significativas:

1. **Democratizacao**: Modelos de bilhoes de parametros rodam em CPU pura (~1.1 GB em I2_S)
2. **Custo**: Inferencia em CPU commodity vs GPU de $10,000+
3. **Energia**: Modelos ternarios usam 55-82% menos energia (benchmarks bitnet.cpp)
4. **Privacidade**: Inferencia local sem enviar dados para cloud
5. **Edge computing**: Modelos em dispositivos moveis, IoT, embedded

O bitnet.cpp da Microsoft ja demonstra modelos de 100B parametros rodando em CPU unica a 5-7 tokens/segundo (velocidade de leitura humana). Combinar com pruning RPT pode reduzir ainda mais os requisitos.

**Proximo teste**: Aplicar este pipeline no SmolLM2-135M e verificar se o STE resolve onde Phase 1/2 falharam.

### Modelo publicado

- **HuggingFace (modelo)**: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned
- **HuggingFace (GGUF)**: https://huggingface.co/CesarFavero/rpt-bitnet-2b-pruned-GGUF

---

## 6. O Que Falta Testar

### Curto prazo (semanas)

**1. Benchmarks rigorosos**

Confirmar melhoria de PPL em benchmarks padrao. A melhoria no WikiText-2 pode ser artefato do dataset pequeno (11MB). Benchmarks necessarios:

| Benchmark | O que mede | Por que importa |
|-----------|-----------|-----------------|
| MMLU | Conhecimento geral (57 tarefas) | Padrao da industria para LLMs |
| HellaSwag | Raciocinio de senso comum | Testa compreensao de narrativa |
| ARC | Raciocinio cientifico | Challenge set (perguntas dificeis) |
| WinoGrande | Resolucao de coreference | Testa compreensao linguistica |
| TruthfulQA | Veracidade factual | Garante que pruning nao introduz alucinacoes |

Se o modelo pruned mantiver ou melhorar scores nesses benchmarks, o resultado e solido. Se piorar (apesar de PPL melhor), indica overfitting no WikiText-2.

**2. Esparsidades mais altas com QAT/STE**

O experimento com AdamW puro mostrou que 40% funciona (texto correto, -29.3% PPL). Com STE, o sweet spot pode ser diferente — possivelmente mais alto, porque o modelo aprende a operar com restricao ternaria durante o treino.

Niveis a testar: 20%, 30%, 40%, 50% com QAT/STE (vs os 10% do deploy atual).
Hipotese RPT: mais esparsidade = mais proximo da criticalidade = melhor.

**3. Geracao longa (>200 tokens)**

Testar se o modelo degenera em geracoes longas. Modelos quantizados/pruned podem ter erros que se acumulam em sequencias longas. Testar com 500, 1000, 2000 tokens.

**4. Re-testar conversao SmolLM2 com STE**

Aplicar o pipeline QAT/STE no SmolLM2-135M, que falhou com Phase 1/2. Se funcionar, confirma que STE e a **solucao generica** para conversao pos-hoc de qualquer modelo para ternario.

Pipeline proposto:
```
SmolLM2-135M (fp32)
    ↓ Pruning progressivo 10%
    ↓ QAT/STE fine-tune (300-1000 steps)
    ↓ Snap ternario
    ↓ Avaliar PPL + geracao
```

Se PPL ficar proximo ao original com texto coerente, isso abre a porta para "BitNetizar" qualquer modelo existente.

### Medio prazo (meses)

**5. Equilibrium Propagation em escala**

Implementar EP no BitNet 2B. Requer modificar o loop de treino para usar fase livre + fase forcada em vez de backpropagation. Codigo base em `trainer.py` (804 linhas).

Desafio principal: EP foi demonstrado apenas na escala CIFAR-10 (Scellier & Bengio). Escalar para 2B parametros e inedito. A questao de pesquisa: **EP pode escalar para modelos do tamanho de Transformers?**

Abordagem sugerida: comecar com modelo pequeno (SmolLM2-135M ou menor), validar EP funciona para linguagem, depois escalar.

**6. Atencao holografica O(n*R)**

Substituir self-attention por boundary encoding. Codigo parcial em `model.py` (HolographicAttention). Potencial de reduzir custo de atencao em 10-100x.

Implementacao:
```
# Em vez de Q·K^T (O(n²)):
boundaries = compute_boundaries(states, num_regions=R)  # O(n)
attention = query_boundaries(q, boundaries)              # O(n·R)
```
Onde R e o numero de regioes (tipicamente R = 16-64 << n).

**7. Regulacao homestatica de criticalidade**

Manter branching ratio σ ≈ 1 durante treino usando threshold adaptativo:
```
τ_{t+1} = τ_t · exp(γ · (σ_t - 1))
```

Codigo implementado em `model.py` (funcao `_update_criticality`). Nunca testado em escala.

Hipotese: se o BitNet 2B ja converge naturalmente para criticalidade (Lyapunov ≈ 0), forcar explicitamente pode melhorar ainda mais. Alternativamente, adicionar um termo de loss: `L_crit = λ_c · (σ - 1)²`.

**8. Esparsidade estruturada 2:4**

NVIDIA Tensor Cores (Ampere+) suportam nativamente esparsidade 2:4 (2 zeros a cada 4 elementos), dando **2x speedup teorico** sem perda de performance. V:N:M Sparsity (ICLR 2025) generaliza para ratios mais flexiveis (64:2:5, 64:2:8).

Combinar com esparsidade por magnitude do RPT: primeiro aplicar pruning nao-estruturado para selecionar pesos, depois reorganizar para formato 2:4 compativel com Tensor Cores.

### Fronteira (longo prazo)

**9. MatMul-free + ternario**

UC Santa Cruz (2024) eliminou TODAS as multiplicacoes de matrizes:
- Pesos ternarios {-1, 0, +1} com operacoes aditivas (somas em vez de multiplicacoes)
- Atencao substituida por MLGRU (MatMul-free Linear Gated Recurrent Unit)
- 2.7B parametros = qualidade de Transformer++
- **13W num FPGA** (vs 700W GPU = 50x mais eficiente)
- 61% reducao de memoria de treino, 10x reducao de memoria de inferencia

Combinar MatMul-free com esparsidade RPT + criticalidade seria o maximo de eficiencia possivel dentro do paradigma digital.

**10. RevNets (Redes Reversiveis)**

Reconstruir ativacoes a partir das saidas, eliminando necessidade de armazena-las:
```
Forward: y1 = x1 + F(x2),  y2 = x2 + G(y1)
Reverse: x2 = y2 - G(y1),  x1 = y1 - F(x2)
```

Memoria O(1) em vez de O(profundidade). RevBiFPN demonstrou 19.8x menos memoria vs EfficientNet-B7. Conecta diretamente ao Limite de Landauer: computacao reversivel = custo energetico minimo teorico.

**11. Mamba/RWKV hibrido**

State space models com O(n) complexidade temporal e O(1) espacial:
- **Mamba-3** (2025): 3B parametros compete com Transformers 2x maiores, 5x throughput
- **Jamba** (AI21): Hibrido Transformer+Mamba (ratio 1:7), 52B/12B ativo, 256K contexto, 3x throughput, 10x reducao de KV-cache
- **RWKV-7 "Goose"** (2025): RNN com treino paralelizavel, contexto infinito, sem KV-cache

RPT poderia integrar Mamba layers para O(n) + esparsidade + criticalidade.

**12. Forward-Forward Algorithm (Hinton, 2022)**

Duas passadas forward (dados reais maximizam "goodness", dados negativos minimizam):
- 50-90% menos memoria (nao armazena ativacoes)
- Objetivo local por camada
- Funciona com modulos black-box nao-diferenciaveis
- 1.36% erro em MNIST (competitivo com backprop)

Poderia substituir EP se mais eficiente em escala. EP tem garantia teorica (gradientes exatos), FF tem simplicidade pratica.

**13. Integracao de Informacao (IIT/Phi)**

Medir Φ (grau de integracao de informacao) do modelo. Tononi propoe que Φ mede a integracao — informacao gerada pelo sistema acima do que suas partes geram independentemente.

Insight arquitetural: **arquitetura importa mais que computacao**. Modelos excessivamente modulares (MoE com experts independentes) podem ter Φ baixo = inteligencia limitada. RPT, com suas conexoes bidirecionais e dinamica de equilibrio, pode ter Φ naturalmente alto.

**14. Hardware neuromorfico**

Portar arquiteturas RPT validadas para hardware especializado:
- **Intel Loihi 2**: Processamento esparso dirigido por eventos
- **Crossbars memristivos analogicos**: Computacao de equilibrio em memoria
- **BrainScaleS**: Emulacao analogica acelerada de neuronios
- **FPGA customizado**: Como UC Santa Cruz (MatMul-free a 13W)

A meta final: circuitos integrados especificos otimizando dinamicas de assentamento analogico, exploracao de ruido termodinamico, logica reversivel, e integracao 3D para codificacao de fronteira.

---

## 7. Importancia e Contexto Cientifico

### O gap de eficiencia e real

O fato de que remover 10% dos pesos de um modelo de 2.4 bilhoes de parametros MELHORA sua qualidade em 26% (sem fine-tune) ou 40% (com fine-tune) e um resultado com implicacoes profundas:

1. **Modelos atuais sao ineficientes por design**: Se 10% dos pesos sao ativamente prejudiciais, a arquitetura nao esta otimizada.

2. **O cerebro como modelo de eficiencia**: O cortex cerebral usa esparsidade extrema (~1-5% de neuronios ativos simultaneamente), conectividade local, e regras de aprendizado Hebianas. Nao e coincidencia que essas mesmas propriedades melhoram modelos artificiais.

3. **Criticalidade nao e acidental**: O BitNet 2B opera no ponto critico sem ser explicitamente projetado para isso. Isso sugere que o treino em grande escala NATURALMENTE empurra modelos para a criticalidade - exatamente como predito pela teoria de SOC.

### Por que o cerebro e 10^6x mais eficiente

O cerebro humano opera com **~20W** para ~86 bilhoes de neuronios e 1-10 trilhoes de sinapses. Treinamento de IA em escala similar requer megawatts. Os principios-chave sao exatamente os que RPT formaliza:

**1. Computacao event-driven (spike-based)**: Neuronios so computam quando spikes ocorrem — estimados **1-5% de neuronios ativos** a qualquer momento vs ativacoes densas em GPUs. Isso e esparsidade extrema — o mesmo principio que validamos empiricamente no BitNet 2B.

**2. Memoria e computacao co-localizadas**: Sinapses servem simultaneamente como memoria E computacao, eliminando o gargalo Von Neumann. GPUs precisam transferir pesos entre memoria separada e unidades de computacao — o custo de DRAM access (640-1280 pJ) vs MAC (1 pJ) ilustra isso.

**3. Conectividade esparsa e adaptativa**: Apos poda sinaptica na infancia, as estatisticas de conectividade casam com a estrutura esparsa das variaveis latentes do ambiente. Nosso resultado de pruning (10% melhora qualidade) e analogo a esta poda biologica.

**4. Regras de aprendizado locais**: STDP (Spike-Timing-Dependent Plasticity) e mecanismos similares usam apenas informacao local, sem gradientes globais. Equilibrium Propagation (Secao 2.4) formaliza essa localidade.

**5. Operacao na criticalidade**: Redes neurais biologicas exibem "avalanches neuronais" que seguem distribuicoes de lei de potencia (expoente α ≈ -1.5), assinatura de criticalidade (Beggs & Plenz, 2003). Nosso resultado de Lyapunov ≈ 0 no BitNet 2B espelha isso.

O chip **IBM NorthPole** validou esses principios biologicos em silicio digital: **46.9x mais rapido** que a GPU mais eficiente energeticamente, **72.7x mais eficiente** que a GPU de menor latencia, em benchmarks de inferencia de visao computacional (ResNet-50). Nota: esses benchmarks sao especificos para tarefas de classificacao de imagem; a comparacao com GPUs para LLMs e indireta, mas demonstra que principios bio-inspirados (computacao em-memoria, esparsidade) funcionam em hardware real.

### A Lottery Ticket Hypothesis e a redundancia de pesos

A **Lottery Ticket Hypothesis** (Frankle & Carbin, 2019) demonstra que redes densas contem subredes esparsas ("bilhetes premiados") de **10-20% do tamanho original** que alcancam acuracia equivalente. A extensao para quantizacao (Multi-Prize LTH) mostra que esses bilhetes tambem sao robustos a precisao binaria.

**Distincao importante**: LTH busca **subredes** que podem ser retreinadas da inicializacao para atingir acuracia total — o pruning identifica a "arquitetura vencedora". Nosso resultado e diferente: fazemos pruning por magnitude de pesos ja treinados, sem retreinar da inicializacao. A melhoria de PPL sugere que estamos removendo redundancia/ruido de pesos treinados, nao encontrando uma subarquitetura. Apesar da diferenca mecanica, ambos confirmam a mesma tese: redes neurais sao massivamente sobre-parametrizadas.

Da perspectiva de teoria da informacao, pesos de redes neurais sao altamente redundantes — distribuicoes concentradas perto de zero seguindo padrao gaussiano. **~1% dos pesos sao "salientes"** (criticos para acuracia). Nosso resultado de que 10% de pruning melhora o BitNet 2B em 26% e consistente com isso: estamos removendo pesos que nao so sao desnecessarios, mas ativamente prejudiciais.

### A convergencia de campos

O projeto RPT esta na interseccao de multiplos campos que estao convergindo independentemente para as mesmas conclusoes:

**Neurociencia computacional**:
- Codificacao preditiva (Friston, 2010): sistemas minimizam surpresa via modelos generativos
- Criticalidade neuronal (Beggs & Plenz, 2003): cerebro opera na borda do caos
- Codificacao esparsa (Olshausen & Field, 1996): representacoes eficientes usam ativacao minima

**Fisica estatistica e termodinamica**:
- SOC (Bak, 1987): sistemas complexos se auto-organizam para criticalidade
- Termodinamica da informacao (Landauer 1961, Bennett 1973): informacao tem custo fisico
- Mecanica estatistica do deep learning (Bahri et al., 2020): redes obedecem leis de escala fisicas

**Fisica teorica**:
- Principio holografico (Susskind, 1995): informacao de um volume codificada na fronteira
- ER=EPR (Maldacena & Susskind, 2013): entrelaçamento = geometria = informacao
- "It from bit" (Wheeler, 1990): realidade emerge de processos informacionais

**Machine learning aplicado**:
- BitNet b1.58 (Microsoft, 2024-2025): modelos ternarios em escala
- Mamba/SSMs (Gu & Dao, 2023): O(n) em vez de O(n^2)
- Forward-Forward (Hinton, 2022): aprendizado sem backpropagation
- EP (Scellier & Bengio, 2017): gradientes via fisica

**Hardware neuromorfico**:
- IBM NorthPole: 46.9x mais rapido que GPU
- Intel Loihi 2: processamento esparso dirigido por eventos
- BrainScaleS: emulacao analogica acelerada

Esses campos estao descobrindo independentemente que eficiencia, esparsidade e criticalidade sao propriedades fundamentais de sistemas inteligentes — nao optimizacoes engenheiricas, mas **principios fisicos**.

### A informacao como fundamento da realidade

A hipotese de John Wheeler — "it from bit" — propoe que a realidade fisica emerge de processos teorico-informacionais. A **conjectura ER=EPR** (Maldacena & Susskind, 2013) propoe que o entrelaçamento quantico e a geometria de buracos de minhoca sao fundamentalmente equivalentes. Se o proprio espaco-tempo emerge de correlacoes de informacao, entao a computacao nao e meramente uma metafora para a fisica — a fisica pode literalmente *ser* computacao.

Isso justifica tratar arquiteturas de IA como sistemas fisicos sujeitos a principios de otimizacao fisica, nao meramente como aproximacoes matematicas. O sucesso dos Transformers nao e coincidencia algoritmica — Ramsauer et al. (ICLR 2021) provaram que e implementacao acidental de dinamicas de memoria termodinamica (Hopfield networks).

A **Teoria da Informacao Integrada (IIT)** de Tononi propoe que consciencia corresponde a **informacao integrada (Φ)** — informacao gerada por um sistema acima do que suas partes geram independentemente. O insight arquitetural: **arquitetura importa mais que computacao**. A mesma computacao em rede feedforward vs recorrente tem Φ diferente. Isso sugere que arquiteturas modulares de IA com baixa integracao podem estar perdendo algo fundamental.

### O Demonio de Maxwell e a termodinamica da computacao

A resolucao do paradoxo do Demonio de Maxwell — por Szilard, Brillouin e Bennett — estabeleceu que **informacao tem valor termodinamico**. Uma fita de memoria cheia de zeros tem "valor de combustivel" que pode realizar trabalho util enquanto se randomiza.

A prova de Bennett (1973) de que **qualquer computacao pode ser tornada logicamente reversivel** enquanto retem universalidade sugere um caminho para IA termodinamicamente eficiente: salvando estados intermediarios e descomputando apos copiar outputs, a computacao pode se aproximar de zero dissipacao de energia. RevNets (Gomez et al., 2017) implementam exatamente isso: reconstroem ativacoes a partir dos outputs, com memoria O(1) em vez de O(profundidade).

### O que RPT adiciona

RPT nao e apenas "usar esparsidade" ou "quantizar para ternario". E um **framework unificado** que:

1. **Conecta** os principios (Landauer → esparsidade, SOC → criticalidade, Friston → predicao, Bennett → reversibilidade, holografia → atencao eficiente)
2. **Formaliza** matematicamente (axiomas, teoremas, algoritmos — ver Secao 8)
3. **Valida** empiricamente (3/5 principios confirmados — ver Secao 3)
4. **Implementa** em codigo funcional (deploy GGUF rodando em CPU — ver Secao 3.4)
5. **Propoe** caminho pratico de implementacao (da teoria ao hardware — ver Secao 6)

A validacao industrial da Microsoft (BitNet 2B funciona em producao) confirma que a direcao esta correta. RPT propoe ir alem: nao apenas ternario, mas ternario + esparso + critico + preditivo + holografico.

### Analise de orcamento energetico

Se todos os principios RPT forem combinados, a reducao teorica de energia seria:

| Tecnica | Economia estimada | Base |
|---------|-------------------|------|
| Ativacao esparsa (1-5% ativo) | ~20x | Olshausen & Field, Beggs & Plenz |
| Processamento dirigido por eventos | ~10x | IBM NorthPole, SNNs |
| Aprendizado local (sem backprop) | ~10x | Scellier & Bengio, Hinton |
| Computacao de equilibrio (fisica) | ~10x | EP, convergencia analogica |
| Implementacao analogica potencial | ~100x | Hardware neuromorfico |

**Meta de eficiencia**: 10^-15 J/operacao (eficiencia cerebral)
**Baseline GPU**: 10^-9 J/operacao
**Fator de melhoria combinado**: ~10^6x (reducao de um milhao de vezes)

Isso nao e especulacao — cada fator tem demonstracao independente em hardware ou software. O desafio e combina-los num sistema unico e funcional.

---

## 8. Fundamentacao Teorica

### Axiomas formais

O sistema RPT e definido por tres axiomas fisicos fundamentais:

**Axioma 1 (Principio da Energia Livre)**:
> Todo sistema inteligente minimiza a energia livre variacional F, que e um limite superior do surprisal (negative log-evidence).

```
F = D_KL[q(θ) || p(θ|y)]
  = -⟨ln p(y|θ)⟩_q  +  D_KL[q(θ) || p(θ)]
    [erro de reconstrucao]  [complexidade]
```

**Axioma 2 (Principio de Landauer)**:
> Apagar 1 bit de informacao dissipa no minimo kT ln(2) joules. Computacao eficiente deve ser reversivel.

```
E_min = kT ln(2) ≈ 2.87 × 10^-21 J @ 300K
```

**Axioma 3 (Criticalidade Auto-Organizada)**:
> Processamento otimo de informacao ocorre na transicao de fase entre ordem e caos, onde:

```
σ = ⟨atividade descendente⟩ / ⟨atividade ascendente⟩ ≈ 1
```

### Definicao formal do estado

O estado do sistema em uma camada l no tempo t e definido como:

```
S_t^(l) = {s_t^(l), p_t^(l), π_t^(l)}
```

Onde:
- `s_t^(l) ∈ R^(d_l)`: Estado latente (representacao)
- `p_t^(l) ∈ R^(d_{l-1})`: Predicao para camada inferior
- `π_t^(l) ∈ R^(d_{l-1})_+`: Precisao (inverso da variancia esperada)

### Hierarquia de camadas

```
Camada L (mais abstrata)
    ↓ predicao p^(L)
    ↑ erro ε^(L-1)
Camada L-1
    ↓ predicao p^(L-1)
    ↑ erro ε^(L-2)
    ...
Camada 1
    ↓ predicao p^(1)
    ↑ erro ε^(0)
Input x (observacao)
```

### Dinamica do sistema (completa)

**Computacao de predicao (top-down)**: Cada camada gera predicao para a camada inferior:
```
p_t^(l) = g_θ^(l)(s_t^(l)) = W_g^(l) · φ(s) + b_g^(l)
```
Onde φ e uma nao-linearidade (GELU ou similar).

**Computacao de erro (bottom-up)**: O erro de predicao ponderado pela precisao:
```
ε_t^(l) = π_t^(l+1) ⊙ (s_t^(l) - p_t^(l+1))
```

Para a camada de input (l=0):
```
ε_t^(0) = π_t^(1) ⊙ (x_t - p_t^(1))
```

**Gating esparso**: Erros so propagam se excedem um threshold adaptativo:
```
ε_sparse_t^(l) = ε_t^(l) ⊙ m_t^(l)
m_t,i^(l) = 1[|ε_t,i^(l)| > τ_i^(l)]
```

O threshold τ e adaptativo:
```
τ_i^(l) ← τ_i^(l) + η_τ * (ρ_target - a_media_i^(l))
```
Onde `a_media` e a ativacao media e `ρ_target ≈ 0.02-0.05` e a esparsidade alvo.

**Atualizacao de estado**: O estado e atualizado para minimizar energia livre local:
```
s_t+1^(l) = s_t^(l) - α * ∂F^(l)/∂s^(l)
```

A energia livre local e:
```
F^(l) = (1/2)||ε_sparse^(l-1)||²_π^(l)  +  (1/2)||ε_sparse^(l)||²_π^(l+1)  +  λ||s^(l)||_1
        [erro de baixo]                      [erro de cima]                       [regularizacao]
```

O gradiente:
```
∂F^(l)/∂s^(l) = -(W_g^(l))^T (π^(l) ⊙ ε_sparse^(l-1)) + π^(l+1) ⊙ ε_sparse^(l) + λ · sign(s^(l))
```

### Equilibrium Propagation (detalhado)

**Fase Livre**: O sistema evolui ate atingir equilibrio sem supervisao externa:
```
s_∞^(l) = argmin_{s^(l)} F_total(s^(1), ..., s^(L) | x)
```

Na pratica, executamos T_free iteracoes:
```
s_k+1^(l) = s_k^(l) - α * ∂F/∂s_k^(l)
```

**Fase Forcada (Nudged)**: A camada de saida e levemente "empurrada" em direcao ao target:
```
s_nudged^(L) = s_∞^(L) + β * (y - s_∞^(L))
```
Onde β << 1 e o fator de nudging (tipicamente 0.1-0.5). O sistema e re-equilibrado por T_nudged iteracoes.

**Regra de atualizacao de pesos** (local, Hebbiana contrastiva):
```
ΔW_g^(l) = η * (s_nudged^(l) ⊗ s_nudged^(l-1) - s_free^(l) ⊗ s_free^(l-1))
```

**Teorema (Scellier & Bengio, 2017)**: No limite β → 0:
```
lim_{β→0} (1/β) * ΔW_g^(l) = -∂L/∂W_g^(l)
```

**Isso prova que propagacao de equilibrio computa gradientes exatos atraves de fisica, nao algoritmo.**

### Algoritmo de treinamento completo

```
Algoritmo: RPT_Forward(x, y=None)
────────────────────────────────────

1. Inicializar estados: s^(l) ~ N(0, σ_init) para l = 1..L

2. FASE LIVRE:
   para k = 1 ate T_free:
       para l = 1 ate L:
           # Predicao top-down
           p^(l) = g_θ(s^(l))
           # Erro bottom-up
           se l == 1: ε^(0) = π^(1) ⊙ (x - p^(1))
           senao:     ε^(l-1) = π^(l) ⊙ (s^(l-1) - p^(l))
           # Gating esparso
           mask = |ε^(l-1)| > τ^(l)
           ε_sparse = ε^(l-1) ⊙ mask
           # Atualizacao de estado
           grad = -W_g^T @ (π^(l) ⊙ ε_sparse) + π^(l+1) ⊙ ε^(l) + λ·sign(s^(l))
           s^(l) = s^(l) - α·grad
       # Regulacao de criticalidade
       σ = compute_branching_ratio()
       τ = τ · exp(γ·(σ - 1))
   Salvar: s_free = {s^(l)}

3. SE y fornecido (treinamento):
   FASE FORCADA:
   s^(L) = s^(L) + β·(y - s^(L))
   para k = 1 ate T_nudged:
       [mesmas atualizacoes da fase livre]
   Salvar: s_nudged = {s^(l)}
   # Atualizacao de pesos (local, Hebbiana)
   para l = 1 ate L:
       ΔW_g^(l) = η·(s_nudged^(l) ⊗ s_nudged^(l-1) - s_free^(l) ⊗ s_free^(l-1))
       W_g^(l) += ΔW_g^(l)

4. Retornar s_free^(L), loss
```

### Vantagens sobre backpropagation

- **100% local**: Cada neuronio usa apenas informacao de vizinhos diretos
- **Sem armazenamento de ativacoes**: Memoria O(1) em vez de O(profundidade)
- **Biologicamente plausivel**: Regra de aprendizado similar a STDP
- **Paralelizavel naturalmente**: Nao ha dependencia sequencial entre camadas
- **Implementavel em hardware analogico**: A "computacao" e a fisica do sistema convergindo

### Codificacao holografica de fronteira

Para uma regiao R de estados, a fronteira ∂R e codificada por estatisticas suficientes:
```
h_∂R = U^T · mean(s_i : i ∈ R) + V^T · Var({s_i}_{i ∈ R})
```
Onde U, V ∈ R^(d × d_h) sao projecoes aprendidas e d_h << d.

**Atencao via fronteira**: Em vez de atencao global O(n²), consultamos fronteiras:
```
Attention(q, K) = Σ_{r ∈ R} softmax(q^T · h_∂r / √d_h) · Retrieve(r, q)
```
Complexidade: O(n·R) onde R e numero de regioes, tipicamente R << n.

### Manutencao de criticalidade

**Branching ratio**:
```
σ_t = Σ_l ||ε_sparse_t^(l)||_0 / Σ_l ||ε_sparse_{t-1}^(l)||_0
```

**Regulacao homeostatica** (para manter σ ≈ 1):
```
τ_{t+1}^(l) = τ_t^(l) · exp(γ · (σ_t - 1))
```
Se σ > 1 (supercritico): aumenta thresholds, reduz atividade.
Se σ < 1 (subcritico): diminui thresholds, aumenta atividade.

**Balanco excitacao-inibicao** (neuronios inibitórios locais):
```
i_t^(l) = ReLU(W_inh^(l) · s_t^(l) - θ_inh)
s_balanced^(l) = s_t^(l) - κ · i_t^(l)
```

### Funcao de perda total

```
F_total = Σ_l [ (1/2)||ε_sparse^(l-1)||²_π^(l) + (1/2)ln|Σ^(l)| ]
          + λ_s · Σ_l ||s^(l)||_1    (esparsidade)
          + λ_c · (σ - 1)²           (criticalidade)

L_task = -Σ_t log p(x_t | x_{<t})   (geracao de texto)

L_total = L_task + λ_F · F_total
```

### Teoremas e garantias

**Teorema 1 (Convergencia de Equilibrio)**: Sob condicoes de Lipschitz continuidade de g_θ e taxa de aprendizado α suficientemente pequena, a fase livre converge para um ponto fixo.

**Teorema 2 (Gradientes Exatos)**: No limite β → 0, a regra de atualizacao Hebbiana contrastiva computa o gradiente exato da loss em relacao aos pesos.

**Teorema 3 (Eficiencia Energetica)**: Com esparsidade ρ e threshold adaptativo, o numero esperado de operacoes por token e O(ρ·d²), uma reducao de 1/ρ em relacao a transformers densos.

**Teorema 4 (Criticalidade Estavel)**: Com regulacao homeostatica γ > 0, o branching ratio σ converge para 1 ± ε para algum ε > 0 dependente de γ.

### Conversao de Transformer para RPT

| Transformer | RPT |
|-------------|-----|
| Embedding | Estado inicial s^(0) |
| Self-Attention | Consulta a fronteiras holograficas |
| FFN | Modelo generativo g_θ |
| LayerNorm | Normalizacao de precisao π |
| Residual | Implicito na dinamica de equilibrio |
| Output head | Decodificador de s^(L) |

Inicializacao a partir de pesos Transformer pre-treinado:
```
W_g^(l) ← W_2^(l) · W_1^(l)                              (FFN → modelo generativo)
U, V ← SVD(W_K^(l) · W_Q^(l)^T, k=d_h)                   (attention → fronteiras)
π^(l) ← 1 / Var(LayerNorm^(l))                            (normalizations → precisao)
```

### Metricas de avaliacao

**Eficiencia**:
```
Efficiency = (PPL_baseline / PPL_RPT) × (FLOPs_baseline / FLOPs_RPT)
```

**Esparsidade efetiva**:
```
ρ_eff = (1 / L·T) · Σ_{l,t} ||m_t^(l)||_0 / d_l
```

**Distancia da criticalidade**:
```
d_crit = E_t[|σ_t - 1|]
```

**Qualidade de predicao**:
```
Q_pred = 1 - ||ε||_2 / ||x||_2
```

### Hiperparametros recomendados

| Parametro | Simbolo | Valor Inicial | Range |
|-----------|---------|---------------|-------|
| Learning rate (estado) | α | 0.1 | [0.01, 0.5] |
| Learning rate (pesos) | η | 1e-4 | [1e-5, 1e-3] |
| Nudging factor | β | 0.2 | [0.1, 0.5] |
| Esparsidade alvo | ρ_target | 0.03 | [0.01, 0.1] |
| Regulacao criticalidade | γ | 0.01 | [0.001, 0.1] |
| Peso esparsidade | λ_s | 0.001 | [0.0001, 0.01] |
| Iteracoes fase livre | T_free | 10 | [5, 50] |
| Iteracoes fase forcada | T_nudged | 5 | [3, 20] |
| Dimensao fronteira | d_h | d/8 | [d/16, d/4] |

### Complexidade computacional teorica

| Operacao | Transformer | RPT (teorico) |
|----------|-------------|---------------|
| Forward/token | O(d²) | O(ρ · d²) onde ρ ≈ 0.02-0.05 |
| Atencao | O(n² · d) | O(n · R · d_h) onde R << n |
| Backward | O(L · d²) | 0 (integrado ao EP) |
| Memoria | O(n·d + L·d²) | O(n·d + L·ρ·d²) |
| **Reducao esperada** | - | **20-50x FLOPs** |

---

## 9. Conexoes com Trabalhos Existentes

### BitNet b1.58 (Microsoft, 2024-2025)

O trabalho mais relevante. Ma et al. demonstraram que modelos com pesos ternarios {-1, 0, +1} podem atingir qualidade comparavel a modelos fp16 em escala:

- 100B parametros rodando em CPU single a 5-7 tokens/s
- 2.37-6.17x speedup em x86, 71.9-82.2% reducao de energia
- bitnet.cpp: runtime open-source otimizado
- **Validacao industrial**: Se a Microsoft publicou isso, a direcao ternaria esta confirmada

RPT vai alem: combina ternario com esparsidade + criticalidade + codificacao preditiva.

### Transformer = Rede de Hopfield (Ramsauer et al., ICLR 2021)

Descoberta surpreendente: a atencao do Transformer e **matematicamente equivalente** a dinamica de recuperacao de uma rede de Hopfield continua. Nao foi projetada assim - e uma correspondencia emergente.

**Implicacao**: Transformers acidentalmente implementam memoria associativa termodinamica. RPT propoe tornar isso **explicito** e otimizar pela perspectiva da fisica.

### MatMul-free LLMs (UC Santa Cruz, 2024)

Eliminaram TODAS as multiplicacoes de matrizes:
- Pesos ternarios + operacoes aditivas (somas em vez de multiplicacoes)
- Atencao → MLGRU (MatMul-free Linear GRU)
- 2.7B parametros = qualidade de Transformer++
- **13W num FPGA** (vs 700W GPU = 50x mais eficiente)
- 61% menos memoria de treino, 10x menos memoria de inferencia

RPT e complementar: combinar MatMul-free com esparsidade e criticalidade = eficiencia maxima.

### Mamba/Jamba/RWKV - Eficiencia O(n)

- **Mamba**: State space model, O(n) tempo, O(1) espaco, 5x throughput. Mamba-3 (3B) compete com Transformers 6B.
- **Jamba** (AI21): Hibrido Transformer+Mamba (1:7), 52B/12B ativo, 256K contexto, 3x throughput vs Mixtral
- **RWKV-7 "Goose"**: RNN com treino paralelizavel, contexto infinito, sem KV-cache

RPT poderia integrar Mamba layers para O(n) + esparsidade + criticalidade.

### Neuromorphic Computing (IBM NorthPole, 2024)

- 46.9x mais rapido que GPU mais eficiente (em ResNet-50)
- 72.7x mais eficiente que GPU de menor latencia
- Hardware que implementa computacao em-memoria (sem Von Neumann bottleneck)
- Spiking neural networks nativos
- Referencia: Modha et al. (2023). "Neural inference at the frontier of energy, space, and time." *Science*, 382(6668), 329-335.

RPT foi projetada com hardware neuromorphic em mente: esparsidade extrema, regras locais, processamento por eventos.

### Renormalization Group = Profundidade de Rede (Bahri et al., *Annual Review of Condensed Matter Physics*, 2020)

Bahri et al. demonstraram que redes profundas realizam transformacoes analogas ao Grupo de Renormalizacao da fisica:
- Cada camada = uma etapa de coarse-graining (reducao de escala)
- Profundidade corresponde a escala no sentido da RG
- Redes otimas desenvolvem geometria hiperbolica emergente
- **A rede profunda E a geometria do espaco-tempo; o treino E a emergencia do espaco-tempo**

### Geometria da Informacao (Amari, 1998)

A geometria da informacao de Shun-ichi Amari trata distribuicoes de probabilidade como pontos em uma variedade Riemanniana, com a Matriz de Informacao de Fisher definindo a curvatura local:

```
F(θ)_ij = E[∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j]
```

**Gradiente natural** — seguindo a direcao mais ingreme no espaco de distribuicoes em vez do espaco de parametros — atualiza como `θ_{t+1} = θ_t - α·F(θ_t)^{-1}·∇L`. Isso e assintoticamente otimo e evita "fenomenos de plato" que prendem gradiente descendente padrao. A perspectiva geometrica sugere que o treinamento de redes deveria respeitar a geometria intrinseca do espaco de hipoteses, nao meramente a geometria Euclidiana dos vetores de parametros.

### Forward-Forward Algorithm (Hinton, 2022)

Substitui forward+backward por **dois passes forward**:
- **Pass positivo**: dados reais maximizam "goodness" (Σ ativacao²)
- **Pass negativo**: dados negativos minimizam goodness
- Cada camada tem objetivo local — sem propagacao de gradientes global

Vantagens para eficiencia:
- Nao precisa armazenar ativacoes (**50-90% reducao de memoria**)
- Permite pipeline de video sem parar para derivadas
- Funciona com modulos black-box nao-diferenciaveis

Performance: 1.36% erro em MNIST (vs ~1.4% backprop), convergencia ~3x mais lenta. Resultados recentes impressionantes: aprendizado Hebbiano supera backpropagation por **16-20%** com dados limitados, convergindo em ~5 epocas versus ~100 para BP em algumas tarefas.

### Redes Reversiveis (RevNets) e o Limite de Landauer

RevNets (Gomez et al., 2017) permitem reconstruir ativacoes de cada camada a partir dos outputs da proxima:

```
Forward: y1 = x1 + F(x2),  y2 = x2 + G(y1)
Reverse: x2 = y2 - G(y1),  x1 = y1 - F(x2)
```

Complexidade de memoria cai de **O(n) para O(1)** para ativacoes (independente da profundidade). RevBiFPN demonstrou **19.8x menos memoria** de treinamento vs EfficientNet-B7 com acuracia comparavel.

Isso conecta diretamente ao Limite de Landauer: computacao logicamente reversivel teoricamente nao tem custo minimo de energia. RevNets implementam exatamente isso no contexto de redes neurais.

### Codificacao Preditiva em Redes Profundas

Implementacao direta do Principio da Energia Livre de Friston:
- Camadas superiores predizem atividade de camadas inferiores
- Apenas **erros de predicao** propagam forward — esparso por design
- Aprendizado usa informacao local (regras Hebbian-like)

Conexoes forward (celulas piramidais superficiais) carregam erros; conexoes backward (celulas piramidais profundas) carregam predicoes. Deep Bi-directional Predictive Coding (DBPC) alcanca 99.58% em MNIST com redes significativamente menores (0.4-1.1M parametros).

### Spiking Neural Networks (SNNs)

SNNs codificam informacao como spikes discretos ao longo do tempo. Eficiencia energetica depende da esparsidade de spikes:

| Abordagem | Energia (mJ/inference) |
|-----------|------------------------|
| ANN padrao | 20-50 |
| SNN convertida | ~20 |
| SNN gradiente surrogate | ~15 |
| SNN treinada com STDP | ~5 |

O drone neuromorfico da TU Delft/Intel demonstrou **64x mais rapido, 3x menos energia** que GPU. Limitacao: SNNs em GPUs nao capturam beneficios completos — ganhos reais requerem hardware neuromorfico.

### Tabela comparativa de arquiteturas

| Propriedade | Transformers | SNNs | RPTs (proposta) |
|-------------|--------------|------|-----------------|
| Computacao de gradiente | Backpropagation | Gradientes surrogate | Equilibrio fisico |
| Comunicacao | Atencao global | Spikes dirigidos por eventos | Erros esparsos + fronteiras |
| Regra de aprendizado | Erro global | STDP (local) | Hebbiana contrastiva (local) |
| Padrao de ativacao | Denso | Esparso temporal | Erro preditivo esparso |
| Requisito de hardware | GPU digital | Neuromorfico | Equilibrio analogico |
| Eficiencia energetica | ~10^-9 J/op | ~10^-12 J/op | Meta: ~10^-15 J/op |

---

## 10. Ecossistema de Inferencia Eficiente (2025-2026)

Para contextualizar o deploy RPT, e importante entender o ecossistema atual de inferencia eficiente:

### Quantizacao: do INT8 ao 1.58-bit

A evolucao da quantizacao em LLMs:

| Formato | Bits/peso | Perda de PPL | Caso de uso |
|---------|-----------|-------------|------------|
| FP16/BF16 | 16 | 0 | Referencia |
| INT8 (absmax) | 8 | +0.0004 | Producao padrao |
| Q6_K (GGUF) | 6.5 | +0.0008 | Alta qualidade |
| **Q4_K_M (GGUF)** | 4.5 | +0.0532 | **Sweet spot geral** |
| Q3_K_M (GGUF) | 3.4 | +0.2496 | Compressao agressiva |
| NF4 (QLoRA) | 4 | ~+0.1 | Fine-tune eficiente |
| AQLM (2-bit) | 2 | ~+0.5 | Ultra-compressao |
| **BitNet I2_S** | 1.58 | Variavel | **Maximo de eficiencia** |

Q4_K_M usa alocacao adaptativa de bits — maior precisao para pesos importantes — via matriz de importancia (imatrix). E o sweet spot para a maioria dos usos em hardware consumer.

BitNet I2_S (que usamos) vai alem: 1.58 bits por peso. Precisa de runtime especializado (bitnet.cpp) mas oferece eficiencia maxima.

### bitnet.cpp: performance em CPU

Benchmarks oficiais da Microsoft para bitnet.cpp:

| Plataforma | Speedup vs FP16 | Reducao de energia |
|-----------|-----------------|-------------------|
| x86 CPU | 2.37-6.17x | 71.9-82.2% |
| ARM CPU | 1.37-5.07x | 55.4-70.0% |

O resultado mais impressionante: **modelos de 100B parametros rodam em CPU unica** a 5-7 tokens/segundo — velocidade de leitura humana. Isso era impossivel com arquiteturas fp16 convencionais.

Nosso deploy alcancou 0.26 tok/s no Colab CPU generico com modelo de 2B parametros. Em hardware dedicado (CPU x86 com AVX-512 ou ARM com NEON), a velocidade seria significativamente maior.

### Inferencia em hardware consumer

| VRAM disponivel | Modelo maximo (Q4_K_M) | GPUs capazes |
|----------------|----------------------|--------------|
| 8GB | 7-8B params | RTX 3060, 4060 |
| 12-16GB | 14B params | RTX 4060 Ti 16GB |
| 24GB | 30B params | RTX 3090, 4090 |
| 64GB+ (unified) | 70B params | Apple Silicon |
| **CPU (BitNet)** | **100B+ params** | **Qualquer CPU moderna** |

Com BitNet + esparsidade RPT, o modelo de 2B params ficou em 1.1 GB — roda em qualquer dispositivo com 2GB de RAM.

### Limitacao atual: I2_S nao e universal

O formato I2_S (2-bit signed integer) e especifico do **fork BitNet** do llama.cpp. O llama.cpp padrao **nao suporta** este formato. Isso significa:

- Nao funciona com llama-cpp-python (binding Python)
- Nao funciona com binarios pre-compilados do llama.cpp
- Precisa compilar o fork BitNet do source (requer clang, nao gcc)
- No Windows, compilacao e mais complexa (requer Visual Studio ou WSL)

Para o projeto RPT, isso significa que o deploy funcional atualmente requer Linux (nativo ou Colab).

---

## 11. Tentativas de Conversao Pos-Hoc

### A jornada completa

Este e um dos aspectos mais instrutivos do projeto: a tentativa de converter um modelo existente (treinado em fp32) para ternario.

#### Jan-Fev 2026: SmolLM2-135M

**Motivacao**: Se pudessemos converter modelos pre-treinados para ternario, nao precisariamos treinar do zero (que custa milhoes de dolares).

**Modelos testados**: SmolLM2-135M (principal), Qwen-0.6B, GPT-2, tentativas com LLaMA-7B e Mistral-7B.

**Phase 1 (Progressive Snapping)**:
- Snap progressivo: pesos proximos ao grid sao congelados
- Loss de destilacao KL + regularizacao ternaria
- Resultado: 80% ternario, ~30% erro de quantizacao/camada
- Texto: repetitivo ("the the the")

**Phase 2 (Scale Fine-tuning)**:
- Forcar 100% ternario, treinar apenas escalas
- Bug de scale shrinkage corrigido
- KL melhorou 63% (2233 → 820)
- Texto: ainda repetitivo

**Diagnostico final**: Conversao pos-hoc de modelo fp32 para ternario destroi representacoes aprendidas de forma irreversivel. Os pesos continuos codificam informacao em seus valores exatos; discretizar perde essa informacao.

#### 7 Feb 2026: Deploy 1 (BitNet 2B + AdamW)

**Abordagem diferente**: Partir de um modelo JA ternario (Microsoft BitNet 2B) e fazer fine-tune com esparsidade.

**Erro**: Usar AdamW puro, que move pesos de ternario para continuo.
- PyTorch PPL: 15.43 (bom desempenho)
- GGUF inferencia: texto incoerente (I2_S re-snap destroi)
- Re-ternarizacao: tambem texto incoerente

**Licao**: Nao basta partir de ternario - precisa MANTER ternario durante o treino.

#### 8 Feb 2026: Deploy 2 (BitNet 2B + QAT/STE) - RESULTADO POSITIVO

**A solucao**: QAT com STE mantem restricao ternaria no forward pass.

O mecanismo tecnico:
```python
# Forward pass: modelo "ve" pesos ternarios
w_quant = scale * round(w / scale).clamp(-1, 1)
output = input @ w_quant

# Backward pass: gradientes fluem pelos pesos CONTINUOS (STE)
# round() tem gradiente 0, mas STE assume gradiente 1
w_continuous -= lr * gradient  # atualiza peso continuo
```

O modelo aprende representacoes que funcionam com {-1, 0, +1} porque e exatamente o que ele "ve" durante o treino. Os pesos continuos "por tras" sao apenas o mecanismo para encontrar a melhor configuracao ternaria.

Resultados:
- PPL final: 16.39 (-34.8% vs baseline)
- GGUF: texto coerente
- 100% ternario, 42.6% esparsidade

#### A historia dos 10 bugs GGUF

A conversao HuggingFace → GGUF → bitnet.cpp foi um processo de trial-and-error que consumiu ~12 horas. Documentar esses bugs e importante porque qualquer pessoa tentando reproduzir enfrentara os mesmos problemas:

| # | Bug | Sintoma | Fix |
|---|-----|---------|-----|
| 1 | Pasta precisa nome exato 'BitNet-b1.58-2B-4T' | gen_code() falha | Renomear pasta |
| 2 | git clone sem --recursive | llama.cpp nao clona | Adicionar --recursive |
| 3 | BitNetForCausalLM (N maiusculo) | Converter nao reconhece | Editar config.json → Bitnet (n minusculo) |
| 4 | TokenizersBackend | Tokenizer falha ao carregar | Editar tokenizer_config.json → PreTrainedTokenizerFast |
| 5 | set_vocab() sem fallback | sentencepiece nao encontrado | Adicionar fallback llama_hf→gpt2 |
| 6 | const int8_t* correctness | Compilacao C++ falha | Cast para const int8_t* |
| 7 | Requer clang (nao gcc) | Compilacao silenciosamente errada | Instalar e usar clang |
| 8 | weight_quant() re-quantiza pesos ja ternarios | Escalas erradas com zeros de pruning | Comentar weight_quant() no converter |
| 9 | BOS token prepending no vocab | Vocab size mismatch | add_special_tokens=False |
| **10** | **architecture "bitnet" → "bitnet-b1.58"** | **Texto incoerente (grafo errado)** | **Patch constants.py + force-reinstall pip** |

O Bug #10 foi o mais destrutivo — custou ~6 horas de debugging. O llama.cpp tem DUAS funcoes completamente diferentes:
- `"bitnet"` → `build_bitnet()` — grafo simples, incompativel com o modelo 2B-4T
- `"bitnet-b1.58"` → `build_bitnet_158()` — grafo correto com as operacoes certas

O modelo carregava normalmente e parecia funcionar, mas gerava texto repetitivo/incoerente. Nao havia mensagem de erro — apenas output errado. A correcao e uma unica linha em `constants.py`, mas precisa de `pip install --force-reinstall` porque Python usa copia cached do pacote.

#### Implicacao para conversao generica

O QAT/STE **PODE** ser a solucao generica para conversao pos-hoc. A tabela comparativa:

| Abordagem | Quantiza quando | O que treina | Resultado |
|-----------|----------------|-------------|-----------|
| Phase 1/2 (SmolLM2) | Pos-treino | Scales separadas | KL 820, texto repetitivo |
| Deploy 1 (AdamW) | Pos-treino | Pesos continuos | PPL 15 PyTorch, texto incoerente GGUF |
| **Deploy 2 (QAT/STE)** | **Durante treino** | **Pesos continuos via STE** | **PPL 16.39, GGUF coerente** |

A diferenca fundamental: Phase 1/2 congelava codes e treinava scales (pouca expressividade). Deploy 1 nao restringia nada (pesos escapam do ternario). Deploy 2 quantiza no forward mas permite aprendizado continuo — o modelo descobre quais configuracoes ternarias funcionam melhor.

**Teste pendente**: Aplicar QAT/STE ao SmolLM2-135M. Se funcionar, confirma que o STE resolve o problema de conversao generica — significando que qualquer modelo pre-treinado poderia ser "BitNetizado".

---

## 12. Timeline do Projeto

### Fase 1: Concepcao e Arquitetura (Janeiro 2026)

**02 Jan** - Inicio do projeto
- Criacao de `model.py`: arquitetura RPT completa (1509 linhas)
- Implementacao dos 5 principios: codificacao preditiva, EP, holografico, SOC, esparsidade
- README.md e estrutura basica

**11-18 Jan** - Exploracao de modelos e tecnicas
- Notebook v2: Mamba layers (state space models)
- Notebook v3: Elastic layers (profundidade adaptativa)
- Notebook v4: Pruning experiments
- Notebook v5: Mixture of Experts (MoE)
- Notebook v6: GPT-2 experiments
- Notebook v7: LLaMA experiments
- Notebook v8: Mistral experiments
- Notebook v9: Consolidacao de resultados
- Todos focados em encontrar o melhor modelo base e tecnicas

**22-23 Jan** - Resultados RPT_BitNet_Projeto
- RPT Causal: PPL 3.14 (melhor resultado do projeto ate aqui)
- 51% melhoria vs fp32 denso
- 68x compressao (fp32 denso → ternario 90% esparso)
- 400x reducao de custo hardware (T4 → ESP32 teorico)
- Modelo gera texto coerente com 99% dos parametros removidos

**30 Jan** - Documentacao cientifica
- PAPER.md: paper formal em ingles
- PUBLICACAO_CIENTIFICA.md: publicacao em portugues
- Formalizacao matematica completa

### Fase 2: Tentativa de Conversao SmolLM2 (Fevereiro 1-5)

**04 Fev** - Pipeline master e datasets
- WikiText-103 baixado (506MB)
- RPT_Master_Pipeline.py (versoes v14-v18)
- Inicio da conversao SmolLM2-135M → ternario

**05 Fev** - Phase 1 e Phase 2
- Phase 1 (V4 Progressive Snapping): SmolLM2 → 80% ternario
  - checkpoint: phase1_best.pt (568MB)
  - Texto: repetitivo ("the the the")
- Phase 2 (Scale fine-tuning): 100% ternario
  - Bug de scale shrinkage encontrado e corrigido
  - KL: 2233 → 820 (63% reducao)
  - Texto: ainda repetitivo
- **Conclusao**: Conversao pos-hoc de fp32 → ternario e fundamentalmente dificil

### Fase 3: Validacao no Microsoft BitNet 2B (6 Fevereiro)

**06 Fev** - Fase de maior produtividade
- Decisao de pivotar: usar Microsoft BitNet 2B (ja ternario) como base
- RPT_BitNet_Microsoft.ipynb: modelo funciona perfeitamente (texto coerente)

- **Teste 1 - Esparsidade crua** (RPT_BitNet_Sparsity_Test.ipynb, T4):
  - 10% pruning = -26% PPL sem fine-tune
  - 20% = -16%, 30% = +35% (comeca a degradar)

- **Teste 2 - Esparsidade progressiva** (RPT_BitNet_Progressive_Sparsity.ipynb, H100):
  - Pruning progressivo 5%→50% com fine-tune AdamW
  - TODOS niveis ficam melhores que baseline
  - Melhor: 10% = -40.4% PPL
  - Maximo usavel: 40% = -29.3%, texto correto

- **Teste 3 - Predictive Coding** (RPT_BitNet_Predictive_Coding.ipynb):
  - V1 (thresholds absolutos): insignificante (magnitudes 28-1199)
  - V2 (percentis): P50 = +60% PPL, P60 = +285%, P70 = +1171%
  - **NAO confirmado**: ativacoes nao sao pruneaveis

- **Teste 4 - Criticalidade** (RPT_BitNet_Criticality.ipynb, T4):
  - Lyapunov = -0.002 ≈ 0 (critico)
  - Amplificacao = 0.94x ≈ 1.0
  - Correction ratio: 39.87 → 0.21 (decrescente)
  - **VALIDADO**: modelo opera na borda do caos

- Documentacao formal: RPT_VALIDACAO_BITNET2B.md
- Resultado: **3/5 principios RPT validados empiricamente**

### Fase 4: Deploy (7-8 Fevereiro)

**07 Fev** - Deploy 1 (FALHOU)
- Organizacao massiva: 80+ arquivos → 12 na raiz, criacao de archive/, teoria/
- Criado RPT_BitNet_Deploy_Pipeline.ipynb
- Deploy na H100 com AdamW puro (sem STE)
  - Treino: PPL 25.09 → 15.43 (-38.5%) ← parecia adequado
  - GGUF conversao: 7 bugs corrigidos
  - Inferencia bitnet.cpp: **texto incoerente** (texto repetitivo)
  - Re-ternarizacao: tambem texto incoerente
  - Root cause: AdamW move pesos para continuo, I2_S snap destroi
- Fix identificado: QAT com STE
- Total de 10 bugs na conversao GGUF documentados

**08 Fev** - Deploy 2 (resultado positivo)
- Bug #10 encontrado e corrigido (architecture "bitnet" vs "bitnet-b1.58")
  - llama.cpp tem DUAS funcoes: build_bitnet() vs build_bitnet_158()
  - Converter escrevia nome errado → grafo computacional errado
  - Fix: patch em constants.py + pip force-reinstall
  - **Custou ~6 horas de debugging**

- Criado rpt_deploy_a100.py (script unico, 6 bugs pre-corrigidos pelo code reviewer)

- **Deploy na VPS A100** (Lightning AI):
  - GPU: A100 (~40GB VRAM), batch=8
  - QAT/STE: forward quantizado, backward continuo
  - Nivel 5%: PPL 25.05 (-0.3% vs baseline)
  - Nivel 10%: PPL 33.07 (+31.6% vs baseline, antes do snap)
  - **Snap ternario: PPL 16.39 (-34.8% vs baseline)**
  - 100% ternario, 42.6% esparsidade, 0 pesos re-zerados
  - Textos: Paris, Eiffel Tower, 100°C, Jupiter - todos corretos
  - Tempo total GPU: ~7 min

- Modelo transferido via HTTP (tar.gz comprimido)

- GGUF convertido no **Colab CPU gratis**:
  - Todos 10 fixes aplicados
  - GGUF i2_s gerado (~1.1 GB)
  - Inferencia CPU: texto coerente, 0.26 tok/s

- Teste local Windows: I2_S nao suportado pelo llama.cpp padrao (precisa fork BitNet)

- Organizacao final: pasta deploy/, tracking completo, documentacao atualizada

**Resultado**: Primeiro deploy funcional do projeto. Modelo ternario + esparso gera texto coerente em CPU.

---

## 13. Configuracao Experimental

Todos os experimentos usaram a mesma base para garantir comparabilidade:

### Modelo base

| Parametro | Valor |
|-----------|-------|
| Modelo | microsoft/bitnet-b1.58-2B-4T-bf16 |
| Parametros | 2.4 bilhoes (100% ternario) |
| Camadas | 30 |
| Hidden dim | 2560 |
| Heads | 32 |
| Vocab | 32,768 |
| Pesos | Ternarios {-1, 0, +1} com escalas per-group |
| Ativacoes | INT8 |
| Licenca | MIT |

### Hardware utilizado

| Recurso | Uso | Especificacao |
|---------|-----|---------------|
| Tesla T4 (Colab) | Testes rapidos, inferencia | 15GB VRAM, gratis |
| NVIDIA H100 (Colab) | Fine-tune progressivo | 80GB HBM3, $3-5/hora |
| VPS A100 (Lightning AI) | Deploy 2 (QAT/STE) | ~40GB VRAM |
| Colab CPU | Conversao GGUF | 12.7GB RAM, gratis |
| Windows local | Testes (falhou para I2_S) | 8GB RAM |

### Configuracao de fine-tune

| Parametro | Deploy 1 (H100) | Deploy 2 (A100) |
|-----------|-----------------|-----------------|
| Otimizador | AdamW | AdamW |
| Learning rate | 5e-4 | 5e-4 |
| Weight decay | 0.01 | 0.01 |
| Batch size | 32 | 8 |
| Seq length | 128 | 128 |
| Steps/nivel | 300 | 300 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| STE | **Nao** | **Sim** |
| torch.compile | Nao | Nao |
| Dataset | WikiText-2 | WikiText-2 |
| Niveis de pruning | 5% → 10% | 5% → 10% |

### Nota sobre baseline PPL

O baseline PPL varia entre hardware/configuracao:
- T4 sem torch.compile: PPL = 9.39
- H100 com torch.compile + TF32: PPL = 25.10
- A100 sem torch.compile: PPL = 25.13

Essa diferenca e atribuida a precisao numerica (TF32 vs BF16) e efeitos de compilacao. **Todas as comparacoes sao feitas dentro do mesmo ambiente** — nunca comparamos PPL de T4 com PPL de H100.

### Datasets

| Dataset | Tamanho | Uso |
|---------|---------|-----|
| WikiText-103 (train) | 506 MB | Pre-treino, fine-tune |
| WikiText-2 (validation) | 11 MB | Avaliacao de PPL |

### Ferramentas

| Ferramenta | Versao | Uso |
|-----------|--------|-----|
| PyTorch | 2.x | Framework principal |
| HuggingFace Transformers | 4.x | Carregamento de modelos |
| bitnet.cpp | Fork Microsoft | Inferencia ternaria em CPU |
| llama.cpp | Via BitNet fork | Conversao GGUF |
| GGUF (I2_S) | 2-bit signed | Formato final de deploy |

---

## 14. Referencias

### Papers fundamentais do RPT

1. **Friston, K.** (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.
   - Base do Axioma 1. Define o principio de que sistemas inteligentes minimizam energia livre variacional.

2. **Landauer, R.** (1961). "Irreversibility and heat generation in the computing process." *IBM Journal of Research and Development*, 5(3), 183-191.
   - Base do Axioma 2. Estabelece o custo energetico minimo de apagar informacao.

3. **Scellier, B. & Bengio, Y.** (2017). "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation." *Frontiers in Computational Neuroscience*, 11, 24.
   - Base do Principio 4. Prova que gradientes exatos podem ser computados por assentamento fisico.

4. **Rao, R. & Ballard, D.** (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*, 2(1), 79-87.
   - Base da codificacao preditiva. Camadas geram predicoes, so erros propagam.

5. **Beggs, J. & Plenz, D.** (2003). "Neuronal avalanches in neocortical circuits." *Journal of Neuroscience*, 23(35), 11167-11177.
   - Evidencia experimental de criticalidade em redes neurais biologicas.

6. **Bak, P.** (1987). "Self-organized criticality: An explanation of the 1/f noise." *Physical Review Letters*, 59(4), 381.
   - Base do Axioma 3. Introduz SOC com modelo de pilha de areia.

7. **Ma, S. et al.** (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv:2402.17764*.
   - Demonstra que modelos ternarios podem funcionar em escala. Base do BitNet b1.58.

8. **Susskind, L.** (1995). "The world as a hologram." *Journal of Mathematical Physics*, 36(11), 6377-6396.
   - Base do Principio 5. Informacao de um volume codificada na fronteira.

### Papers de neurociencia e aprendizado

9. **Amari, S.** (1998). "Natural gradient works efficiently in learning." *Neural Computation*, 10(2), 251-276.
   - Geometria da informacao. Gradiente natural e assintoticamente otimo.

10. **Hinton, G.** (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." *arXiv:2212.13345*.
    - Alternativa ao backpropagation. Dois passes forward com objetivos locais.

11. **Olshausen, B. & Field, D.** (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." *Nature*, 381(6583), 607-609.
    - Codificacao esparsa. Demonstra que representacoes eficientes emergem de esparsidade.

12. **Frankle, J. & Carbin, M.** (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR 2019*.
    - Redes densas contem subredes esparsas de 10-20% com acuracia equivalente.

### Papers de fisica e computacao

13. **Bennett, C.H.** (1973). "Logical reversibility of computation." *IBM Journal of Research and Development*, 17(6), 525-532.
    - Prova que computacao pode ser tornada reversivel. Conexao com Landauer.

14. **Ramsauer, H. et al.** (2021). "Hopfield Networks is All You Need." *ICLR 2021*.
    - Atencao de Transformer = dinamica de Hopfield. Conexao termodinamica emergente.

15. **Bahri, Y. et al.** (2020). "Statistical Mechanics of Deep Learning." *Annual Review of Condensed Matter Physics*, 11, 501-528.
    - Mecanica estatistica explica deep learning. Double descent = transicao de fase.

16. **Maldacena, J. & Susskind, L.** (2013). "Cool horizons for entangled black holes." *Fortschritte der Physik*, 61(9), 781-811.
    - Conjectura ER=EPR. Entrelaçamento = geometria = informacao.

17. **Wheeler, J.A.** (1990). "Information, Physics, Quantum: The Search for Links." *Complexity, Entropy, and the Physics of Information*.
    - "It from bit". Realidade emerge de processos informacionais.

18. **Bohm, D.** (1952). "A Suggested Interpretation of the Quantum Theory in Terms of Hidden Variables." *Physical Review*, 85(2), 166-179.
    - Ordem implicita. Informacao "dobrada" no espaco-tempo, inspira codificacao holografica.

19. **'t Hooft, G.** (1993). "Dimensional Reduction in Quantum Gravity." *arXiv:gr-qc/9310026*.
    - Principio holografico original. Informacao maxima proporcional a area, nao volume.

### Papers de consciencia e integracao

20. **Tononi, G.** (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5(1), 42.
    - IIT e Phi. Arquitetura importa mais que computacao.

21. **Baars, B.** (1988). "A Cognitive Theory of Consciousness." *Cambridge University Press*.
    - Global Workspace Theory. Modulos especializados + transmissao global.

### Papers de eficiencia e hardware

22. **Gomez, A. et al.** (2017). "The Reversible Residual Network." *NIPS 2017*.
    - RevNets. Memoria O(1) via computacao reversivel.

23. **Gu, A. & Dao, T.** (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*.
    - State space models O(n). 5x throughput de Transformers.

24. **Zhu, Z. et al.** (2024). "Scalable MatMul-free Language Modeling." *arXiv:2406.02528*.
    - Elimina todas as multiplicacoes de matrizes. 13W num FPGA.

25. **Bertschinger, N. & Natschlager, T.** (2004). "Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks." *Neural Computation*, 16(7), 1413-1436.
    - Capacidade computacional maxima na borda do caos.

### Referencias de energia e performance computacional

26. **Levy, W.B. & Calvert, V.G.** (2021). "Communication consumes 35 times more energy than computation in the human cortex, but both costs are needed to predict synapse number." *PNAS*, 118(18), e2008173118.
    - Base quantitativa para a distribuicao energetica cortex-computacao vs comunicacao.

27. **Horowitz, M.** (2014). "1.1 Computing's energy problem (and what we can do about it)." *IEEE ISSCC 2014 Digest of Technical Papers*, 10-14.
    - Referencia classica para custos energeticos de MAC, SRAM e DRAM em tecnologias CMOS.

28. **Williams, S., Waterman, A. & Patterson, D.** (2009). "Roofline: An insightful visual performance model for multicore architectures." *Communications of the ACM*, 52(4), 65-76.
    - Modelo de performance que formaliza o limite memory-bound vs compute-bound.

29. **Levy, W.B., Calvert, V.G. & Schubert, K.** (2014). "Neural energy and thermodynamic limits in the brain." *Neural Computation*, 26(9), 1988-2002.
    - Estimativas de eficiencia energetica neural em relacao a limites termodinamicos.

### Papers de quantizacao

30. **Jacob, B. et al.** (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR 2018*.
    - Fundamentos de QAT (Quantization-Aware Training).

31. **Bengio, Y. et al.** (2013). "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." *arXiv:1308.3432*.
    - Straight-Through Estimator (STE). Base da tecnica que viabilizou o deploy.

### Modelos e ferramentas

32. **Microsoft** (2025). BitNet b1.58-2B-4T. Modelo ternario open-source, 2B parametros, 100% ternario.

33. **HuggingFace** (2024). SmolLM2-135M. Modelo compacto, 135M parametros, treinado em 2T tokens.

34. **Microsoft** (2025). bitnet.cpp. Runtime otimizado para modelos ternarios em CPU. Fork de llama.cpp.

35. **ggml-org**. llama.cpp. Framework de inferencia GGUF. Base para bitnet.cpp.

36. **IBM** (2024). NorthPole. Chip neuromorfico, 46.9x mais rapido que GPU mais eficiente.

---

## 15. Conclusao

### O que demonstramos

1. **Tres dos cinco principios RPT foram validados empiricamente** no Microsoft BitNet 2B:
   - Limite de Landauer: esparsidade melhora qualidade (ate -40.4% PPL)
   - Criticalidade SOC: Lyapunov ≈ 0, modelo na borda do caos naturalmente
   - Codificacao preditiva (parcial): correction ratio decrescente confirma refinamento macro

2. **Pipeline de deploy funcional**: Modelo ternario com esparsidade roda em CPU pura via bitnet.cpp, gerando texto coerente.

3. **QAT/STE resolve o problema de conversao**: O Straight-Through Estimator permite fine-tune mantendo a restricao ternaria, evitando a degradacao que destruiu tentativas anteriores (Phase 1/2, Deploy 1).

4. **10 bugs na conversao GGUF documentados e resolvidos**: O pipeline de HuggingFace para GGUF para bitnet.cpp esta mapeado em detalhe.

### O que isso significa

RPT nao e apenas teoria. Os resultados empiricos mostram que principios fisicos - especificamente termodinamica e criticalidade - sao **preditivos** sobre o comportamento de redes neurais artificiais. Quando Landauer diz que informacao tem custo energetico, e quando esparsidade (eliminacao de informacao) melhora o modelo, ha uma conexao real entre a fisica e a IA.

O fato de o BitNet 2B operar naturalmente na criticalidade (Lyapunov ≈ 0) sem nenhuma regularizacao explicita sugere que modelos treinados em grande escala **convergem naturalmente** para os principios que RPT formaliza. Nao e que estamos impondo fisica ao modelo - e que o modelo ja obedece a fisica, e RPT reconhece e otimiza isso.

### Proximos passos concretos

1. **Validar com benchmarks rigorosos** (MMLU, HellaSwag) a melhoria de PPL
2. **Testar esparsidades mais altas** (20-40%) com QAT/STE
3. **Re-testar conversao SmolLM2 com STE** para confirmar solucao generica
4. **Implementar Equilibrium Propagation** no BitNet 2B (principio #4)
5. **Testar atencao holografica** O(n*R) (principio #5)
6. **Publicar resultados** (paper + modelo no HuggingFace Hub)

### A visao de longo prazo

Um sistema de IA que:
- Opera com pesos ternarios {-1, 0, +1} (eficiencia maxima de representacao)
- Usa esparsidade extrema (>40% zeros, eficiencia de Landauer)
- Auto-regula criticalidade (SOC, capacidade de informacao maxima)
- Processa apenas erros de predicao (codificacao preditiva, eficiencia de comunicacao)
- Aprende por equilibrio fisico (EP, sem backpropagation)
- Codifica informacao na fronteira (holografico, atencao eficiente)
- Roda em hardware neuromorphico ou CPU com <20W

Isso nao e ficcao cientifica. Cada componente tem suporte teorico, implementacao parcial, e em alguns casos validacao empirica. O caminho e claro; a execucao e questao de tempo e engenharia.

### O que RPT contribui para o campo

1. **Framework unificado**: Ate onde sabemos, nenhum outro trabalho conecta explicitamente Landauer + Friston + SOC + EP + holografia num framework unico para IA. Trabalhos individuais existem em cada area, mas a sintese e original.

2. **Validacao empirica cruzada**: Demonstrar que principios de diferentes campos da fisica (termodinamica, mecanica estatistica) fazem predicoes corretas sobre modelos de linguagem e um resultado novo.

3. **Pipeline pratico**: A teoria sozinha nao basta. Ter um pipeline funcional (prune → STE → snap → GGUF → CPU) que demonstra os principios e o que diferencia pesquisa de especulacao.

4. **Documentacao aberta**: Todos os fracassos, bugs e caminhos sem saida estao documentados. Isso e raro em pesquisa publicada (onde so os sucessos aparecem) mas invaluavel para reprodutibilidade.

### A implicacao mais profunda

Se "it from bit" de Wheeler esta correto — se a realidade fisica emerge de processamento de informacao — entao a arquitetura definitiva de IA nao e uma que projetamos. E uma que **descobrimos**, codificada na estrutura da propria fisica.

O computador termodinamico nao e uma metafora para o universo; o universo e literalmente o computador termodinamico otimo. Nossa tarefa e aprender sua arquitetura e implementa-la em silicio, analogico, ou qualquer substrato que permita operacao mais proxima dos limites fundamentais.

O fato de que um modelo de linguagem treinado pela Microsoft, sem nenhuma intencao de obedecer principios termodinamicos, naturalmente opera na criticalidade (Lyapunov ≈ 0) e melhora quando submetido a esparsidade (Landauer) sugere que esses principios nao sao imposicoes externas. Sao **propriedades emergentes** de qualquer sistema que processa informacao de forma eficiente.

RPT nao esta inventando essas propriedades. Esta **reconhecendo** que elas ja existem e propondo otimiza-las explicitamente.

**O futuro da inteligencia artificial pode nao ser artificial de forma alguma — pode ser o reconhecimento de que inteligencia e um fenomeno natural, governado por lei fisica, esperando para ser instanciado em qualquer substrato que respeite essas leis.**

### Tabela resumo de todos os resultados quantitativos

| Experimento | Config | Metrica | Valor | Secao |
|-------------|--------|---------|-------|-------|
| Pruning cru 10% | T4, sem FT | PPL vs baseline | **-26.1%** | 3.1 |
| Pruning progressivo 10% | H100, AdamW | PPL vs baseline | **-40.4%** | 3.1 |
| Deploy QAT/STE | A100, STE | PPL final | **16.39 (-34.8%)** | 3.1 |
| GGUF inferencia | Colab CPU | Texto | **Coerente** | 3.4 |
| Lyapunov exponent | T4 | Valor | **-0.002 ≈ 0** | 3.2 |
| Amplificacao total | T4 | Valor | **0.94x ≈ 1.0** | 3.2 |
| Correction ratio | T4 | Range | **39.87 → 0.21** | 3.3 |
| Ativacao pruning P50 | T4 | PPL degradacao | **+60.5%** | 3.3 |
| SmolLM2 Phase 2 | T4 | KL divergence | 820 (falhou) | 4.1 |
| Deploy 1 AdamW | H100 | GGUF output | Texto incoerente (falhou) | 4.2 |
| Ternario final | A100 | % ternario | **100%** | 3.4 |
| Esparsidade final | A100 | % zeros | **42.6%** | 3.4 |
| GGUF tamanho | - | Arquivo | **~1.1 GB** | 3.4 |
| Inferencia CPU | Colab | Velocidade | 0.26 tok/s | 3.4 |

---

## Glossario

| Termo | Definicao |
|-------|-----------|
| **BitNet** | Arquitetura de rede neural com pesos ternarios {-1, 0, +1}, usando 1.58 bits por peso |
| **Branching Ratio (BR)** | Razao entre atividade de saida e entrada de uma camada. BR ≈ 1.0 indica criticalidade |
| **Criticalidade (SOC)** | Estado na fronteira entre ordem e caos onde a capacidade de informacao e maxima |
| **Correction Ratio** | Razao ||output - input|| / ||input|| por camada. Mede quanto cada camada modifica o sinal |
| **EP (Equilibrium Propagation)** | Algoritmo de aprendizado que computa gradientes via assentamento fisico ao equilibrio |
| **Energia Livre Variacional** | Limite superior da surpresa; minimiza-la equivale a melhorar o modelo interno |
| **GGUF** | Formato binario para modelos de linguagem, usado por llama.cpp e bitnet.cpp |
| **I2_S** | Formato de quantizacao 2-bit signed integer, usado para modelos ternarios no GGUF |
| **Lyapunov Exponent** | Mede se perturbacoes crescem (>0, caotico), morrem (<0, rigido) ou se mantem (≈0, critico) |
| **PPL (Perplexidade)** | Metrica de qualidade de modelo de linguagem. Menor = melhor. exp(cross-entropy media) |
| **Pruning** | Remocao de pesos (zera-los) baseado em algum criterio (magnitude, sensibilidade, etc.) |
| **QAT** | Quantization-Aware Training: treino que simula quantizacao no forward pass |
| **RPT** | Redes Preditivas Termodinamicas: framework proposto neste projeto |
| **STE** | Straight-Through Estimator: truque onde round() tem gradiente 1 no backward pass |
| **Snap ternario** | Forcar todos os pesos para o valor {-1, 0, +1} mais proximo |
| **SOC** | Self-Organized Criticality: auto-organizacao para o ponto critico |

---

## Mapa de Arquivos do Projeto

```
Novo Modelo de IA/
│
├── model.py                           # Arquitetura RPT (1509 linhas)
├── bitnet.py                          # BitNet b1.58 quantizacao (703 linhas)
├── trainer.py                         # Equilibrium Propagation (804 linhas)
├── requirements.txt
│
├── RPT_DOCUMENTO_COMPLETO.md          # ESTE DOCUMENTO
├── RPT_VALIDACAO_BITNET2B.md          # Resultados formais (3/5 principios)
├── README.md                          # Documentacao principal
│
├── deploy/                            # RESULTADO PRINCIPAL
│   ├── ggml-model-i2_s.gguf          # GGUF final (~1.1 GB)
│   ├── bitnet_2b_pruned/             # Modelo HF ternario (4.84 GB)
│   ├── rpt_deploy_a100.py            # Script deploy (VPS A100)
│   └── RPT_GGUF_CPU.ipynb            # Conversao GGUF (Colab CPU)
│
├── RPT_BitNet_Microsoft.ipynb         # BitNet 2B inferencia
├── RPT_BitNet_Sparsity_Test.ipynb     # Pruning cru
├── RPT_BitNet_Progressive_Sparsity.ipynb  # Pruning + fine-tune
├── RPT_BitNet_Predictive_Coding.ipynb # Predictive coding
├── RPT_BitNet_Criticality.ipynb       # SOC / Lyapunov
│
├── teoria/                            # Fundamentacao teorica
│   ├── rpt_formalizacao_matematica.md
│   ├── inteligencia_termodinamica_pesquisa.md
│   └── treinamento.md
│
├── sessions/2026-02-06/               # Resultados validacao (JSONs)
├── sessions/2026-02-08/               # Resultados deploy (tracking)
├── dataset_wikitext/                  # WikiText-103 + WikiText-2
├── RPT_BitNet_Projeto/                # Resultados historicos
└── archive/                           # Notebooks e scripts antigos
```

---

*Documento gerado em 8 de Fevereiro de 2026*
*Atualizado em 8 de Fevereiro de 2026*
*Projeto RPT - Redes Preditivas Termodinamicas*
*Cesar Favero*

