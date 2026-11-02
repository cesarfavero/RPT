# Estado das evidências e mudanças de validação

Esta revisão usa o RPT histórico em a1c904ba0ebee3c8aab236f62ef1685eef3d2f69.
Os resultados históricos não foram editados. Os novos testes exercitam contratos
de avaliação, integridade e um experimento sintético; não substituem a execução
do modelo final em hardware real.

## O que os resultados existentes sustentam

| Evidência | Interpretação permitida | O que ainda não demonstra |
| --- | --- | --- |
| Packed recuperado e Triton com 100/100 | Recuperação da suíte de desenvolvimento e execução funcional | Generalização, pois as mesmas respostas entraram no recovery |
| External500 | Avaliação da base densa/pruned com adaptador anterior | Qualidade do packed recuperado final |
| Arquivo MLP com 414.056.448 bytes | Packing de 78 tensores em 2 bits, 8x contra FP16 desses tensores | Modelo inteiro em 1,58 bit ou download 8x menor |
| Média histórica de 5,66 tok/s na T4 | Verificação de oito pedidos curtos após prewarm | Throughput agregado, cold start ou aceleração sobre baseline equivalente |
| Kernel Triton atual | Decodificação de tiles packed e dot em ponto flutuante | Eliminação de multiplicações ou execução esparsa dos zeros |

Os notebooks 15 e 17 treinam nas respostas esperadas da mesma suíte de 100
questões. O teste externo final foi antecipado no roadmap: deve ocorrer antes
de nova seleção de checkpoint, alteração de recovery ou promessa de ganho.

## Reavaliação dos textos salvos

O avaliador antigo aceitava substring em ambas as direções. Isso permitia
casos como vazio para Paris, 19 para 9 e uma resposta que negava Paris.
O avaliador v2 compara a resposta completa após normalização documentada.

| Arquivo histórico | Corretor original | rpt-exact-match-v2 |
| --- | ---: | ---: |
| Referência HF original, notebook16 | 90/100 | 82/100 |
| student_pruned, notebook16 | 92/100 | 81/100 |
| Packed recuperado, notebook18 | 100/100 | 100/100 |
| Packed Triton, notebook24 | 100/100 | 100/100 |

Essa tabela reavalia as mesmas respostas salvas; não é inferência nova. A regra
mais conservadora pode rejeitar respostas semanticamente corretas com texto
adicional não listado nos aliases. Não ajustar aliases depois de observar um
teste reservado. Os julgamentos que mudaram estão registrados individualmente
em [historical_regrade_20261102.json](../../validation/results/historical_regrade_20261102.json).

Os dois 100/100 permanecem. A principal limitação continua sendo a independência
dos dados e a ausência de um controle denso com o mesmo orçamento de recovery.

## Teoria e prioridade científica

Redução de perplexidade após pruning não é uma verificação experimental do
limite de Landauer. O proxy histórico de amplificação entre camadas também não
estabelece criticalidade auto-organizada. Essas conexões permanecem hipóteses
de pesquisa. As observações de PPL e zeros podem ser estudadas com controles de
treinamento, seeds, ambiente e telemetria energética.

A combinação genérica de quantização ternária, resíduos e execução adaptativa
já possui precedentes. Referências primárias prioritárias:

- [ParetoQ, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/83b17fb3369b1effa97ca5409526b02e-Abstract-Conference.html).
- [CAT-Q, ICML 2026](https://proceedings.mlr.press/v306/wang26dr.html).
- [QTEA, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1775/).
- [Sherry, ACL 2026](https://aclanthology.org/2026.acl-long.513/).
- [PTQTP](https://arxiv.org/abs/2509.16989) e [MatGPTQ](https://arxiv.org/abs/2602.03537).
- [DecDEC](https://arxiv.org/abs/2412.20185) e [T-MAC](https://arxiv.org/abs/2407.00088).
- [CALM](https://arxiv.org/abs/2207.07061) e [Conformal Risk Control](https://arxiv.org/abs/2208.02814).
- [Landauer, experimento de apagamento lógico](https://www.nature.com/articles/nature10872).

O novo protótipo estuda uma pergunta delimitada: limites baratos podem permitir
parar as correções da projeção final preservando o argmax da representação
completa? O [relatório sintético](../../research/adaptive_output/README.md)
registra onde isso funcionou e onde o custo adicional tornou a execução mais
lenta. Não há reivindicação de prioridade nem validação de LLM nesse protótipo.

## Trabalho ainda dependente de hardware e artefatos

1. Avaliar o bundle final em dados externos congelados, com controle denso.
2. Medir kernel e modelo completo na SCNet e em NVIDIA, sem misturar backends.
3. Implementar e comparar carregamento direto do formato packed, sem checkpoint
   denso duplicado, preservando a equivalência com o caminho atual.
4. Otimizar decode/prefill somente depois de localizar os custos dominantes.
5. Repetir recovery com treino, desenvolvimento e teste separados, seeds e
   controles de mesmo orçamento; comparar pontos 2/3/4 bits competitivos.
6. Aprender conteúdo e ordem das correções e medir todos os custos do controlador.
7. Repetir em uma segunda família de modelos antes de generalizar a conclusão.

O protocolo em [validation/protocol.json](../../validation/protocol.json)
registra essas decisões antes das campanhas. Nenhuma tarefa pendente aparece
como resultado concluído.
