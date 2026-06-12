# RPT

Versões deste README:

- English: [`README.md`](README.md)
- Português: `README.pt-BR.md`
- 中文: [`README.zh-CN.md`](README.zh-CN.md)
- Русский: [`README.ru.md`](README.ru.md)

Repositório de pesquisa e engenharia com duas trilhas relacionadas:

1. pesquisa em `BitNet / sparsity / QAT-STE`, que serviu como base inicial para o paper e para a metodologia;
2. `gemma4_theory/`, que é a trilha prática atual: pegar um modelo Gemma 2B denso moderno, podar, ternarizar o caminho MLP, recuperar função e gerar um bundle utilizável com runtime Triton e método próprio.

## Intenção

Este repositório é um projeto de pesquisa e engenharia.

Ele documenta o que foi testado, o que funcionou, o que falhou e o que hoje está utilizável. Ele não deve ser lido como material de venda; os claims aqui são claims técnicos, limitados pelos benchmarks e ambientes realmente validados.

## O Que Importa Agora

Se você quer ir direto para o resultado prático atual, comece em [`gemma4_theory/README.md`](gemma4_theory/README.md).

Essa trilha já validou o seguinte no benchmark local de 100 perguntas:

- referência HF original `google/gemma-2-2b-it`: `0.90`
- nosso checkpoint denso melhorado `student_pruned`: `0.92`
- nosso caminho ternário packed após adapter recovered: `1.00`
- nosso runtime Triton integrado no caminho packed: `1.00`

Esse é o claim mais importante hoje no repositório:

- melhoramos o resultado local em relação ao Gemma 2B original;
- convertemos todo o caminho MLP para uma representação packed ternária;
- recuperamos funcionalidade sobre esse caminho packed;
- produzimos um bundle de runtime utilizável para Linux/NVIDIA/Colab.

## De Microsoft Como Guia Para Stack Própria

O material de BitNet/Microsoft foi útil no início como guia conceitual para pensamento ternário, pruning e disciplina de deploy.

Mas a trilha Gemma atual não é um clone do runtime BitNet da Microsoft. A stack prática atual neste repositório é nossa:

- nosso próprio caminho de pruning/recovery sobre Gemma 2B
- nosso próprio export packed ternário de MLP
- nosso próprio processo de recovery do caminho packed
- nosso próprio caminho estrutural e Triton de runtime
- nosso próprio bundle de produção para Linux/NVIDIA/Colab

A forma honesta de dizer isso é:

- Microsoft/BitNet foi uma referência útil no começo
- o método funcional e o runtime atuais são nossos

## Estado Atual Por Trilha

### 1. Trilha BitNet / Paper

O repositório ainda contém o trabalho anterior focado em BitNet:

- fonte do paper: `paper/rpt_sparsity_paper.tex`
- scripts de deploy: `deploy/`
- notebooks e logs históricos

Isso continua importante para o histórico de pesquisa, mas não é mais a única parte ativa do repositório.

### 2. Trilha Gemma

O caminho prático atual está em `gemma4_theory/`.

Marcos já validados:

- referência HF original Gemma 2B na suíte local: `0.90`
- nosso checkpoint denso melhorado `student_pruned`: `0.92`
- expansão completa do MLP ternário recuperada na suíte local: `0.92`
- caminho packed/materialized recuperado: `1.00`
- runtime Triton integrado: `1.00`
- verificação pós-prewarm do bundle em T4:
  - `avg_tokens_per_sec ~= 5.66`
  - `min_latency ~= 0.35s`
  - `max_latency ~= 1.02s`

Isso significa que o projeto já passou da fase “só teoria”. Hoje ele já tem um bundle de runtime funcional para o ambiente suportado, sem depender da toolchain BitNet da Microsoft.

## Repositório Separado de Runtime

O runtime e a CLI agora também têm um repositório dedicado:

- [cesarfavero/rpt-runtime](https://github.com/cesarfavero/rpt-runtime)

O repositório principal RPT continua sendo a fonte de verdade de pesquisa e validação.

## Por Onde Começar

### Se você quer a narrativa de pesquisa

Abra:

- [`gemma4_theory/README.md`](gemma4_theory/README.md)

### Se você quer ver artefato packed e runtime

Abra:

- [`gemma4_theory/deploy/README.md`](gemma4_theory/deploy/README.md)
- [`gemma4_theory/runtime/packed_recovered_l78/README.md`](gemma4_theory/runtime/packed_recovered_l78/README.md)

### Se você já baixou o zip de produção

O fluxo esperado é:

```bash
unzip production_triton_packed_recovered_l78.zip
cd production_triton_packed_recovered_l78
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

Os detalhes estão documentados nos READMEs de runtime e deploy.

## Estrutura do Repositório

- `paper/`: fonte do paper e arquivos ligados à publicação.
- `deploy/`: trilha legada de deploy BitNet.
- `gemma4_theory/`: trilha atual denso -> pruned -> ternário -> runtime.
- `sessions/`: logs históricos de experimento.
- `results/`: saídas JSON/CSV versionadas dos experimentos.

## Reprodutibilidade

A regra deste repositório é simples:

- notebooks salvam artefatos timestampados;
- resultados não são editados manualmente;
- toda etapa útil deve terminar com commit + push.

## Ecossistema Público

O ecossistema público do RPT é separado por responsabilidade:

- [`RPT`](https://github.com/cesarfavero/RPT): pesquisa, notebooks, validação, papers, artefatos de resultado e notas de reprodutibilidade.
- [`rpt-runtime`](https://github.com/cesarfavero/rpt-runtime): camada de runtime e CLI para executar artefatos RPT empacotados.
- [`rpt-model-bundles`](https://github.com/cesarfavero/rpt-model-bundles): manifests, checksums, notas de release e metadados de distribuição dos bundles.

Use GitHub Issues no repositório relevante para bugs, relatos de reprodução,
lacunas de documentação e dúvidas de integração.

## Licença

Apache License 2.0. Veja [`LICENSE`](LICENSE) e [`NOTICE`](NOTICE).

Pesos de modelos, datasets e checkpoints de terceiros continuam sujeitos às
suas próprias licenças e termos upstream. Em especial, assets da família Gemma
continuam sujeitos aos Gemma Terms of Use aplicáveis.
