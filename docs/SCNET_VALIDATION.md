# Validar RPT na SCNet

## Objetivo e estado

Usar o recurso gratuito disponível na conta para testar integridade, avaliação
independente e compatibilidade real. O acesso a uma oferta no marketplace não
comprova provisionamento, saldo ou compatibilidade do acelerador.

A oferta BW-1 observada em 2 de novembro de 2026 usa o ecossistema DTK/DAS.
A documentação da SCNet orienta utilizar a imagem PyTorch adaptada ao hardware;
DTK não tem correspondência direta com versões CUDA, e dependências que chamam
CUDA diretamente podem precisar de adaptação. Fontes oficiais:

- [Ambiente PyTorch](https://www.scnet.cn/help/docs/mainsite/ai/appendix/environment-pytorch/index.html).
- [Perguntas frequentes sobre DTK e CUDA](https://www.scnet.cn/help/docs/mainsite/ai/faq/index.html).
- [Uso da plataforma, E-Shell e Slurm](https://www.scnet.cn/help/docs/mainsite/bw-core-node/platform-basic-usage/index.html).
- [Acesso SSH ao Notebook](https://www1.scnet.cn/help/docs/mainsite/ai/notebook/function-introduction/SSH/).

## 1. Instância

Entrar na conta, verificar qual recurso gratuito foi concedido e abrir uma
instância Notebook na imagem PyTorch correspondente à placa. Confirmar a cota
e o preço total antes de provisionar. Não selecionar automaticamente uma opção
paga nem trocar a imagem por CUDA genérico.

O protocolo precisa de uma instância para iniciar; oito placas não são
necessárias. Registrar modelo da placa, memória, imagem, versões do framework,
limite de tempo e capacidade de armazenamento disponível. Os comandos abaixo
não criam recursos e não alteram a configuração da conta.

## 2. Checkout e diagnóstico

```bash
git clone https://github.com/cesarfavero/RPT.git
cd RPT
git rev-parse HEAD
python3 -m validation.environment --output scnet_environment.json
bash scripts/scnet_validate.sh --phase cpu
```

Antes de uma campanha, fixar o checkout em um commit completo e guardar esse
SHA. Caso NumPy esteja ausente, usar o pacote compatível com a imagem do
provedor; os testes de referência locais usaram NumPy 2.3.5. Não reinstalar
PyTorch/Triton do índice CUDA sobre a distribuição DTK.

Em clusters Slurm, a SCNet informa que o nó de login tem acesso à internet e os
nós de computação podem não ter. Preparar dependências, datasets e modelos no
ambiente permitido antes de submeter os jobs. Não executar benchmarks de GPU no
nó de login. Usar a fila e a cota realmente concedidas pela plataforma.

## 3. Teste do kernel, antes de gastar com avaliação longa

```bash
bash scripts/scnet_validate.sh --phase kernel --full-shapes
```

Esse comando executa o operador Triton real contra uma referência densa com
dados sintéticos. Inclui shapes não múltiplos dos tiles, decode e prefill,
primeira chamada, chamadas aquecidas, erro numérico e razão de tempo contra
F.linear. O primeiro tempo pode usar cache de compilação já existente; o
relatório declara essa limitação.

Ausência de Torch/Triton, erro no driver, compilação incompatível e diferença
numérica ficam registrados como blocked ou failed. Nenhum fallback recebe o
rótulo de Triton aprovado. O resultado vale para o operador e para os shapes
medidos, não para qualidade ou velocidade do modelo completo.

Se DTK não executar o kernel atual, a máquina ainda pode ser útil para controles
densos, recuperação e testes de representação. Uma adaptação DTK será uma etapa
separada. Os números NVIDIA/T4 existentes não certificam essa adaptação.

## 4. Pesos e dados

O bundle precisa conter student_pruned, adapter_recovered_l78 e os dois arquivos
packed/manifest.json e packed/mlp_ternary_2bit.bin. O arquivo packed histórico
tem 414.056.448 bytes; o checkpoint denso tem cerca de 5,23 GB de pesos. Verificar
espaço livre antes da transferência e incluir os adaptadores, tokenizer e
temporários no orçamento. Esses valores descrevem o bundle antigo, que ainda
duplica as MLPs densas.

Transferir os artefatos pela forma de acesso autorizada para a conta. Não mudar
permissões de compartilhamento para facilitar o download. Verificar os hashes
na máquina de destino. A instrução de identificação do artefato está no guia
de reprodutibilidade.

Usar os mesmos datasets congelados nos controles densos e no caminho packed.
Na primeira campanha não alterar o recovery para melhorar o teste externo.
Revisar resultados somente depois de concluir as variantes declaradas.

## 5. Avaliação e resultados

Seguir [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). O runner externo tem modos
explicitamente separados packed-triton e hf-dense. O modo packed exige execução
Triton observada nas 78 camadas; o modo denso é um controle identificado.

Guardar JSONs, logs, hashes, configuração, versões e consumo da cota. Se não
houver acesso à telemetria de energia, registrar energia como não medida.
Tempo menor não é, por si só, uma medição de energia menor.

Ao terminar a campanha, liberar a instância pelo mecanismo normal da plataforma,
depois de verificar que os resultados foram copiados para armazenamento durável.
