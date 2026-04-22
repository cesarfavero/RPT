# RPT Packed Recovered L78 Runtime

Runtime de inferência para o artefato `packed/materialized L78`.

## Estado técnico

Este runtime executa o modelo validado no marco packed recovered L78:

```text
student_pruned -> materialize packed MLP -> recovered adapter
```

Ele é executável e reproduzível, mas ainda não é um runtime ternário nativo. O `packed_bin` substitui os 78 MLPs no carregamento; a execução final ainda usa tensores materializados no modelo base.

## Layout esperado do modelo

Para uma pasta `MODEL_DIR`, coloque:

```text
MODEL_DIR/
  student_pruned/
    config.json
    tokenizer.json
    tokenizer_config.json
    model-*.safetensors ou pytorch_model*.bin
  adapter_recovered_l78/
    adapter_config.json
    adapter_model.safetensors
  packed/
    manifest.json
    mlp_ternary_2bit.bin
  runtime/
    loader.py
    run_inference.py
    requirements.txt
```

O notebook `21_build_distributable_packed_recovered_l78.ipynb` monta essa estrutura automaticamente a partir do Drive/local.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
```

## Inferência

```bash
python runtime/run_inference.py \
  --model-dir /path/to/MODEL_DIR \
  --prompt "Qual a capital da França?"
```

## Verificação rápida

```bash
python runtime/run_inference.py \
  --model-dir /path/to/MODEL_DIR \
  --prompt "Quanto é 12 x 8?" \
  --max-new-tokens 24
```

## Próximo marco de leveza real

Para reduzir memória de inferência de verdade, o próximo runtime precisa fazer uma destas duas coisas:

1. Remover os MLPs densos do checkpoint distribuído e materializar apenas a partir do packed no carregamento.
2. Executar os MLPs packed diretamente com módulo/kernel ternário, sem materializar para float.

Enquanto isso não existir, o pacote é executável e menor na parte empacotada, mas ainda depende da base densa.
