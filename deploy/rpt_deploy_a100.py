"""
RPT Deploy - Script completo para A100 (40GB)

Faz TUDO em uma execucao:
  1. Carrega BitNet 2B
  2. Pruning progressivo 5% -> 10%
  3. Fine-tune QAT/STE (pesos ternarios no forward)
  4. Snap ternario + save
  5. Compila bitnet.cpp
  6. Converte GGUF (com todos os 10 fixes)
  7. Testa inferencia CPU

Uso (VPS com A100):
  pip install torch transformers accelerate datasets
  python rpt_deploy_a100.py

Tempo estimado: ~30 min total
"""
import os
import sys
import json
import time
import random
import shutil
import struct
import subprocess
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = 'microsoft/bitnet-b1.58-2B-4T-bf16'
TARGET_SPARSITY = 10
PROGRESSIVE = True           # 5% -> 10%
FT_STEPS = 300               # steps por nivel
FT_LR = 5e-4
SEQ_LEN = 128
SAVE_DIR = 'bitnet_2b_pruned'
GGUF_MODEL_DIR = 'BitNet-b1.58-2B-4T'
BITNET_DIR = 'BitNet'
PROTECT_LAYERS = ['embed', 'lm_head']

# ============================================================
# PART 1: GPU - TREINO
# ============================================================
print('=' * 60)
print('PART 1: TREINO (GPU)')
print('=' * 60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', 'ERRO: GPU nao disponivel! Verifique drivers CUDA.'

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu_name} ({vram_gb:.0f} GB)')

# Batch size baseado na VRAM
if vram_gb >= 45:
    BATCH_SIZE = 16
elif vram_gb >= 35:
    BATCH_SIZE = 8
else:
    BATCH_SIZE = 4
    print(f'AVISO: {vram_gb:.0f}GB VRAM pode ser pouco. Se der OOM, use A100.')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# --- Carregar modelo ---
print(f'\n[1/8] Carregando {MODEL_ID}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto'
)
n_params = sum(p.numel() for p in model.parameters())
print(f'  {n_params/1e9:.1f}B params, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# --- Dataset ---
print('[2/8] Carregando dataset...')
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
all_ids = []
for ex in dataset:
    text = ex['text'].strip()
    if len(text) >= 20:
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
chunks = [torch.tensor(all_ids[i:i+SEQ_LEN], dtype=torch.long)
          for i in range(0, len(all_ids) - SEQ_LEN, SEQ_LEN)]

val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
val_ids = []
for ex in val_dataset:
    text = ex['text'].strip()
    if len(text) >= 20:
        val_ids.extend(tokenizer.encode(text, add_special_tokens=False))
val_chunks = [torch.tensor(val_ids[i:i+SEQ_LEN], dtype=torch.long)
              for i in range(0, len(val_ids) - SEQ_LEN, SEQ_LEN)]
print(f'  Train: {len(chunks)} chunks, Val: {len(val_chunks)} chunks')


# --- Funcoes ---
def compute_ppl(mdl, val_data, max_batches=50):
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    for chunk in val_data[:max_batches]:
        input_ids = chunk.unsqueeze(0).cuda()
        with torch.no_grad():
            out = mdl(input_ids=input_ids, labels=input_ids)
        total_loss += out.loss.item() * (chunk.shape[0] - 1)
        total_tokens += chunk.shape[0] - 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def test_gen(mdl, tok, prompts):
    mdl.eval()
    results = []
    for p in prompts:
        inp = tok(p, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=30, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        results.append(tok.decode(out[0], skip_special_tokens=True))
    return results


def prune_magnitude(mdl, sparsity_pct):
    samples = []
    for name, param in mdl.named_parameters():
        if param.dim() < 2 or any(p in name for p in PROTECT_LAYERS):
            continue
        w = param.data.cpu().float()
        nonzero = w[w != 0].abs()
        if nonzero.numel() == 0:
            continue
        if nonzero.numel() > 50000:
            idx = torch.randperm(nonzero.numel())[:50000]
            samples.append(nonzero[idx])
        else:
            samples.append(nonzero)

    sample = torch.cat(samples)
    idx = min(int(len(sample) * sparsity_pct / 100.0), len(sample) - 1)
    threshold = sample.sort().values[idx].item()
    del sample, samples

    masks = {}
    total_pruned, total_weights = 0, 0
    for name, param in mdl.named_parameters():
        if param.dim() < 2 or any(p in name for p in PROTECT_LAYERS):
            continue
        w = param.data.cpu().float()
        mask = w.abs() > threshold
        masks[name] = mask
        param.data.copy_((w * mask.float()).to(param.dtype).to(param.device))
        total_pruned += (~mask).sum().item()
        total_weights += w.numel()

    actual = 100.0 * total_pruned / total_weights
    print(f'  Pruned: {actual:.1f}% ({total_pruned:,}/{total_weights:,})')
    return actual, masks


def finetune_ste(mdl, train_chunks, masks, original_scales, n_steps, lr, batch_size):
    torch.cuda.empty_cache()
    mdl.train()
    mdl.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    ternary_params = [(n, p) for n, p in mdl.named_parameters()
                      if p.dim() >= 2 and not any(x in n for x in PROTECT_LAYERS)]
    print(f'  STE ativo em {len(ternary_params)} camadas')

    gpu_masks = {}
    for name, param in ternary_params:
        if name in masks:
            gpu_masks[name] = masks[name].to(device=param.device, dtype=param.dtype)

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        batch = random.choices(train_chunks, k=batch_size)
        input_ids = torch.stack(batch).cuda()

        # STE forward: quantizar para ternario
        saved_data = {}
        for name, param in ternary_params:
            saved_data[name] = param.data.clone()
            w = param.data.float()
            nonzero = w[w != 0]
            if nonzero.numel() > 0:
                scale = original_scales[name]
                w_q = scale * (w / scale).round().clamp(-1, 1)
                if name in gpu_masks:
                    w_q *= gpu_masks[name].float()
                param.data.copy_(w_q.to(param.dtype))

        out = mdl(input_ids=input_ids, labels=input_ids)
        out.loss.backward()

        # STE backward: restaurar pesos continuos
        for name, param in ternary_params:
            param.data.copy_(saved_data[name])
            if name in gpu_masks and param.grad is not None:
                param.grad.data *= gpu_masks[name]
        del saved_data

        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())

        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            eta = (n_steps - step - 1) / sps
            print(f'  Step {step+1}/{n_steps} | loss: {avg:.3f} | {sps:.1f} it/s | ETA: {eta:.0f}s')

    mdl.gradient_checkpointing_disable()
    mdl.eval()
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, gpu_masks
    torch.cuda.empty_cache()

    final_loss = sum(losses[-50:]) / min(50, len(losses))
    print(f'  Pronto em {time.time()-t0:.0f}s | Loss: {final_loss:.3f}')
    return final_loss


# --- Baseline ---
print('[3/8] Baseline...')
ppl_baseline = compute_ppl(model, val_chunks)
texts_base = test_gen(model, tokenizer, ['The capital of France is'])
print(f'  PPL: {ppl_baseline:.2f}')
print(f'  Texto: {texts_base[0]}')

# Capturar escalas originais (ANTES de qualquer pruning)
original_scales = {}
for name, param in model.named_parameters():
    if param.dim() < 2 or any(p in name for p in PROTECT_LAYERS):
        continue
    w = param.data.float()
    nonzero = w[w != 0]
    if nonzero.numel() > 0:
        original_scales[name] = nonzero.abs().mean().item()
print(f'  Escalas capturadas: {len(original_scales)} camadas')

# --- Pruning + Fine-tune ---
print(f'\n[4/8] Pruning progressivo + QAT/STE fine-tune...')
print(f'  Config: batch={BATCH_SIZE}, lr={FT_LR}, steps={FT_STEPS}/nivel')

if PROGRESSIVE:
    all_levels = [5, 10, 15, 20, 25, 30, 40, 50]
    sparsity_steps = [s for s in all_levels if s <= TARGET_SPARSITY]
    if TARGET_SPARSITY not in sparsity_steps:
        sparsity_steps.append(TARGET_SPARSITY)
else:
    sparsity_steps = [TARGET_SPARSITY]

final_masks = None
for target in sparsity_steps:
    print(f'\n--- Nivel {target}% ---')
    actual_sp, masks = prune_magnitude(model, target)
    final_masks = masks

    ppl_before = compute_ppl(model, val_chunks)
    print(f'  PPL antes FT: {ppl_before:.2f}')

    finetune_ste(model, chunks, masks, original_scales,
                 n_steps=FT_STEPS, lr=FT_LR, batch_size=BATCH_SIZE)

    ppl_after = compute_ppl(model, val_chunks)
    delta = 100 * (ppl_after / ppl_baseline - 1)
    print(f'  PPL apos FT: {ppl_after:.2f} ({delta:+.1f}% vs baseline)')

# --- Snap ternario ---
print('\n[5/8] Snap ternario + verificacao...')

# Re-aplicar mascara
n_rezeroed = 0
for name, param in model.named_parameters():
    if name in final_masks:
        mask = final_masks[name].to(param.device).to(param.dtype)
        before = (param.data == 0).sum().item()
        param.data *= mask
        n_rezeroed += (param.data == 0).sum().item() - before
print(f'  {n_rezeroed:,} pesos re-zerados (drift do FT)')

# Snap
n_snapped = 0
for name, param in model.named_parameters():
    if param.dim() < 2 or any(p in name for p in PROTECT_LAYERS):
        continue
    w = param.data.float()
    nonzero = w[w != 0]
    if nonzero.numel() > 0:
        scale = original_scales[name]
        w_q = scale * (w / scale).round().clamp(-1, 1)
        if name in final_masks:
            w_q *= final_masks[name].to(device=param.device, dtype=torch.float32)
        n_snapped += (w_q != w).sum().item()
        param.data.copy_(w_q.to(param.dtype))
print(f'  {n_snapped:,} pesos ajustados')

# Verificacao
total_zero = sum((p.data == 0).sum().item() for n, p in model.named_parameters()
                 if p.dim() >= 2 and not any(x in n for x in PROTECT_LAYERS))
total_params = sum(p.numel() for n, p in model.named_parameters()
                   if p.dim() >= 2 and not any(x in n for x in PROTECT_LAYERS))
final_sparsity = 100.0 * total_zero / total_params

# Contar pesos ternarios
n_ternary = 0
for name, param in model.named_parameters():
    if param.dim() < 2 or any(p in name for p in PROTECT_LAYERS):
        continue
    w = param.data.float()
    nonzero = w[w != 0]
    if nonzero.numel() > 0:
        scale = nonzero.abs().median()
        ternary_mask = (w == 0) | ((w.abs() - scale).abs() < 1e-4)
        n_ternary += ternary_mask.sum().item()
    else:
        n_ternary += w.numel()
pct_ternary = 100 * n_ternary / total_params

ppl_final = compute_ppl(model, val_chunks)
delta_final = 100 * (ppl_final / ppl_baseline - 1)
print(f'  Esparsidade: {final_sparsity:.1f}%')
print(f'  Ternario: {pct_ternary:.1f}%')
print(f'  PPL final: {ppl_final:.2f} ({delta_final:+.1f}% vs baseline {ppl_baseline:.2f})')

texts_final = test_gen(model, tokenizer, [
    'The capital of France is',
    'Water boils at',
    'The largest planet in the solar system is',
])
for t in texts_final:
    print(f'  {t}')

# --- Salvar ---
print('\n[6/8] Salvando modelo...')
save_path = Path(SAVE_DIR)
save_path.mkdir(exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

metadata = {
    'base_model': MODEL_ID,
    'method': 'qat_ste',
    'target_sparsity': TARGET_SPARSITY,
    'actual_sparsity': final_sparsity,
    'pct_ternary': pct_ternary,
    'progressive': PROGRESSIVE,
    'ft_steps_per_level': FT_STEPS,
    'ft_lr': FT_LR,
    'batch_size': BATCH_SIZE,
    'seq_len': SEQ_LEN,
    'baseline_ppl': ppl_baseline,
    'final_ppl': ppl_final,
    'improvement': f'{delta_final:+.1f}%',
    'gpu': gpu_name,
}
with open(save_path / 'rpt_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

total_size = sum(f.stat().st_size for f in save_path.iterdir())
print(f'  Salvo em {save_path}/ ({total_size/1e9:.2f} GB)')

# Liberar GPU
del model
torch.cuda.empty_cache()
print('  GPU liberada')

# ============================================================
# PART 2: CPU - BITNET.CPP + GGUF
# ============================================================
print('\n' + '=' * 60)
print('PART 2: CONVERSAO GGUF (CPU)')
print('=' * 60)

# --- Instalar clang ---
print('\n[7/8] Compilando bitnet.cpp...')
if not shutil.which('clang'):
    subprocess.run(['sudo', 'apt-get', 'update', '-qq'],
                   capture_output=True, text=True)
    r = subprocess.run(['sudo', 'apt-get', 'install', '-y', '-qq', 'clang', 'lld'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'AVISO apt-get: {r.stderr[-200:]}')
        print('Se clang ja esta instalado, OK. Senao, instale manualmente.')
assert shutil.which('clang'), 'ERRO: clang nao instalou!'

# --- Clonar BitNet ---
if not os.path.exists(BITNET_DIR):
    subprocess.run(['git', 'clone', '--recursive',
                    'https://github.com/microsoft/BitNet.git'],
                   check=True, capture_output=True)
else:
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'],
                   cwd=BITNET_DIR, capture_output=True)

# === FIX 10: Architecture bitnet -> bitnet-b1.58 (O MAIS CRITICO) ===
constants_file = os.path.join(BITNET_DIR, '3rdparty/llama.cpp/gguf-py/gguf/constants.py')
with open(constants_file) as f:
    code = f.read()
if 'MODEL_ARCH.BITNET:         "bitnet",' in code:
    code = code.replace(
        'MODEL_ARCH.BITNET:         "bitnet",',
        'MODEL_ARCH.BITNET:         "bitnet-b1.58",'
    )
    with open(constants_file, 'w') as f:
        f.write(code)
    print('  Fix 10: architecture bitnet -> bitnet-b1.58')

# Instalar deps (force-reinstall gguf-py APOS o patch!)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                os.path.join(BITNET_DIR, 'requirements.txt')],
               capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '--force-reinstall', '--no-deps',
                os.path.join(BITNET_DIR, '3rdparty/llama.cpp/gguf-py')],
               capture_output=True)

# --- Copiar modelo com nome correto ---
if not os.path.exists(GGUF_MODEL_DIR):
    shutil.copytree(SAVE_DIR, GGUF_MODEL_DIR)

# === FIX 3: config.json ===
cfg_path = os.path.join(GGUF_MODEL_DIR, 'config.json')
with open(cfg_path) as f:
    cfg = json.load(f)
if cfg.get('architectures', [''])[0] == 'BitNetForCausalLM':
    cfg['architectures'] = ['BitnetForCausalLM']
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('  Fix 3: BitNetForCausalLM -> BitnetForCausalLM')

# === FIX 4: tokenizer_config.json ===
tok_path = os.path.join(GGUF_MODEL_DIR, 'tokenizer_config.json')
with open(tok_path) as f:
    tok = json.load(f)
if tok.get('tokenizer_class') == 'TokenizersBackend':
    tok['tokenizer_class'] = 'PreTrainedTokenizerFast'
    with open(tok_path, 'w') as f:
        json.dump(tok, f, indent=2)
    print('  Fix 4: TokenizersBackend -> PreTrainedTokenizerFast')

# --- Gerar kernels + fix const ---
subprocess.run([sys.executable, 'utils/codegen_tl2.py',
                '--model', 'bitnet_b1_58-3B',
                '--BM', '160,320,320', '--BK', '96,96,96', '--bm', '32,32,32'],
               cwd=BITNET_DIR, capture_output=True)

mad_file = os.path.join(BITNET_DIR, 'src/ggml-bitnet-mad.cpp')
if os.path.exists(mad_file):
    with open(mad_file) as f:
        code = f.read()
    if 'int8_t * y_col = y + col * by;' in code:
        code = code.replace('int8_t * y_col = y + col * by;',
                            'const int8_t * y_col = y + col * by;')
        with open(mad_file, 'w') as f:
            f.write(code)
        print('  Fix 6: const correctness')

# --- Compilar ---
r = subprocess.run(['cmake', '-B', 'build', '-DBITNET_X86_TL2=OFF',
                    '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++'],
                   cwd=BITNET_DIR, capture_output=True, text=True)
if r.returncode != 0:
    print(f'ERRO cmake configure:\n{r.stderr[-500:]}')
    sys.exit(1)
r = subprocess.run(['cmake', '--build', 'build', '--config', 'Release', '-j4'],
                    cwd=BITNET_DIR, capture_output=True, text=True)
if r.returncode != 0:
    print(f'ERRO cmake build:\n{r.stderr[-500:]}')
    sys.exit(1)
quantize_bin = os.path.join(BITNET_DIR, 'build/bin/llama-quantize')
cli_bin = os.path.join(BITNET_DIR, 'build/bin/llama-cli')
assert os.path.exists(quantize_bin), 'ERRO: llama-quantize nao encontrado apos build!'
print('  Compilacao OK')

# === FIXES NO CONVERTER ===
conv_file = os.path.join(BITNET_DIR, 'utils/convert-hf-to-gguf-bitnet.py')
with open(conv_file) as f:
    code = f.read()

fixes = []

# Fix 5: vocab fallback
bitnet_pos = code.rfind('class BitnetModel')
if bitnet_pos > 0:
    old_pos = code.find('self._set_vocab_sentencepiece()', bitnet_pos)
    if old_pos > 0:
        method_start = code.rfind('def set_vocab(self):', 0, old_pos)
        method_end = old_pos + len('self._set_vocab_sentencepiece()')
        code = (code[:method_start] +
                "def set_vocab(self):\n"
                "        try:\n"
                "            self._set_vocab_sentencepiece()\n"
                "        except FileNotFoundError:\n"
                "            try:\n"
                "                self._set_vocab_llama_hf()\n"
                "            except (FileNotFoundError, TypeError):\n"
                "                self._set_vocab_gpt2()" +
                code[method_end:])
        fixes.append('vocab fallback')

# Fix 8: weight_quant
if 'data_torch = self.weight_quant(data_torch)' in code:
    code = code.replace(
        'data_torch = self.weight_quant(data_torch)',
        'pass  # RPT fix: pesos ja ternarios'
    )
    fixes.append('weight_quant')

# Fix 9: BOS prepending
if 'tokenizer.decode(tokenizer.encode(reverse_vocab[i]))' in code:
    code = code.replace(
        'tokenizer.decode(tokenizer.encode(reverse_vocab[i]))',
        'tokenizer.decode(tokenizer.encode(reverse_vocab[i], add_special_tokens=False))'
    )
    fixes.append('BOS prepending')

with open(conv_file, 'w') as f:
    f.write(code)
for fix in fixes:
    print(f'  Fix: {fix}')

# --- Converter ---
print('\n[8/8] Convertendo GGUF...')
f32_gguf = os.path.join(GGUF_MODEL_DIR, 'ggml-model-f32.gguf')
i2s_gguf = os.path.join(GGUF_MODEL_DIR, 'ggml-model-i2_s.gguf')

# Limpar antigos
for old in [f32_gguf, i2s_gguf]:
    if os.path.exists(old):
        os.remove(old)

# HF -> f32
print('  Convertendo HF -> f32 GGUF...')
r = subprocess.run(
    [sys.executable, 'utils/convert-hf-to-gguf-bitnet.py',
     os.path.abspath(GGUF_MODEL_DIR), '--outtype', 'f32'],
    cwd=BITNET_DIR, capture_output=True, text=True
)
if r.returncode != 0:
    print(f'ERRO conversao:\n{r.stderr[-1000:]}')
    sys.exit(1)
assert os.path.exists(f32_gguf), 'ERRO: f32 GGUF nao gerado!'
print(f'  f32: {os.path.getsize(f32_gguf)/1e9:.1f} GB')

# f32 -> I2_S
print('  Quantizando f32 -> I2_S...')
r = subprocess.run([quantize_bin, f32_gguf, i2s_gguf, 'I2_S', '1'],
                    capture_output=True, text=True)
if r.returncode != 0:
    print(f'ERRO quantizacao:\n{r.stderr[-500:]}')
    sys.exit(1)
assert os.path.exists(i2s_gguf), 'ERRO: I2_S nao gerado!'
print(f'  I2_S: {os.path.getsize(i2s_gguf)/1e6:.0f} MB')

# Limpar f32
os.remove(f32_gguf)

# --- Verificar architecture no GGUF ---
print('\n  Verificando GGUF...')
with open(i2s_gguf, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]
    _skip_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    arch_found = False
    for _ in range(min(n_meta, 50)):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8', errors='replace')
        vtype = struct.unpack('<I', f.read(4))[0]
        if key == 'general.architecture' and vtype == 8:
            val_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(val_len).decode('utf-8')
            print(f'  general.architecture = "{val}"')
            if val != 'bitnet-b1.58':
                print('  AVISO: architecture ERRADA! Deveria ser bitnet-b1.58')
            else:
                print('  Architecture OK!')
            arch_found = True
            break
        elif vtype in _skip_sizes:
            f.read(_skip_sizes[vtype])
        elif vtype == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            for _a in range(arr_len):
                if arr_type in _skip_sizes:
                    f.read(_skip_sizes[arr_type])
                elif arr_type == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
        else:
            break
    if not arch_found:
        print('  AVISO: general.architecture nao encontrado!')

# --- Teste inferencia ---
print('\n--- TESTE DE INFERENCIA ---')
prompts = [
    'The capital of France is',
    'Water boils at',
]
for prompt in prompts:
    r = subprocess.run(
        [cli_bin, '-m', i2s_gguf, '-p', prompt, '-n', '50', '--temp', '0'],
        capture_output=True, text=True
    )
    lines = [l for l in r.stdout.split('\n') if l.strip() and not l.startswith('llama_')
             and not l.startswith('llm_') and not l.startswith('common_')
             and not l.startswith('build:') and not l.startswith('system_info')
             and not l.startswith('sampling') and not l.startswith('generate')
             and not l.startswith('main:') and not l.startswith('ggml_')]
    output = '\n'.join(lines[-5:]) if lines else '(sem saida)'
    print(f'\nPrompt: {prompt}')
    print(f'{output}')

# ============================================================
# RESUMO
# ============================================================
print('\n' + '=' * 60)
print('PRONTO!')
print('=' * 60)
print(f'GPU: {gpu_name}')
print(f'Modelo base: {MODEL_ID}')
print(f'Esparsidade: {final_sparsity:.1f}%')
print(f'Ternario: {pct_ternary:.1f}%')
print(f'PPL: {ppl_baseline:.2f} -> {ppl_final:.2f} ({delta_final:+.1f}%)')
print(f'GGUF: {os.path.abspath(i2s_gguf)} ({os.path.getsize(i2s_gguf)/1e6:.0f} MB)')
print(f'\nIniciando servidor HTTP na porta 8080...')
print(f'Baixe em: http://<IP_DA_VPS>:8080/ggml-model-i2_s.gguf')

os.chdir(os.path.abspath(GGUF_MODEL_DIR))
import http.server
import socketserver
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(('0.0.0.0', 8080), handler) as httpd:
    print('Servidor rodando em 0.0.0.0:8080 - Ctrl+C para parar')
    httpd.serve_forever()
