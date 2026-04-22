"""Standalone loader for RPT packed recovered L78.

Usage:
    model, tokenizer, report = load_packed_recovered_l78('/path/to/gemma4_theory')
"""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

EXPECTED_PACKED_SHA256 = '0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116'

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def unpack_codes_from_blob(blob, n):
    raw = np.frombuffer(blob, dtype=np.uint8)
    codes = np.empty(raw.size * 4, dtype=np.uint8)
    codes[0::4] = raw & 0b11
    codes[1::4] = (raw >> 2) & 0b11
    codes[2::4] = (raw >> 4) & 0b11
    codes[3::4] = (raw >> 6) & 0b11
    return codes[:n]

def unpack_tensor_from_blob(blob, n, shape, scale, dtype=torch.float32):
    codes = unpack_codes_from_blob(blob, n)
    vals = np.zeros(n, dtype=np.float32)
    vals[codes == 1] = scale
    vals[codes == 2] = -scale
    return torch.from_numpy(vals).view(tuple(shape)).to(dtype=dtype)

def materialize_packed_mlp_into_model(model, manifest, bin_path):
    named_params = dict(model.named_parameters())
    applied = []
    with open(bin_path, 'rb') as f:
        for item in manifest['tensors']:
            name = item['name']
            param = named_params.get(name)
            if param is None:
                raise RuntimeError(f'Missing parameter: {name}')
            if getattr(param, 'is_meta', False):
                raise RuntimeError(f'Meta parameter cannot be materialized directly: {name}')
            f.seek(item['offset'])
            blob = f.read(item['packed_bytes'])
            recon = unpack_tensor_from_blob(blob, item['numel'], item['shape'], item['scale'], dtype=torch.float32)
            if tuple(param.shape) != tuple(recon.shape):
                raise RuntimeError(f'Shape mismatch for {name}: param={tuple(param.shape)} recon={tuple(recon.shape)}')
            with torch.no_grad():
                param.copy_(recon.to(device=param.device, dtype=param.dtype))
            applied.append(name)
    if len(applied) != 78:
        raise RuntimeError(f'Expected 78 tensors, got {len(applied)}')
    return applied

def resolve_package_dir(project_root):
    candidates = [project_root / 'exports' / 'ternary_mlp_l78', project_root / 'deploy']
    for path in candidates:
        if (path / 'manifest.json').exists() and (path / 'mlp_ternary_2bit.bin').exists():
            return path
    raise FileNotFoundError('Packed package not found under exports/ternary_mlp_l78 or deploy')

def load_packed_recovered_l78(project_root, token=None, device_map='auto'):
    project_root = Path(project_root)
    checkpoints = project_root / 'checkpoints'
    pruned_checkpoint = checkpoints / 'student_pruned'
    adapter = checkpoints / 'student_packed_ternary_recovered_l78_t4_resume'
    package_dir = resolve_package_dir(project_root)
    manifest_path = package_dir / 'manifest.json'
    bin_path = package_dir / 'mlp_ternary_2bit.bin'
    packed_sha = sha256_file(bin_path)
    if packed_sha != EXPECTED_PACKED_SHA256:
        raise RuntimeError(f'Packed SHA mismatch: {packed_sha}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(pruned_checkpoint), token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(str(pruned_checkpoint), dtype='auto', device_map=device_map, token=token)
    applied = materialize_packed_mlp_into_model(base_model, manifest, bin_path)
    model = PeftModel.from_pretrained(base_model, str(adapter), token=token, is_trainable=False)
    model.eval()
    report = {'packed_sha256': packed_sha, 'applied_packed_mlp_tensors': len(applied), 'load_order': 'student_pruned -> packed MLP -> recovered adapter'}
    return model, tokenizer, report
