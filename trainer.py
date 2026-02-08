"""
Trainer RPT com Equilibrium Propagation e QAT
==============================================
Implementa treinamento usando propagação de equilíbrio
(regras de aprendizado locais Hebbianas) com suporte
para Quantization-Aware Training (QAT) BitNet.

Autor: Cesar Favero
Data: Janeiro 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Callable, List, Union
import time
import json
import os
from dataclasses import dataclass, field
from tqdm import tqdm
import math

from model import RPTModel, RPTConfig

# Importa versão BitNet se disponível
try:
    from model import RPTModelBitNet, convert_rpt_to_bitnet, HAS_BITNET
    from bitnet import BitLinear, count_ternary_params, update_quant_ratio, get_quant_ratio
except ImportError:
    HAS_BITNET = False
    RPTModelBitNet = None
    update_quant_ratio = None
    get_quant_ratio = None


@dataclass
class TrainingConfig:
    """Configuração de treinamento."""

    # Dados
    train_data_path: str = ""
    eval_data_path: str = ""
    max_seq_len: int = 512

    # Batch
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Otimização
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 100
    num_epochs: int = 3
    max_steps: int = -1  # -1 = sem limite

    # Equilibrium Propagation
    use_equilibrium_propagation: bool = True
    ep_learning_rate: float = 0.01  # Para atualizações contrastivas

    # Logging
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000

    # Checkpointing
    output_dir: str = "./rpt_output"
    save_total_limit: int = 3

    # Hardware
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False

    # =====================
    # QAT (BitNet) Settings
    # =====================
    use_bitnet: bool = False          # Usar modelo BitNet quantizado
    qat_learning_rate: float = 3e-4   # LR maior para QAT (tipicamente 2-3x)
    qat_warmup_steps: int = 1000      # Mais warmup para estabilidade QAT
    qat_quant_warmup_ratio: float = 0.2  # Fração do treino para warmup de quantização
    qat_quant_schedule: str = 'cosine'   # Schedule: 'linear', 'cosine', 'exponential'


@dataclass
class QATConfig:
    """Configuração específica para Quantization-Aware Training."""

    # Quantização
    weight_bits: float = 1.58         # Ternário {-1, 0, +1}
    activation_bits: int = 8          # INT8 para ativações

    # Treinamento QAT
    learning_rate: float = 3e-4       # LR maior para QAT
    weight_decay: float = 0.01
    warmup_steps: int = 1000          # Mais warmup para estabilidade
    max_grad_norm: float = 1.0

    # Warmup progressivo de quantização (Fase 2)
    quant_warmup_ratio: float = 0.2   # 20% do treino para warmup de quantização
    quant_schedule: str = 'cosine'    # Schedule: 'linear', 'cosine', 'exponential'

    # Batch (pode ser maior devido a menor memória)
    batch_size: int = 16
    gradient_accumulation_steps: int = 4

    # Epochs (QAT pode precisar de mais)
    num_epochs: int = 5

    # Logging
    log_quantization_stats: bool = True
    log_every: int = 10


class RPTTrainer:
    """
    Trainer para Redes Preditivas Termodinâmicas.

    Suporta:
    - Treinamento padrão com backprop
    - Treinamento com Equilibrium Propagation
    - Treinamento híbrido
    - QAT (Quantization-Aware Training) com BitNet
    """

    def __init__(
        self,
        model: Union[RPTModel, 'RPTModelBitNet'],
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer=None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer

        # Detecta se é modelo BitNet
        self.is_bitnet = HAS_BITNET and isinstance(model, RPTModelBitNet)

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Ajusta learning rate para QAT se usando BitNet
        if self.is_bitnet and config.use_bitnet:
            self.effective_lr = config.qat_learning_rate
            self.effective_warmup = config.qat_warmup_steps
            print(f"[QAT] Usando LR={self.effective_lr}, warmup={self.effective_warmup}")
        else:
            self.effective_lr = config.learning_rate
            self.effective_warmup = config.warmup_steps

        # Otimizador
        self.optimizer = self._create_optimizer()

        # Scheduler
        total_steps = self._compute_total_steps()
        self.scheduler = self._create_scheduler(total_steps)

        # Quantization warmup (Fase 2)
        self.quant_warmup_steps = 0
        self.quant_schedule = 'cosine'
        if self.is_bitnet and config.use_bitnet:
            self.quant_warmup_steps = int(total_steps * config.qat_quant_warmup_ratio)
            self.quant_schedule = config.qat_quant_schedule
            print(f"[QAT] Quant warmup: {self.quant_warmup_steps} steps ({config.qat_quant_warmup_ratio*100:.0f}%)")
            print(f"[QAT] Quant schedule: {self.quant_schedule}")

        # Estado de treinamento
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

        # Histórico
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'sparsity': [],
            'branching_ratio': [],
            'free_energy': []
        }

        # Histórico específico QAT
        if self.is_bitnet:
            self.history['weight_sparsity'] = []  # Fração de zeros nos pesos quantizados
            self.history['quantization_scale'] = []  # Escala média dos pesos
            self.history['quant_ratio'] = []  # Ratio de quantização (warmup progressivo)

        # Diretório de saída
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Cria otimizador com weight decay."""
        # Separa parâmetros com e sem weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'bias' in name or 'norm' in name or 'threshold' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        return optim.AdamW(
            optimizer_groups,
            lr=self.effective_lr,  # Usa LR ajustado para QAT
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _compute_total_steps(self) -> int:
        """Computa número total de steps de treinamento."""
        if self.config.max_steps > 0:
            return self.config.max_steps
        
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs
    
    def _create_scheduler(self, total_steps: int):
        """Cria learning rate scheduler com warmup."""
        warmup_steps = self.effective_warmup  # Usa warmup ajustado para QAT

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executa um step de treinamento.

        Pode usar:
        - Backpropagation padrão
        - Equilibrium Propagation
        - Híbrido
        """
        self.model.train()

        # Atualiza quant_ratio para warmup progressivo (Fase 2)
        current_quant_ratio = 1.0
        if self.is_bitnet and self.quant_warmup_steps > 0 and update_quant_ratio is not None:
            current_quant_ratio = update_quant_ratio(
                self.model,
                self.global_step,
                self.quant_warmup_steps,
                self.quant_schedule
            )

        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids.clone()).to(self.device)

        # Forward com labels (ativa fase nudged)
        outputs = self.model(input_ids, labels=labels)

        loss = outputs['loss']
        
        # Normaliza pelo gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Aplica atualizações Equilibrium Propagation se habilitado
        if self.config.use_equilibrium_propagation and 'contrastive_updates' in outputs:
            self._apply_ep_updates(outputs['contrastive_updates'])
        
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'ce_loss': outputs['ce_loss'].item(),
            'free_energy': outputs.get('free_energy_loss', 0),
            'sparsity': outputs['free_metrics']['sparsity_history'][-1] if outputs['free_metrics']['sparsity_history'] else 0,
            'branching_ratio': self.model.branching_ratio.item()
        }

        # Adiciona métricas QAT
        if self.is_bitnet:
            metrics['quant_ratio'] = current_quant_ratio

        return metrics
    
    def _apply_ep_updates(self, contrastive_updates: Dict[str, torch.Tensor]):
        """
        Aplica atualizações de Equilibrium Propagation.
        
        Estas são atualizações locais Hebbianas que não precisam de backprop.
        """
        ep_lr = self.config.ep_learning_rate
        
        with torch.no_grad():
            for name, delta in contrastive_updates.items():
                # Extrai índice da camada
                layer_idx = int(name.split('_')[1])
                
                if layer_idx >= len(self.model.layers):
                    continue
                
                layer = self.model.layers[layer_idx]
                
                # Aplica delta ao modelo generativo
                # Delta é correlação (nudged - free), então aplicamos diretamente
                for param_name, param in layer.generative_model.named_parameters():
                    if 'weight' in param_name and param.shape == delta.shape:
                        param.add_(ep_lr * delta)
    
    def optimizer_step(self):
        """Executa step do otimizador."""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Avalia modelo no conjunto de validação."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_samples = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids.clone()).to(self.device)
            
            outputs = self.model(input_ids, labels=labels)
            
            batch_size = input_ids.shape[0]
            total_loss += outputs['loss'].item() * batch_size
            total_ce_loss += outputs['ce_loss'].item() * batch_size
            total_samples += batch_size
        
        metrics = {
            'eval_loss': total_loss / total_samples,
            'eval_ce_loss': total_ce_loss / total_samples,
            'eval_perplexity': math.exp(total_ce_loss / total_samples)
        }
        
        return metrics
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Salva checkpoint do modelo."""
        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f'checkpoint-{self.global_step}'
            )
        
        os.makedirs(path, exist_ok=True)
        
        # Modelo
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Otimizador
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss
        }, os.path.join(path, 'trainer_state.pt'))
        
        # Histórico
        with open(os.path.join(path, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        
        print(f"Checkpoint salvo em: {path}")
        
        # Limpa checkpoints antigos
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove checkpoints antigos para economizar espaço."""
        checkpoints = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith('checkpoint-'):
                step = int(name.split('-')[1])
                checkpoints.append((step, name))
        
        checkpoints.sort(reverse=True)
        
        for _, name in checkpoints[self.config.save_total_limit:]:
            path = os.path.join(self.config.output_dir, name)
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
    
    def load_checkpoint(self, path: str):
        """Carrega checkpoint."""
        self.model.load_state_dict(
            torch.load(os.path.join(path, 'model.pt'))
        )
        
        state = torch.load(os.path.join(path, 'trainer_state.pt'))
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.best_eval_loss = state['best_eval_loss']
        
        with open(os.path.join(path, 'history.json'), 'r') as f:
            self.history = json.load(f)
        
        print(f"Checkpoint carregado de: {path}")
    
    def train(self):
        """Loop principal de treinamento."""
        print("=" * 60)
        print("Iniciando treinamento RPT")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Equilibrium Propagation: {self.config.use_equilibrium_propagation}")
        print("=" * 60)
        
        total_steps = self._compute_total_steps()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            accumulation_steps = 0
            accumulated_metrics = {}
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                
                # Acumula métricas
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v
                
                accumulation_steps += 1
                
                # Optimizer step após acumulação
                if accumulation_steps >= self.config.gradient_accumulation_steps:
                    self.optimizer_step()
                    
                    # Média das métricas
                    for k in accumulated_metrics:
                        accumulated_metrics[k] /= accumulation_steps
                    
                    # Log
                    if self.global_step % self.config.log_every == 0:
                        self._log_metrics(accumulated_metrics)
                    
                    # Atualiza histórico
                    self.history['train_loss'].append(accumulated_metrics['loss'])
                    self.history['sparsity'].append(accumulated_metrics['sparsity'])
                    self.history['branching_ratio'].append(accumulated_metrics['branching_ratio'])
                    self.history['free_energy'].append(accumulated_metrics['free_energy'])
                    self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])

                    # Histórico QAT
                    if self.is_bitnet and 'quant_ratio' in accumulated_metrics:
                        self.history['quant_ratio'].append(accumulated_metrics['quant_ratio'])
                    
                    epoch_loss += accumulated_metrics['loss']
                    epoch_steps += 1
                    
                    # Update progress bar
                    postfix = {
                        'loss': f"{accumulated_metrics['loss']:.4f}",
                        'sparsity': f"{accumulated_metrics['sparsity']:.3f}",
                        'br': f"{accumulated_metrics['branching_ratio']:.3f}"
                    }
                    if self.is_bitnet and 'quant_ratio' in accumulated_metrics:
                        postfix['qr'] = f"{accumulated_metrics['quant_ratio']:.2f}"
                    progress_bar.set_postfix(postfix)
                    
                    # Reset accumulation
                    accumulation_steps = 0
                    accumulated_metrics = {}
                    
                    # Evaluation
                    if self.global_step % self.config.eval_every == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics:
                            self._log_metrics(eval_metrics, prefix="eval")
                            self.history['eval_loss'].append(eval_metrics['eval_loss'])
                            
                            # Save best
                            if eval_metrics['eval_loss'] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics['eval_loss']
                                self.save_checkpoint(
                                    os.path.join(self.config.output_dir, 'best')
                                )
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint()
                    
                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break
            
            # Fim da epoch
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            print(f"\nEpoch {epoch + 1} concluída. Loss média: {avg_epoch_loss:.4f}")
            
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break
        
        # Salva modelo final
        self.save_checkpoint(os.path.join(self.config.output_dir, 'final'))
        
        print("\n" + "=" * 60)
        print("Treinamento concluído!")
        print("=" * 60)
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log de métricas."""
        log_str = f"Step {self.global_step}"
        for k, v in metrics.items():
            log_str += f" | {prefix}_{k}: {v:.4f}"
        print(log_str)


class SimpleDataset(Dataset):
    """Dataset simples para testes."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokeniza todos os textos
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]
        }


# =============================================================================
# FUNÇÕES AUXILIARES PARA QAT
# =============================================================================

def get_quantization_stats(model: nn.Module) -> Dict[str, float]:
    """
    Coleta estatísticas de quantização do modelo BitNet.

    Retorna:
        Dict com esparsidade média, escala média, etc.
    """
    if not HAS_BITNET:
        return {}

    sparsities = []
    scales = []

    for module in model.modules():
        if isinstance(module, BitLinear):
            W_q, scale = module.quantize_weights()
            sparsity = (W_q == 0).float().mean().item()
            sparsities.append(sparsity)
            scales.append(scale.item())

    if not sparsities:
        return {}

    return {
        'weight_sparsity_mean': sum(sparsities) / len(sparsities),
        'weight_sparsity_min': min(sparsities),
        'weight_sparsity_max': max(sparsities),
        'scale_mean': sum(scales) / len(scales),
        'scale_min': min(scales),
        'scale_max': max(scales),
        'num_bitlinear_layers': len(sparsities),
    }


def prepare_model_for_inference(model: nn.Module) -> nn.Module:
    """
    Prepara modelo BitNet para inferência eficiente.

    - Pré-quantiza todos os pesos
    - Coloca em modo eval
    - Opcionalmente converte para BitLinearCPU

    Args:
        model: Modelo RPTModelBitNet treinado

    Returns:
        Modelo pronto para inferência
    """
    model.eval()

    if not HAS_BITNET:
        return model

    # Pré-quantiza todos os BitLinear
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.prepare_for_inference()

    print("[Inferência] Modelo preparado:")
    stats = get_quantization_stats(model)
    if stats:
        print(f"  - Camadas BitLinear: {stats['num_bitlinear_layers']}")
        print(f"  - Esparsidade média: {stats['weight_sparsity_mean']*100:.1f}%")
        print(f"  - Escala média: {stats['scale_mean']:.4f}")

    return model


def create_bitnet_trainer(
    config: RPTConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    tokenizer=None,
    qat_config: Optional[QATConfig] = None
) -> RPTTrainer:
    """
    Cria trainer configurado para QAT com modelo BitNet.

    Args:
        config: Configuração do modelo RPT
        train_dataloader: DataLoader de treinamento
        eval_dataloader: DataLoader de avaliação (opcional)
        tokenizer: Tokenizer
        qat_config: Configuração QAT (usa defaults se None)

    Returns:
        RPTTrainer configurado para QAT
    """
    if not HAS_BITNET:
        raise ImportError("Módulo bitnet necessário para criar trainer QAT")

    # Cria modelo BitNet
    model = RPTModelBitNet(config)

    # Configura trainer
    if qat_config is None:
        qat_config = QATConfig()

    train_config = TrainingConfig(
        use_bitnet=True,
        learning_rate=qat_config.learning_rate,
        qat_learning_rate=qat_config.learning_rate,
        qat_warmup_steps=qat_config.warmup_steps,
        qat_quant_warmup_ratio=qat_config.quant_warmup_ratio,
        qat_quant_schedule=qat_config.quant_schedule,
        weight_decay=qat_config.weight_decay,
        max_grad_norm=qat_config.max_grad_norm,
        batch_size=qat_config.batch_size,
        gradient_accumulation_steps=qat_config.gradient_accumulation_steps,
        num_epochs=qat_config.num_epochs,
        log_every=qat_config.log_every,
    )

    trainer = RPTTrainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
    )

    return trainer


def create_dummy_dataloader(
    tokenizer,
    num_samples: int = 100,
    max_length: int = 128,
    batch_size: int = 8
) -> DataLoader:
    """Cria dataloader com dados dummy para testes."""

    # Textos dummy
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Machine learning is a subset of artificial intelligence. " * 5,
        "Python is a popular programming language. " * 5,
        "Natural language processing enables computers to understand text. " * 5,
    ] * (num_samples // 4)

    dataset = SimpleDataset(texts[:num_samples], tokenizer, max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


if __name__ == "__main__":
    # Teste do trainer
    from transformers import AutoTokenizer
    from model import RPTModel, RPTConfig, create_rpt_from_smollm2_config

    print("=" * 60)
    print("Teste do Trainer RPT")
    print("=" * 60)

    # Carrega tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cria modelo pequeno para teste
    config = RPTConfig(
        vocab_size=49152,
        hidden_dim=128,  # Menor para teste
        num_layers=4,    # Menor para teste
        num_heads=4,
        intermediate_dim=512,
        max_seq_len=256,
        boundary_dim=16,
        num_regions=8
    )
    model = RPTModel(config)
    
    # Cria dataloaders
    train_dataloader = create_dummy_dataloader(tokenizer, num_samples=32, batch_size=4)
    eval_dataloader = create_dummy_dataloader(tokenizer, num_samples=8, batch_size=4)
    
    # Configuração de treinamento
    train_config = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=5,
        num_epochs=1,
        log_every=2,
        eval_every=10,
        save_every=100,
        use_equilibrium_propagation=True,
        output_dir="./rpt_test_output"
    )
    
    # Cria trainer
    trainer = RPTTrainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer
    )
    
    # Treina
    trainer.train()
    
    print("\nTeste concluído com sucesso!")
