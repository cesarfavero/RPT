"""
BitNet b1.58 Implementation for RPT
====================================
Implementa quantização ternária (pesos {-1, 0, +1}) e ativações INT8
para inferência eficiente em CPU.

Baseado em:
- Microsoft BitNet b1.58: https://arxiv.org/pdf/2504.12285
- The Era of 1-bit LLMs: https://arxiv.org/html/2402.17764v1

Autor: Cesar Favero
Data: Janeiro 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional
import math


# =============================================================================
# FUNÇÕES DE QUANTIZAÇÃO
# =============================================================================

def quantize_weights_ternary(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantiza pesos para valores ternários {-1, 0, +1}.

    Usa absmean scaling (método do BitNet b1.58):
    W_q = RoundClip(W / (gamma + eps), -1, 1)
    onde gamma = mean(|W|)

    Args:
        W: Tensor de pesos [out_features, in_features]

    Returns:
        W_q: Pesos quantizados {-1, 0, +1}
        scale: Fator de escala (gamma)
    """
    # Calcula fator de escala absmean
    scale = W.abs().mean() + 1e-8

    # Normaliza e arredonda para ternário
    W_normalized = W / scale
    W_q = torch.clamp(torch.round(W_normalized), -1, 1)

    return W_q, scale


def quantize_activations_int8(
    x: torch.Tensor,
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantiza ativações para INT8 usando absmax per-token.

    Formula: x_q = Clip(Round(x * Qb / max(|x|)), -Qb-1, Qb)
    onde Qb = 2^(bits-1) - 1 = 127 para 8 bits

    Args:
        x: Tensor de ativações [batch, seq, hidden]
        bits: Precisão de bits (default 8)

    Returns:
        x_q: Ativações quantizadas
        scale: Fator de escala por token [batch, seq, 1]
    """
    Qb = 2 ** (bits - 1) - 1  # 127 para 8 bits

    # Per-token scaling (absmax ao longo da última dimensão)
    scale = x.abs().amax(dim=-1, keepdim=True) / Qb + 1e-8

    # Quantiza
    x_q = torch.clamp(torch.round(x / scale), -Qb - 1, Qb)

    return x_q, scale


def dequantize(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantiza tensor multiplicando pela escala."""
    return x_q * scale


# =============================================================================
# STRAIGHT-THROUGH ESTIMATOR (STE)
# =============================================================================

class STETernaryFunction(Function):
    """
    Straight-Through Estimator para quantização ternária.

    Forward: aplica quantização não-diferenciável para {-1, 0, +1}
    Backward: passa gradiente como identidade (aproximação)

    Isso permite treinar com pesos quantizados enquanto mantém
    gradientes fluindo para os pesos latentes fp32.
    """

    @staticmethod
    def forward(ctx, W: torch.Tensor) -> torch.Tensor:
        """
        Quantiza pesos para ternário no forward.

        Args:
            W: Pesos fp32 [out_features, in_features]

        Returns:
            W_scaled: Pesos ternários escalados de volta
        """
        # Salva para backward (opcional, para clipping de gradientes)
        ctx.save_for_backward(W)

        # Quantiza
        W_q, scale = quantize_weights_ternary(W)

        # Retorna com escala para manter magnitude aproximada
        return W_q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        STE: passa gradiente diretamente (identidade).

        Opcionalmente pode clipar gradientes para estabilidade.
        """
        W, = ctx.saved_tensors

        # Passa gradiente como identidade
        grad_input = grad_output.clone()

        # Opcional: clipa gradientes onde |W| > threshold
        # Isso pode ajudar na estabilidade do treinamento
        # grad_input = grad_input * (W.abs() <= 1.5).float()

        return grad_input


def ste_ternary_quantize(W: torch.Tensor) -> torch.Tensor:
    """Aplica quantização ternária com STE."""
    return STETernaryFunction.apply(W)


class STEINT8Function(Function):
    """
    STE para quantização de ativações INT8.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int = 8) -> torch.Tensor:
        x_q, scale = quantize_activations_int8(x, bits)
        ctx.save_for_backward(scale)
        return x_q * scale  # Dequantiza imediatamente para manter fp32 no grafo

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # STE: passa gradiente como identidade
        return grad_output, None


def ste_int8_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Aplica quantização INT8 com STE."""
    return STEINT8Function.apply(x, bits)


# =============================================================================
# BITLINEAR - CAMADA LINEAR QUANTIZADA
# =============================================================================

class BitLinear(nn.Module):
    """
    Camada Linear com pesos ternários {-1, 0, +1} e ativações INT8.

    Substitui nn.Linear com quantização BitNet b1.58.
    Durante treinamento, mantém pesos latentes em fp32 e usa STE.
    Durante inferência, usa pesos pré-quantizados.

    Arquitetura:
    1. RMSNorm na entrada (SubLN do BitNet)
    2. Quantiza ativações para INT8
    3. Quantiza pesos para ternário
    4. Matmul (ou soma/subtração condicional em CPU)
    5. Escala resultado

    Args:
        in_features: Dimensão de entrada
        out_features: Dimensão de saída
        bias: Se usar bias (default False, BitNet não usa)
        activation_bits: Precisão das ativações (default 8)
        use_progressive_quant: Se usar warmup progressivo de quantização
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
        use_progressive_quant: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.use_progressive_quant = use_progressive_quant

        # Ratio de quantização: 0.0 = fp32 puro, 1.0 = ternário puro
        # Começa em 0 se progressive, senão já começa em 1
        self.quant_ratio = 0.0 if use_progressive_quant else 1.0

        # Pesos latentes em fp32 (para treinamento QAT)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # RMSNorm antes da quantização (SubLN architecture)
        self.norm = nn.RMSNorm(in_features, eps=1e-6)

        # Bias opcional (geralmente não usado em BitNet)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Buffers para inferência (pesos pré-quantizados)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('is_quantized', torch.tensor(False))

        # Inicialização dos pesos
        self._init_weights()

    def _init_weights(self):
        """
        Inicializa pesos para BitLinear.

        IMPORTANTE: Usa std=0.02 igual ao modelo fp32 para garantir que o início
        do treinamento (quando quant_ratio=0) tenha comportamento idêntico ao
        baseline. A quantização ternária se adapta a qualquer escala de pesos.
        """
        # Usa mesma inicialização do modelo fp32 para consistência no warmup
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def set_quant_ratio(self, ratio: float):
        """
        Define o ratio de quantização para warmup progressivo.

        Args:
            ratio: 0.0 = fp32 puro, 1.0 = ternário puro
                   Valores intermediários interpolam entre os dois
        """
        self.quant_ratio = max(0.0, min(1.0, ratio))

    def quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantiza pesos para ternário."""
        return quantize_weights_ternary(self.weight)

    def quantize_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantiza ativações para INT8."""
        return quantize_activations_int8(x, self.activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com quantização.

        Args:
            x: Input tensor [batch, seq, in_features]

        Returns:
            output: [batch, seq, out_features]
        """
        # 1. Normaliza input (SubLN)
        x = self.norm(x)

        if self.training:
            # TREINAMENTO: quantiza on-the-fly com STE
            return self._forward_train(x)
        else:
            # INFERÊNCIA: usa pesos pré-quantizados
            return self._forward_inference(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward durante treinamento com STE e warmup progressivo.

        Se quant_ratio < 1.0, interpola entre pesos fp32 e ternários:
        W_eff = (1 - quant_ratio) * W_fp32 + quant_ratio * W_ternary

        IMPORTANTE: Quando quant_ratio=0, usa fp32 PURO (sem quantizar
        ativações nem pesos). Isso garante que o início do warmup seja
        estável e equivalente ao treinamento fp32 normal.
        """
        # Caso especial: fp32 puro (sem quantização de nada)
        if self.quant_ratio <= 0.0:
            return F.linear(x, self.weight, self.bias)

        # Quantiza ativações com STE
        x_q, x_scale = self.quantize_activations(x)

        # Quantiza pesos com STE
        W_q, W_scale = self.quantize_weights()

        if self.quant_ratio >= 1.0:
            # Quantização completa (comportamento original)
            # STE trick: detach a diferença para que gradientes fluam para pesos originais
            W_ste = self.weight + (W_q * W_scale - self.weight).detach()
        else:
            # Interpolação progressiva entre fp32 e ternário
            # W_eff = (1 - r) * W_fp32 + r * W_ternary
            W_ternary = W_q * W_scale
            W_interpolated = (1 - self.quant_ratio) * self.weight + self.quant_ratio * W_ternary

            # STE: gradientes fluem para self.weight
            W_ste = self.weight + (W_interpolated - self.weight).detach()

        # Matmul (simulado em fp32 para compatibilidade de gradientes)
        # x_q * x_scale @ W_ste^T
        output = F.linear(x_q * x_scale, W_ste, self.bias)

        return output

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Forward durante inferência com pesos pré-quantizados."""
        # Garante que pesos estão quantizados
        if not self.is_quantized:
            self.prepare_for_inference()

        # Quantiza ativações
        x_q, x_scale = self.quantize_activations(x)

        # Matmul com pesos ternários
        # Em CPU otimizado: seria substituído por soma/subtração condicional
        output = F.linear(x_q, self.weight_quantized.float()) * x_scale * self.weight_scale

        if self.bias is not None:
            output = output + self.bias

        return output

    def prepare_for_inference(self):
        """
        Pré-computa pesos quantizados para inferência eficiente.
        Chamar após treinamento e antes de deploy.
        """
        with torch.no_grad():
            W_q, W_scale = self.quantize_weights()
            self.weight_quantized.copy_(W_q.to(torch.int8))
            self.weight_scale.copy_(W_scale)
            self.is_quantized.fill_(True)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, activation_bits={self.activation_bits}, '
                f'quant_ratio={self.quant_ratio:.2f}')


# =============================================================================
# BITLINEAR PARA INFERÊNCIA CPU (SEM MULTIPLICAÇÃO)
# =============================================================================

class BitLinearCPU(nn.Module):
    """
    BitLinear otimizado para inferência em CPU.

    Usa apenas adições/subtrações em vez de multiplicações:
    y = Σ(x onde W=+1) - Σ(x onde W=-1)

    Esta versão empacota pesos (4 por byte) e usa operações
    vetorizadas para máxima eficiência em CPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_bits: int = 8
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits

        # RMSNorm
        self.norm = nn.RMSNorm(in_features, eps=1e-6)

        # Pesos empacotados: 4 valores ternários por byte
        # Cada valor usa 2 bits: 00=0, 01=+1, 11=-1
        packed_size = (in_features + 3) // 4  # Arredonda para cima
        self.register_buffer('weight_packed', torch.zeros(out_features, packed_size, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.ones(1))

        # Máscaras pré-computadas para eficiência
        self.register_buffer('mask_pos', torch.zeros(out_features, in_features, dtype=torch.bool))
        self.register_buffer('mask_neg', torch.zeros(out_features, in_features, dtype=torch.bool))

    @classmethod
    def from_bitlinear(cls, bitlinear: BitLinear) -> 'BitLinearCPU':
        """
        Converte BitLinear treinado para BitLinearCPU.

        Args:
            bitlinear: BitLinear com pesos treinados

        Returns:
            BitLinearCPU pronto para inferência
        """
        cpu_layer = cls(
            bitlinear.in_features,
            bitlinear.out_features,
            bitlinear.activation_bits
        )

        # Copia normalização
        cpu_layer.norm.load_state_dict(bitlinear.norm.state_dict())

        # Quantiza pesos
        with torch.no_grad():
            W_q, W_scale = bitlinear.quantize_weights()

            # Cria máscaras
            cpu_layer.mask_pos.copy_(W_q == 1)
            cpu_layer.mask_neg.copy_(W_q == -1)
            cpu_layer.weight_scale.copy_(W_scale)

            # Empacota pesos (4 por byte)
            cpu_layer._pack_weights(W_q)

        return cpu_layer

    def _pack_weights(self, W_q: torch.Tensor):
        """
        Empacota pesos ternários: 4 valores por byte.

        Encoding:
        - 00 (0): peso = 0
        - 01 (1): peso = +1
        - 11 (3): peso = -1
        """
        out_f, in_f = W_q.shape
        packed_size = (in_f + 3) // 4

        # Mapeia {-1, 0, +1} para {3, 0, 1}
        W_mapped = W_q.clone()
        W_mapped[W_q == -1] = 3
        W_mapped[W_q == 1] = 1
        W_mapped[W_q == 0] = 0

        # Empacota
        for i in range(out_f):
            for j in range(packed_size):
                packed_val = 0
                for k in range(4):
                    idx = j * 4 + k
                    if idx < in_f:
                        packed_val |= (int(W_mapped[i, idx].item()) << (k * 2))
                self.weight_packed[i, j] = packed_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward otimizado: apenas adições e subtrações.

        y[o] = Σ(x[i] onde W[o,i]=+1) - Σ(x[i] onde W[o,i]=-1)
        """
        # Normaliza
        x = self.norm(x)

        # Quantiza ativações
        x_q, x_scale = quantize_activations_int8(x, self.activation_bits)
        x_scaled = x_q * x_scale

        batch, seq, _ = x_scaled.shape
        output = torch.zeros(batch, seq, self.out_features, device=x.device, dtype=x.dtype)

        # Operação principal: soma/subtração condicional
        # Esta versão é para clareza; versão otimizada usaria SIMD
        for o in range(self.out_features):
            # Soma onde W = +1
            sum_pos = x_scaled[..., self.mask_pos[o]].sum(dim=-1)
            # Soma onde W = -1
            sum_neg = x_scaled[..., self.mask_neg[o]].sum(dim=-1)
            # Resultado
            output[..., o] = sum_pos - sum_neg

        # Aplica escala dos pesos
        output = output * self.weight_scale

        return output


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def convert_linear_to_bitlinear(
    linear: nn.Linear,
    activation_bits: int = 8
) -> BitLinear:
    """
    Converte nn.Linear para BitLinear preservando pesos.

    Útil para converter modelos pré-treinados.
    """
    bitlinear = BitLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        activation_bits=activation_bits
    )

    # Copia pesos
    with torch.no_grad():
        bitlinear.weight.copy_(linear.weight)
        if linear.bias is not None and bitlinear.bias is not None:
            bitlinear.bias.copy_(linear.bias)

    return bitlinear


def count_ternary_params(model: nn.Module) -> dict:
    """
    Conta parâmetros ternários e não-ternários no modelo.
    """
    ternary_params = 0
    full_params = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            ternary_params += module.weight.numel()
        elif isinstance(module, nn.Linear):
            full_params += module.weight.numel()
            if module.bias is not None:
                full_params += module.bias.numel()
        elif isinstance(module, nn.Embedding):
            full_params += module.weight.numel()

    return {
        'ternary_params': ternary_params,
        'full_precision_params': full_params,
        'total_params': ternary_params + full_params,
        'ternary_memory_mb': ternary_params * 2 / 8 / 1024 / 1024,  # 2 bits por peso
        'full_memory_mb': full_params * 4 / 1024 / 1024,  # fp32
    }


def compute_sparsity(W_q: torch.Tensor) -> float:
    """Calcula esparsidade (fração de zeros) em pesos quantizados."""
    return (W_q == 0).float().mean().item()


def update_quant_ratio(
    model: nn.Module,
    step: int,
    warmup_steps: int,
    schedule: str = 'linear'
) -> float:
    """
    Atualiza o quant_ratio em todos os módulos BitLinear do modelo.

    Implementa warmup progressivo de quantização para estabilizar
    o treinamento de modelos maiores.

    Args:
        model: Modelo contendo BitLinear modules
        step: Step atual de treinamento
        warmup_steps: Número de steps para warmup completo
        schedule: Tipo de schedule ('linear', 'cosine', 'exponential')

    Returns:
        ratio: O quant_ratio atual aplicado

    Schedules:
        - linear: ratio = step / warmup_steps
        - cosine: ratio = 0.5 * (1 - cos(pi * step / warmup_steps))
        - exponential: ratio = 1 - exp(-3 * step / warmup_steps)
    """
    if warmup_steps <= 0:
        ratio = 1.0
    elif step >= warmup_steps:
        ratio = 1.0
    else:
        progress = step / warmup_steps

        if schedule == 'linear':
            ratio = progress
        elif schedule == 'cosine':
            # Começa devagar, acelera no meio, desacelera no fim
            ratio = 0.5 * (1 - math.cos(math.pi * progress))
        elif schedule == 'exponential':
            # Cresce rápido no início, desacelera no fim
            ratio = 1.0 - math.exp(-3.0 * progress)
        else:
            raise ValueError(f"Schedule desconhecido: {schedule}")

    # Aplica a todos os BitLinear
    for module in model.modules():
        if isinstance(module, BitLinear):
            module.set_quant_ratio(ratio)

    return ratio


def get_quant_ratio(model: nn.Module) -> Optional[float]:
    """
    Retorna o quant_ratio atual do modelo.

    Returns:
        ratio: O quant_ratio do primeiro BitLinear encontrado, ou None
    """
    for module in model.modules():
        if isinstance(module, BitLinear):
            return module.quant_ratio
    return None


# =============================================================================
# TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Teste de BitLinear")
    print("=" * 60)

    # Configuração
    batch_size = 2
    seq_len = 16
    in_features = 576
    out_features = 1536

    # Cria BitLinear
    bitlinear = BitLinear(in_features, out_features)

    # Input de teste
    x = torch.randn(batch_size, seq_len, in_features)

    # Teste forward (treinamento)
    print("\n[Treinamento]")
    bitlinear.train()
    y_train = bitlinear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_train.shape}")

    # Verifica quantização
    W_q, W_scale = bitlinear.quantize_weights()
    print(f"  Pesos únicos: {torch.unique(W_q).tolist()}")
    print(f"  Escala: {W_scale.item():.4f}")
    print(f"  Esparsidade: {compute_sparsity(W_q)*100:.1f}%")

    # Teste backward
    loss = y_train.sum()
    loss.backward()
    print(f"  Gradiente shape: {bitlinear.weight.grad.shape}")
    print(f"  Gradiente não-nulo: {(bitlinear.weight.grad != 0).float().mean()*100:.1f}%")

    # Teste forward (inferência)
    print("\n[Inferência]")
    bitlinear.eval()
    bitlinear.prepare_for_inference()
    y_infer = bitlinear(x)
    print(f"  Output shape: {y_infer.shape}")
    print(f"  Pesos quantizados dtype: {bitlinear.weight_quantized.dtype}")

    # Teste BitLinearCPU
    print("\n[CPU Otimizado]")
    bitlinear_cpu = BitLinearCPU.from_bitlinear(bitlinear)
    y_cpu = bitlinear_cpu(x)
    print(f"  Output shape: {y_cpu.shape}")

    # Diferença entre versões
    diff = (y_infer - y_cpu).abs().mean()
    print(f"  Diferença média (BitLinear vs CPU): {diff.item():.6f}")

    # Estatísticas de memória
    print("\n[Memória]")
    print(f"  Pesos fp32: {bitlinear.weight.numel() * 4 / 1024:.1f} KB")
    print(f"  Pesos ternários (2 bits): {bitlinear.weight.numel() * 2 / 8 / 1024:.1f} KB")
    print(f"  Redução: {(1 - 2/32)*100:.0f}%")

    # Teste warmup progressivo
    print("\n[Warmup Progressivo]")
    bitlinear_prog = BitLinear(in_features, out_features, use_progressive_quant=True)
    bitlinear_prog.train()
    x_test = torch.randn(1, 4, in_features)

    # Simula diferentes estágios do warmup
    for step, expected_ratio in [(0, 0.0), (50, 0.5), (100, 1.0), (150, 1.0)]:
        ratio = update_quant_ratio(bitlinear_prog, step, warmup_steps=100, schedule='linear')
        y = bitlinear_prog(x_test)
        print(f"  Step {step:3d}: quant_ratio = {ratio:.2f}, output_std = {y.std().item():.4f}")

    # Teste diferentes schedules
    print("\n[Schedules de Quantização]")
    for schedule in ['linear', 'cosine', 'exponential']:
        bitlinear_test = BitLinear(64, 64, use_progressive_quant=True)
        ratios = []
        for step in [0, 25, 50, 75, 100]:
            r = update_quant_ratio(bitlinear_test, step, warmup_steps=100, schedule=schedule)
            ratios.append(f"{r:.2f}")
        print(f"  {schedule:12s}: {' -> '.join(ratios)}")

    print("\n" + "=" * 60)
    print("Teste concluído com sucesso!")
    print("=" * 60)
