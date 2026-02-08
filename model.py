"""
Redes Preditivas Termodinâmicas (RPT)
=====================================
Uma nova arquitetura de IA baseada em princípios físicos fundamentais.

Autor: Cesar Favero
Data: Janeiro 2026
Licença: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import math


@dataclass
class RPTConfig:
    """Configuração da arquitetura RPT."""
    
    # Dimensões
    vocab_size: int = 49152  # SmolLM2 vocab size
    hidden_dim: int = 576    # SmolLM2-135M hidden dim
    num_layers: int = 30     # SmolLM2-135M layers
    num_heads: int = 9       # SmolLM2-135M heads
    intermediate_dim: int = 1536  # FFN intermediate
    max_seq_len: int = 2048
    
    # RPT específico
    boundary_dim: int = 72           # hidden_dim / 8
    num_regions: int = 16            # Regiões para codificação holográfica
    target_sparsity: float = 0.03    # 3% ativação
    
    # Dinâmica de equilíbrio
    free_phase_steps: int = 10
    nudged_phase_steps: int = 5
    state_lr: float = 0.1            # Learning rate para estados
    nudge_factor: float = 0.2        # β
    
    # Criticalidade
    criticality_gamma: float = 0.01
    initial_threshold: float = 1.0  # Aumentado de 0.1 para bloquear mais
    
    # Regularização
    sparsity_lambda: float = 0.001
    criticality_lambda: float = 0.01
    
    # Precisão
    precision_init: float = 1.0


class PredictiveLayer(nn.Module):
    """
    Camada preditiva individual.
    
    Implementa:
    - Modelo generativo (top-down predictions)
    - Unidades de erro (bottom-up errors)
    - Gating esparso
    - Precisão adaptativa
    """
    
    def __init__(self, config: RPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        
        # Modelo generativo g_θ (gera predições top-down)
        self.generative_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        )
        
        # Projeções para codificação de fronteira (holográfica)
        self.boundary_proj_mean = nn.Linear(config.hidden_dim, config.boundary_dim, bias=False)
        self.boundary_proj_var = nn.Linear(config.hidden_dim, config.boundary_dim, bias=False)
        
        # Query para atenção via fronteira
        self.boundary_query = nn.Linear(config.hidden_dim, config.boundary_dim, bias=False)
        self.boundary_retrieve = nn.Linear(config.boundary_dim, config.hidden_dim, bias=False)
        
        # Precisão (inverso da variância esperada) - aprendível
        self.log_precision = nn.Parameter(
            torch.ones(config.hidden_dim) * math.log(config.precision_init)
        )
        
        # Threshold adaptativo para esparsidade
        self.register_buffer(
            'threshold', 
            torch.ones(config.hidden_dim) * config.initial_threshold
        )
        
        # Inibição local (para balanço E/I)
        self.inhibition = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        # Normalização
        self.norm = nn.RMSNorm(config.hidden_dim, eps=1e-6)
        
    @property
    def precision(self) -> torch.Tensor:
        """Retorna precisão (sempre positiva)."""
        return torch.exp(self.log_precision)
    
    def generate_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """Gera predição top-down para camada inferior."""
        return self.generative_model(self.norm(state))
    
    def compute_error(
        self, 
        state: torch.Tensor, 
        prediction: torch.Tensor,
        precision: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computa erro de predição ponderado pela precisão.
        
        ε = π ⊙ (s - p)
        """
        if precision is None:
            precision = self.precision
        
        error = state - prediction
        weighted_error = precision.unsqueeze(0).unsqueeze(0) * error
        return weighted_error
    
    def apply_sparse_gating(
        self,
        error: torch.Tensor,
        update_threshold: bool = False,
        target_sparsity: float = 0.03
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica gating esparso - só propaga erros acima do threshold.
        Usa percentil dinamico para garantir esparsidade correta.

        Retorna:
            - erro esparso
            - máscara de ativação
        """
        # Magnitude do erro por dimensão
        error_magnitude = torch.abs(error)

        # Usa percentil para threshold dinamico (garante esparsidade exata)
        # Se target_sparsity = 0.03, queremos o percentil 97
        percentile = 1.0 - target_sparsity

        # Calcula threshold como percentil da magnitude dos erros
        flat_magnitude = error_magnitude.flatten()
        if flat_magnitude.numel() > 0:
            dynamic_threshold = torch.quantile(flat_magnitude, percentile)
        else:
            dynamic_threshold = self.threshold.mean()

        # Máscara binária usando threshold dinamico
        mask = (error_magnitude > dynamic_threshold).float()

        # Erro esparso
        sparse_error = error * mask

        # Atualiza threshold buffer para referencia (opcional)
        if update_threshold and self.training:
            with torch.no_grad():
                self.threshold.fill_(dynamic_threshold.item())

        return sparse_error, mask
    
    def compute_boundary_encoding(
        self,
        states: torch.Tensor,  # [batch, seq, hidden]
        region_size: int = None
    ) -> torch.Tensor:
        """
        Codificação holográfica de fronteira.

        Comprime informação de uma região em estatísticas de fronteira.
        """
        batch_size, seq_len, hidden = states.shape

        if region_size is None:
            region_size = max(1, seq_len // self.config.num_regions)

        num_regions = max(1, seq_len // region_size)
        
        # Reshape para regiões
        states_regions = states[:, :num_regions * region_size].view(
            batch_size, num_regions, region_size, hidden
        )
        
        # Média por região
        region_mean = states_regions.mean(dim=2)  # [batch, num_regions, hidden]
        
        # Variância por região
        region_var = states_regions.var(dim=2)  # [batch, num_regions, hidden]
        
        # Projeção para espaço de fronteira
        boundary_mean = self.boundary_proj_mean(region_mean)  # [batch, num_regions, boundary_dim]
        boundary_var = self.boundary_proj_var(region_var)
        
        # Combinação
        boundary = boundary_mean + boundary_var
        
        return boundary
    
    def boundary_attention(
        self,
        query_state: torch.Tensor,     # [batch, seq, hidden]
        boundary_keys: torch.Tensor,    # [batch, num_regions, boundary_dim]
        states: torch.Tensor            # [batch, seq, hidden] estados originais
    ) -> torch.Tensor:
        """
        Atenção via fronteiras holográficas.
        
        Em vez de O(n²), consultamos fronteiras O(n·R).
        """
        batch_size, seq_len, _ = query_state.shape
        num_regions = boundary_keys.shape[1]
        region_size = seq_len // num_regions
        
        # Query projection
        q = self.boundary_query(query_state)  # [batch, seq, boundary_dim]
        
        # Attention scores com fronteiras
        # [batch, seq, boundary_dim] @ [batch, boundary_dim, num_regions]
        scores = torch.matmul(q, boundary_keys.transpose(-1, -2))
        scores = scores / math.sqrt(self.config.boundary_dim)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq, num_regions]
        
        # Recuperar informação das regiões relevantes
        output = self.boundary_retrieve(
            torch.matmul(attn_weights, boundary_keys)
        )
        
        return output
    
    def apply_inhibition(self, state: torch.Tensor) -> torch.Tensor:
        """Aplica inibição local para balanço E/I."""
        inhibition = F.relu(self.inhibition(state))
        return state - 0.1 * inhibition
    
    def forward(
        self,
        state: torch.Tensor,
        lower_state: Optional[torch.Tensor] = None,
        upper_prediction: Optional[torch.Tensor] = None,
        upper_precision: Optional[torch.Tensor] = None,
        compute_boundary: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass de uma camada preditiva.
        
        Args:
            state: Estado atual desta camada [batch, seq, hidden]
            lower_state: Estado da camada inferior (para computar erro de predição)
            upper_prediction: Predição vinda da camada superior
            upper_precision: Precisão da camada superior
            compute_boundary: Se deve computar codificação de fronteira
            
        Returns:
            Dict com prediction, error, sparse_error, mask, boundary, updated_state
        """
        outputs = {}
        
        # 1. Gera predição para camada inferior
        prediction = self.generate_prediction(state)
        outputs['prediction'] = prediction
        
        # 2. Computa erro bottom-up (se temos estado inferior)
        if lower_state is not None:
            error_down = self.compute_error(lower_state, prediction, self.precision)
            sparse_error_down, mask_down = self.apply_sparse_gating(
                error_down, 
                update_threshold=self.training
            )
            outputs['error_down'] = error_down
            outputs['sparse_error_down'] = sparse_error_down
            outputs['mask_down'] = mask_down
        
        # 3. Computa erro top-down (se temos predição superior)
        if upper_prediction is not None:
            error_up = self.compute_error(state, upper_prediction, upper_precision)
            sparse_error_up, mask_up = self.apply_sparse_gating(error_up)
            outputs['error_up'] = error_up
            outputs['sparse_error_up'] = sparse_error_up
            outputs['mask_up'] = mask_up
        
        # 4. Codificação de fronteira holográfica
        if compute_boundary:
            boundary = self.compute_boundary_encoding(state)
            outputs['boundary'] = boundary
        
        # 5. Aplica inibição
        outputs['inhibited_state'] = self.apply_inhibition(state)
        
        return outputs


class RPTModel(nn.Module):
    """
    Modelo completo de Redes Preditivas Termodinâmicas.
    
    Implementa:
    - Hierarquia de camadas preditivas
    - Dinâmica de equilíbrio (fase livre e forçada)
    - Aprendizado por propagação de equilíbrio
    - Manutenção de criticalidade
    """
    
    def __init__(self, config: RPTConfig):
        super().__init__()
        self.config = config
        
        # Embedding de entrada
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Camadas preditivas
        self.layers = nn.ModuleList([
            PredictiveLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Camada de saída
        self.output_norm = nn.RMSNorm(config.hidden_dim, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Estado de criticalidade
        self.register_buffer('branching_ratio', torch.ones(1))
        self.register_buffer('activity_history', torch.zeros(100))
        self.activity_idx = 0
        
        # Inicialização
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def initialize_states(
        self, 
        batch_size: int, 
        seq_len: int, 
        device: torch.device
    ) -> List[torch.Tensor]:
        """Inicializa estados de todas as camadas."""
        states = []
        for _ in range(self.config.num_layers):
            state = torch.randn(
                batch_size, seq_len, self.config.hidden_dim,
                device=device
            ) * 0.01
            states.append(state)
        return states
    
    def compute_state_gradient(
        self,
        layer_idx: int,
        state: torch.Tensor,
        error_down: Optional[torch.Tensor],
        error_up: Optional[torch.Tensor],
        layer: PredictiveLayer
    ) -> torch.Tensor:
        """
        Computa gradiente do estado para minimizar energia livre local.
        
        ∂F/∂s = -W_g^T @ (π @ ε_down) + π_up @ ε_up + λ·sign(s)
        """
        grad = torch.zeros_like(state)
        
        # Contribuição do erro de baixo
        if error_down is not None:
            # Gradiente através do modelo generativo
            with torch.enable_grad():
                state_temp = state.detach().requires_grad_(True)
                pred = layer.generate_prediction(state_temp)
                # Pseudo-gradiente
                grad_down = torch.autograd.grad(
                    pred, state_temp, 
                    grad_outputs=error_down,
                    retain_graph=False
                )[0]
            grad = grad - grad_down
        
        # Contribuição do erro de cima
        if error_up is not None:
            grad = grad + error_up
        
        # Regularização L1 para esparsidade
        grad = grad + self.config.sparsity_lambda * torch.sign(state)
        
        return grad
    
    def free_phase(
        self,
        input_embeds: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Fase livre: sistema evolui até equilíbrio sem supervisão.
        
        Returns:
            states: Lista de estados finais por camada
            metrics: Métricas de convergência
        """
        if num_steps is None:
            num_steps = self.config.free_phase_steps
        
        batch_size, seq_len, _ = input_embeds.shape
        device = input_embeds.device
        
        # Inicializa estados
        states = self.initialize_states(batch_size, seq_len, device)
        states[0] = input_embeds  # Primeira camada recebe input
        
        metrics = {
            'energy_history': [],
            'sparsity_history': [],
            'branching_history': []
        }
        
        for step in range(num_steps):
            total_energy = 0.0
            total_active = 0
            total_units = 0
            prev_active = getattr(self, '_prev_active', 0)
            
            # Processa cada camada
            new_states = [None] * self.config.num_layers
            
            for l in range(self.config.num_layers):
                layer = self.layers[l]
                state = states[l]
                
                # Estado da camada inferior
                lower_state = states[l-1] if l > 0 else input_embeds
                
                # Predição da camada superior
                upper_pred = None
                upper_prec = None
                if l < self.config.num_layers - 1:
                    upper_layer = self.layers[l+1]
                    upper_pred = upper_layer.generate_prediction(states[l+1])
                    upper_prec = upper_layer.precision
                
                # Forward da camada
                outputs = layer(
                    state=state,
                    lower_state=lower_state,
                    upper_prediction=upper_pred,
                    upper_precision=upper_prec,
                    compute_boundary=(step == num_steps - 1)  # Só no último step
                )
                
                # Computa gradiente do estado
                grad = self.compute_state_gradient(
                    layer_idx=l,
                    state=state,
                    error_down=outputs.get('sparse_error_down'),
                    error_up=outputs.get('sparse_error_up'),
                    layer=layer
                )
                
                # Atualiza estado
                new_state = state - self.config.state_lr * grad
                new_state = layer.apply_inhibition(new_state)
                new_states[l] = new_state
                
                # Métricas
                if 'sparse_error_down' in outputs:
                    energy = 0.5 * (outputs['sparse_error_down'] ** 2).sum()
                    total_energy += energy.item()
                    
                    active = (outputs['mask_down'] > 0).sum().item()
                    total_active += active
                    total_units += outputs['mask_down'].numel()
            
            # Atualiza estados
            states = new_states
            
            # Computa branching ratio
            if prev_active > 0:
                branching = total_active / prev_active
            else:
                branching = 1.0
            self._prev_active = total_active
            
            # Atualiza thresholds para manter criticalidade
            if self.training:
                self._update_criticality(branching)
            
            # Registra métricas
            metrics['energy_history'].append(total_energy)
            metrics['sparsity_history'].append(
                total_active / max(total_units, 1)
            )
            metrics['branching_history'].append(branching)
        
        return states, metrics
    
    def nudged_phase(
        self,
        states: List[torch.Tensor],
        target: torch.Tensor,
        input_embeds: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Fase forçada: última camada é "empurrada" para o target.
        
        Args:
            states: Estados da fase livre
            target: Target de supervisão [batch, seq, hidden] ou [batch, seq, vocab]
            input_embeds: Embeddings de entrada
            
        Returns:
            states_nudged: Estados após nudging
            metrics: Métricas
        """
        if num_steps is None:
            num_steps = self.config.nudged_phase_steps
        
        # Copia estados
        states_nudged = [s.clone() for s in states]
        
        # Se target é logits, converte para embeddings via soft lookup
        if target.shape[-1] == self.config.vocab_size:
            target_probs = F.softmax(target, dim=-1)
            target_embed = target_probs @ self.embed_tokens.weight
        else:
            target_embed = target
        
        # Aplica nudge na última camada
        beta = self.config.nudge_factor
        states_nudged[-1] = states_nudged[-1] + beta * (target_embed - states_nudged[-1])
        
        metrics = {'energy_history': []}
        
        # Re-equilibra
        for step in range(num_steps):
            total_energy = 0.0
            new_states = [None] * self.config.num_layers
            
            for l in range(self.config.num_layers):
                layer = self.layers[l]
                state = states_nudged[l]
                
                lower_state = states_nudged[l-1] if l > 0 else input_embeds
                
                upper_pred = None
                upper_prec = None
                if l < self.config.num_layers - 1:
                    upper_layer = self.layers[l+1]
                    upper_pred = upper_layer.generate_prediction(states_nudged[l+1])
                    upper_prec = upper_layer.precision
                
                outputs = layer(
                    state=state,
                    lower_state=lower_state,
                    upper_prediction=upper_pred,
                    upper_precision=upper_prec,
                    compute_boundary=False
                )
                
                grad = self.compute_state_gradient(
                    layer_idx=l,
                    state=state,
                    error_down=outputs.get('sparse_error_down'),
                    error_up=outputs.get('sparse_error_up'),
                    layer=layer
                )
                
                # Na última camada, mantém nudge
                if l == self.config.num_layers - 1:
                    new_state = state - self.config.state_lr * grad
                    new_state = new_state + beta * (target_embed - new_state)
                else:
                    new_state = state - self.config.state_lr * grad
                
                new_states[l] = new_state
                
                if 'sparse_error_down' in outputs:
                    energy = 0.5 * (outputs['sparse_error_down'] ** 2).sum()
                    total_energy += energy.item()
            
            states_nudged = new_states
            metrics['energy_history'].append(total_energy)
        
        return states_nudged, metrics
    
    def _update_criticality(self, branching_ratio: float):
        """Atualiza thresholds para manter sistema na criticalidade."""
        gamma = self.config.criticality_gamma
        adjustment = math.exp(gamma * (branching_ratio - 1.0))
        
        for layer in self.layers:
            layer.threshold.mul_(adjustment)
            layer.threshold.clamp_(min=1e-6, max=10.0)
        
        self.branching_ratio.fill_(branching_ratio)
    
    def compute_contrastive_update(
        self,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computa atualização Hebbiana contrastiva.
        
        ΔW = η * (s_nudged ⊗ s_nudged_lower - s_free ⊗ s_free_lower)
        """
        updates = {}
        
        for l, layer in enumerate(self.layers):
            if l == 0:
                continue  # Primeira camada não tem lower
            
            s_free = states_free[l]
            s_free_lower = states_free[l-1]
            s_nudged = states_nudged[l]
            s_nudged_lower = states_nudged[l-1]
            
            # Correlação fase nudged
            corr_nudged = torch.einsum('bsi,bsj->ij', s_nudged, s_nudged_lower)
            
            # Correlação fase livre  
            corr_free = torch.einsum('bsi,bsj->ij', s_free, s_free_lower)
            
            # Diferença (gradiente via Equilibrium Propagation)
            delta = (corr_nudged - corr_free) / (s_free.shape[0] * s_free.shape[1])
            
            updates[f'layer_{l}_delta'] = delta
        
        return updates
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass completo.
        
        Args:
            input_ids: Token IDs [batch, seq]
            labels: Labels para treinamento [batch, seq]
            
        Returns:
            Dict com logits, loss, métricas
        """
        # Embedding
        input_embeds = self.embed_tokens(input_ids)
        
        # Fase livre
        states_free, free_metrics = self.free_phase(input_embeds)
        
        # Output
        final_state = states_free[-1]
        final_state = self.output_norm(final_state)
        logits = self.lm_head(final_state)
        
        outputs = {
            'logits': logits,
            'states': states_free,
            'free_metrics': free_metrics
        }
        
        # Treinamento
        if labels is not None:
            # Target embeddings
            target_embeds = self.embed_tokens(labels)
            
            # Fase nudged
            states_nudged, nudged_metrics = self.nudged_phase(
                states_free, target_embeds, input_embeds
            )
            
            # Contrastive update (para logging, aplicação real em optimizer)
            contrastive_updates = self.compute_contrastive_update(
                states_free, states_nudged
            )
            
            # Loss padrão (cross-entropy)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Loss de energia livre
            free_energy_loss = sum(free_metrics['energy_history']) / len(free_metrics['energy_history'])
            
            # Loss de criticalidade
            criticality_loss = self.config.criticality_lambda * (
                (self.branching_ratio - 1.0) ** 2
            )
            
            # Loss total
            total_loss = loss + 0.01 * free_energy_loss + criticality_loss
            
            outputs['loss'] = total_loss
            outputs['ce_loss'] = loss
            outputs['free_energy_loss'] = free_energy_loss
            outputs['nudged_metrics'] = nudged_metrics
            outputs['contrastive_updates'] = contrastive_updates
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Geração autoregressiva."""
        for _ in range(max_new_tokens):
            # Trunca se necessário
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward
            outputs = self.forward(idx_cond)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# =============================================================================
# VERSÃO BITNET (QUANTIZADA)
# =============================================================================

try:
    from bitnet import BitLinear, BitLinearCPU, count_ternary_params
    HAS_BITNET = True
except ImportError:
    HAS_BITNET = False
    print("[!] Módulo bitnet não encontrado. RPTModelBitNet não disponível.")


class PredictiveLayerBitNet(nn.Module):
    """
    Camada preditiva com quantização BitNet b1.58.

    Usa pesos ternários {-1, 0, +1} nas camadas lineares
    para eficiência em CPU (elimina multiplicações).

    Camadas quantizadas:
    - generative_model (up e down projection)
    - boundary projections
    - inhibition

    Não quantizadas:
    - log_precision (parâmetro dinâmico)
    - threshold (buffer)
    - norm (RMSNorm)
    """

    def __init__(self, config: RPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim

        if not HAS_BITNET:
            raise ImportError("Módulo bitnet necessário para PredictiveLayerBitNet")

        # Modelo generativo QUANTIZADO
        self.generative_model = nn.Sequential(
            BitLinear(config.hidden_dim, config.intermediate_dim, bias=False),
            nn.GELU(),
            BitLinear(config.intermediate_dim, config.hidden_dim, bias=False)
        )

        # Projeções de fronteira QUANTIZADAS
        self.boundary_proj_mean = BitLinear(config.hidden_dim, config.boundary_dim, bias=False)
        self.boundary_proj_var = BitLinear(config.hidden_dim, config.boundary_dim, bias=False)

        # Query para atenção via fronteira QUANTIZADA
        self.boundary_query = BitLinear(config.hidden_dim, config.boundary_dim, bias=False)
        self.boundary_retrieve = BitLinear(config.boundary_dim, config.hidden_dim, bias=False)

        # Inibição QUANTIZADA
        self.inhibition = BitLinear(config.hidden_dim, config.hidden_dim, bias=False)

        # Precisão (NÃO quantizada - parâmetro dinâmico)
        self.log_precision = nn.Parameter(
            torch.ones(config.hidden_dim) * math.log(config.precision_init)
        )

        # Threshold (NÃO quantizado - buffer)
        self.register_buffer(
            'threshold',
            torch.ones(config.hidden_dim) * config.initial_threshold
        )

        # Normalização (NÃO quantizada)
        self.norm = nn.RMSNorm(config.hidden_dim, eps=1e-6)

    @property
    def precision(self) -> torch.Tensor:
        """Retorna precisão (sempre positiva)."""
        return torch.exp(self.log_precision)

    def generate_prediction(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gera predição top-down para camada inferior.

        NOTA: Não aplicamos self.norm() aqui porque BitLinear já tem
        RMSNorm interno (SubLN architecture). Normalização dupla causa
        instabilidade em modelos maiores.
        """
        return self.generative_model(state)

    def compute_error(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
        precision: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computa erro de predição ponderado pela precisão."""
        if precision is None:
            precision = self.precision

        error = state - prediction
        weighted_error = precision.unsqueeze(0).unsqueeze(0) * error
        return weighted_error

    def apply_sparse_gating(
        self,
        error: torch.Tensor,
        update_threshold: bool = False,
        target_sparsity: float = 0.03
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica gating esparso usando percentil dinâmico."""
        error_magnitude = torch.abs(error)
        percentile = 1.0 - target_sparsity

        flat_magnitude = error_magnitude.flatten()
        if flat_magnitude.numel() > 0:
            dynamic_threshold = torch.quantile(flat_magnitude, percentile)
        else:
            dynamic_threshold = self.threshold.mean()

        mask = (error_magnitude > dynamic_threshold).float()
        sparse_error = error * mask

        if update_threshold and self.training:
            with torch.no_grad():
                self.threshold.fill_(dynamic_threshold.item())

        return sparse_error, mask

    def compute_boundary_encoding(
        self,
        states: torch.Tensor,
        region_size: int = None
    ) -> torch.Tensor:
        """Codificação holográfica de fronteira."""
        batch_size, seq_len, hidden = states.shape

        if region_size is None:
            region_size = max(1, seq_len // self.config.num_regions)

        num_regions = max(1, seq_len // region_size)

        states_regions = states[:, :num_regions * region_size].view(
            batch_size, num_regions, region_size, hidden
        )

        region_mean = states_regions.mean(dim=2)
        region_var = states_regions.var(dim=2)

        boundary_mean = self.boundary_proj_mean(region_mean)
        boundary_var = self.boundary_proj_var(region_var)

        boundary = boundary_mean + boundary_var

        return boundary

    def apply_inhibition(self, state: torch.Tensor) -> torch.Tensor:
        """Aplica inibição local para balanço E/I."""
        inhibition = F.relu(self.inhibition(state))
        return state - 0.1 * inhibition

    def forward(
        self,
        state: torch.Tensor,
        lower_state: Optional[torch.Tensor] = None,
        upper_prediction: Optional[torch.Tensor] = None,
        upper_precision: Optional[torch.Tensor] = None,
        compute_boundary: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass de uma camada preditiva BitNet."""
        outputs = {}

        prediction = self.generate_prediction(state)
        outputs['prediction'] = prediction

        if lower_state is not None:
            error_down = self.compute_error(lower_state, prediction, self.precision)
            sparse_error_down, mask_down = self.apply_sparse_gating(
                error_down,
                update_threshold=self.training
            )
            outputs['error_down'] = error_down
            outputs['sparse_error_down'] = sparse_error_down
            outputs['mask_down'] = mask_down

        if upper_prediction is not None:
            error_up = self.compute_error(state, upper_prediction, upper_precision)
            sparse_error_up, mask_up = self.apply_sparse_gating(error_up)
            outputs['error_up'] = error_up
            outputs['sparse_error_up'] = sparse_error_up
            outputs['mask_up'] = mask_up

        if compute_boundary:
            boundary = self.compute_boundary_encoding(state)
            outputs['boundary'] = boundary

        outputs['inhibited_state'] = self.apply_inhibition(state)

        return outputs


class RPTModelBitNet(nn.Module):
    """
    Modelo RPT com quantização BitNet b1.58.

    Pesos ternários {-1, 0, +1} para eficiência em CPU.
    Mantém embedding e lm_head em precisão total.

    Benefícios:
    - 71-82% redução de energia em CPU
    - 2.37-6.17x speedup em inferência
    - 20x redução de memória para pesos lineares
    """

    def __init__(self, config: RPTConfig):
        super().__init__()
        self.config = config

        if not HAS_BITNET:
            raise ImportError("Módulo bitnet necessário para RPTModelBitNet")

        # Embedding NÃO QUANTIZADO (crítico para qualidade)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Camadas preditivas QUANTIZADAS
        self.layers = nn.ModuleList([
            PredictiveLayerBitNet(config, i) for i in range(config.num_layers)
        ])

        # Camada de saída NÃO QUANTIZADA
        self.output_norm = nn.RMSNorm(config.hidden_dim, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # Nota: lm_head mantido em fp para qualidade de output

        # Estado de criticalidade
        self.register_buffer('branching_ratio', torch.ones(1))
        self.register_buffer('activity_history', torch.zeros(100))
        self.activity_idx = 0

        # Inicialização
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initialize_states(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> List[torch.Tensor]:
        """Inicializa estados de todas as camadas."""
        states = []
        for _ in range(self.config.num_layers):
            state = torch.randn(
                batch_size, seq_len, self.config.hidden_dim,
                device=device
            ) * 0.01
            states.append(state)
        return states

    def compute_state_gradient(
        self,
        layer_idx: int,
        state: torch.Tensor,
        error_down: Optional[torch.Tensor],
        error_up: Optional[torch.Tensor],
        layer: PredictiveLayerBitNet
    ) -> torch.Tensor:
        """Computa gradiente do estado para minimizar energia livre local."""
        grad = torch.zeros_like(state)

        if error_down is not None:
            with torch.enable_grad():
                state_temp = state.detach().requires_grad_(True)
                pred = layer.generate_prediction(state_temp)
                grad_down = torch.autograd.grad(
                    pred, state_temp,
                    grad_outputs=error_down,
                    retain_graph=False
                )[0]
            grad = grad - grad_down

        if error_up is not None:
            grad = grad + error_up

        grad = grad + self.config.sparsity_lambda * torch.sign(state)

        return grad

    def free_phase(
        self,
        input_embeds: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Fase livre: sistema evolui até equilíbrio."""
        if num_steps is None:
            num_steps = self.config.free_phase_steps

        batch_size, seq_len, _ = input_embeds.shape
        device = input_embeds.device

        states = self.initialize_states(batch_size, seq_len, device)
        states[0] = input_embeds

        metrics = {
            'energy_history': [],
            'sparsity_history': [],
            'branching_history': []
        }

        for step in range(num_steps):
            total_energy = 0.0
            total_active = 0
            total_units = 0
            prev_active = getattr(self, '_prev_active', 0)

            new_states = [None] * self.config.num_layers

            for l in range(self.config.num_layers):
                layer = self.layers[l]
                state = states[l]

                lower_state = states[l-1] if l > 0 else input_embeds

                upper_pred = None
                upper_prec = None
                if l < self.config.num_layers - 1:
                    upper_layer = self.layers[l+1]
                    upper_pred = upper_layer.generate_prediction(states[l+1])
                    upper_prec = upper_layer.precision

                outputs = layer(
                    state=state,
                    lower_state=lower_state,
                    upper_prediction=upper_pred,
                    upper_precision=upper_prec,
                    compute_boundary=(step == num_steps - 1)
                )

                grad = self.compute_state_gradient(
                    layer_idx=l,
                    state=state,
                    error_down=outputs.get('sparse_error_down'),
                    error_up=outputs.get('sparse_error_up'),
                    layer=layer
                )

                new_state = state - self.config.state_lr * grad
                new_state = layer.apply_inhibition(new_state)
                new_states[l] = new_state

                if 'sparse_error_down' in outputs:
                    energy = 0.5 * (outputs['sparse_error_down'] ** 2).sum()
                    total_energy += energy.item()

                    active = (outputs['mask_down'] > 0).sum().item()
                    total_active += active
                    total_units += outputs['mask_down'].numel()

            states = new_states

            if prev_active > 0:
                branching = total_active / prev_active
            else:
                branching = 1.0
            self._prev_active = total_active

            if self.training:
                self._update_criticality(branching)

            metrics['energy_history'].append(total_energy)
            metrics['sparsity_history'].append(
                total_active / max(total_units, 1)
            )
            metrics['branching_history'].append(branching)

        return states, metrics

    def nudged_phase(
        self,
        states: List[torch.Tensor],
        target: torch.Tensor,
        input_embeds: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], Dict]:
        """Fase forçada: última camada é empurrada para target."""
        if num_steps is None:
            num_steps = self.config.nudged_phase_steps

        states_nudged = [s.clone() for s in states]

        if target.shape[-1] == self.config.vocab_size:
            target_probs = F.softmax(target, dim=-1)
            target_embed = target_probs @ self.embed_tokens.weight
        else:
            target_embed = target

        beta = self.config.nudge_factor
        states_nudged[-1] = states_nudged[-1] + beta * (target_embed - states_nudged[-1])

        metrics = {'energy_history': []}

        for step in range(num_steps):
            total_energy = 0.0
            new_states = [None] * self.config.num_layers

            for l in range(self.config.num_layers):
                layer = self.layers[l]
                state = states_nudged[l]

                lower_state = states_nudged[l-1] if l > 0 else input_embeds

                upper_pred = None
                upper_prec = None
                if l < self.config.num_layers - 1:
                    upper_layer = self.layers[l+1]
                    upper_pred = upper_layer.generate_prediction(states_nudged[l+1])
                    upper_prec = upper_layer.precision

                outputs = layer(
                    state=state,
                    lower_state=lower_state,
                    upper_prediction=upper_pred,
                    upper_precision=upper_prec,
                    compute_boundary=False
                )

                grad = self.compute_state_gradient(
                    layer_idx=l,
                    state=state,
                    error_down=outputs.get('sparse_error_down'),
                    error_up=outputs.get('sparse_error_up'),
                    layer=layer
                )

                if l == self.config.num_layers - 1:
                    new_state = state - self.config.state_lr * grad
                    new_state = new_state + beta * (target_embed - new_state)
                else:
                    new_state = state - self.config.state_lr * grad

                new_states[l] = new_state

                if 'sparse_error_down' in outputs:
                    energy = 0.5 * (outputs['sparse_error_down'] ** 2).sum()
                    total_energy += energy.item()

            states_nudged = new_states
            metrics['energy_history'].append(total_energy)

        return states_nudged, metrics

    def _update_criticality(self, branching_ratio: float):
        """Atualiza thresholds para manter sistema na criticalidade."""
        gamma = self.config.criticality_gamma
        adjustment = math.exp(gamma * (branching_ratio - 1.0))

        for layer in self.layers:
            layer.threshold.mul_(adjustment)
            layer.threshold.clamp_(min=1e-6, max=10.0)

        self.branching_ratio.fill_(branching_ratio)

    def compute_contrastive_update(
        self,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computa atualização Hebbiana contrastiva."""
        updates = {}

        for l, layer in enumerate(self.layers):
            if l == 0:
                continue

            s_free = states_free[l]
            s_free_lower = states_free[l-1]
            s_nudged = states_nudged[l]
            s_nudged_lower = states_nudged[l-1]

            corr_nudged = torch.einsum('bsi,bsj->ij', s_nudged, s_nudged_lower)
            corr_free = torch.einsum('bsi,bsj->ij', s_free, s_free_lower)

            delta = (corr_nudged - corr_free) / (s_free.shape[0] * s_free.shape[1])

            updates[f'layer_{l}_delta'] = delta

        return updates

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass completo."""
        input_embeds = self.embed_tokens(input_ids)

        states_free, free_metrics = self.free_phase(input_embeds)

        final_state = states_free[-1]
        final_state = self.output_norm(final_state)
        logits = self.lm_head(final_state)

        outputs = {
            'logits': logits,
            'states': states_free,
            'free_metrics': free_metrics
        }

        if labels is not None:
            target_embeds = self.embed_tokens(labels)

            states_nudged, nudged_metrics = self.nudged_phase(
                states_free, target_embeds, input_embeds
            )

            contrastive_updates = self.compute_contrastive_update(
                states_free, states_nudged
            )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

            free_energy_loss = sum(free_metrics['energy_history']) / len(free_metrics['energy_history'])

            criticality_loss = self.config.criticality_lambda * (
                (self.branching_ratio - 1.0) ** 2
            )

            total_loss = loss + 0.01 * free_energy_loss + criticality_loss

            outputs['loss'] = total_loss
            outputs['ce_loss'] = loss
            outputs['free_energy_loss'] = free_energy_loss
            outputs['nudged_metrics'] = nudged_metrics
            outputs['contrastive_updates'] = contrastive_updates

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Geração autoregressiva."""
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            outputs = self.forward(idx_cond)
            logits = outputs['logits'][:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def prepare_for_inference(self):
        """Prepara modelo para inferência CPU eficiente."""
        self.eval()
        for module in self.modules():
            if hasattr(module, 'prepare_for_inference'):
                module.prepare_for_inference()


def convert_rpt_to_bitnet(model: RPTModel) -> RPTModelBitNet:
    """
    Converte modelo RPT fp32 para RPTModelBitNet.

    Copia pesos preservando valores (quantização acontece no forward).
    """
    if not HAS_BITNET:
        raise ImportError("Módulo bitnet necessário para conversão")

    bitnet_model = RPTModelBitNet(model.config)

    # Copia embedding e lm_head (não quantizados)
    bitnet_model.embed_tokens.load_state_dict(model.embed_tokens.state_dict())
    bitnet_model.lm_head.load_state_dict(model.lm_head.state_dict())
    bitnet_model.output_norm.load_state_dict(model.output_norm.state_dict())

    # Copia buffers de criticalidade
    bitnet_model.branching_ratio.copy_(model.branching_ratio)
    bitnet_model.activity_history.copy_(model.activity_history)

    # Copia camadas
    for l in range(model.config.num_layers):
        src_layer = model.layers[l]
        dst_layer = bitnet_model.layers[l]

        # Copia pesos do generative_model
        for i, (src_mod, dst_mod) in enumerate(zip(
            src_layer.generative_model,
            dst_layer.generative_model
        )):
            if hasattr(src_mod, 'weight') and hasattr(dst_mod, 'weight'):
                dst_mod.weight.data.copy_(src_mod.weight.data)

        # Copia outros pesos
        for name in ['boundary_proj_mean', 'boundary_proj_var',
                     'boundary_query', 'boundary_retrieve', 'inhibition']:
            src_mod = getattr(src_layer, name)
            dst_mod = getattr(dst_layer, name)
            dst_mod.weight.data.copy_(src_mod.weight.data)

        # Copia parâmetros não quantizados
        dst_layer.log_precision.data.copy_(src_layer.log_precision.data)
        dst_layer.threshold.copy_(src_layer.threshold)
        dst_layer.norm.load_state_dict(src_layer.norm.state_dict())

    return bitnet_model


def create_rpt_bitnet_from_config(config: RPTConfig = None) -> RPTModelBitNet:
    """Cria modelo RPT BitNet com configuração."""
    if config is None:
        config = RPTConfig(
            vocab_size=49152,
            hidden_dim=576,
            num_layers=30,
            num_heads=9,
            intermediate_dim=1536,
            max_seq_len=2048,
            boundary_dim=72,
            num_regions=16,
            target_sparsity=0.03,
        )

    model = RPTModelBitNet(config)
    param_stats = count_ternary_params(model)

    print(f"Modelo RPT BitNet criado:")
    print(f"  Parâmetros ternários: {param_stats['ternary_params']:,}")
    print(f"  Parâmetros fp: {param_stats['full_precision_params']:,}")
    print(f"  Memória ternária: {param_stats['ternary_memory_mb']:.1f} MB")
    print(f"  Memória fp: {param_stats['full_memory_mb']:.1f} MB")

    return model


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_rpt_from_smollm2_config() -> RPTModel:
    """Cria modelo RPT com configuração baseada no SmolLM2-135M."""
    config = RPTConfig(
        vocab_size=49152,
        hidden_dim=576,
        num_layers=30,
        num_heads=9,
        intermediate_dim=1536,
        max_seq_len=2048,
        boundary_dim=72,
        num_regions=16,
        target_sparsity=0.03,
        free_phase_steps=10,
        nudged_phase_steps=5,
        state_lr=0.1,
        nudge_factor=0.2,
        criticality_gamma=0.01,
        initial_threshold=0.1,
        sparsity_lambda=0.001,
        criticality_lambda=0.01,
        precision_init=1.0
    )
    
    model = RPTModel(config)
    print(f"Modelo RPT criado com {count_parameters(model):,} parâmetros")
    return model


if __name__ == "__main__":
    # Teste básico
    print("=" * 60)
    print("Teste de Redes Preditivas Termodinâmicas (RPT)")
    print("=" * 60)
    
    # Cria modelo
    model = create_rpt_from_smollm2_config()
    
    # Dados de teste
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 49152, (batch_size, seq_len))
    labels = torch.randint(0, 49152, (batch_size, seq_len))
    
    # Forward pass
    print("\nExecutando forward pass...")
    outputs = model(input_ids, labels=labels)
    
    print(f"\nResultados:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss total: {outputs['loss'].item():.4f}")
    print(f"  CE Loss: {outputs['ce_loss'].item():.4f}")
    print(f"  Free Energy Loss: {outputs['free_energy_loss']:.4f}")
    print(f"  Branching Ratio: {model.branching_ratio.item():.4f}")
    
    # Métricas da fase livre
    print(f"\nMétricas da Fase Livre:")
    print(f"  Energia final: {outputs['free_metrics']['energy_history'][-1]:.4f}")
    print(f"  Esparsidade média: {sum(outputs['free_metrics']['sparsity_history'])/len(outputs['free_metrics']['sparsity_history']):.4f}")
    
    # Teste de geração
    print("\nTestando geração...")
    prompt = torch.randint(0, 49152, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"  Tokens gerados: {generated.shape[1] - 10}")
    
    print("\n" + "=" * 60)
    print("Teste concluído com sucesso!")
    print("=" * 60)
