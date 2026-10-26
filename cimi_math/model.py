from dataclasses import dataclass

import torch
from einops import einsum, rearrange, reduce
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 49152
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    d_ffn: int = 2816
    expert_d_ffn: int = 1408
    num_experts: int = 4
    expert_top_k: int = 2
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self):
        for name in (
            "vocab_size", "d_model", "num_layers", "num_heads",
            "d_ffn", "expert_d_ffn", "num_experts",
            "expert_top_k", "max_seq_len",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.expert_top_k > self.num_experts:
            raise ValueError("expert_top_k must not exceed num_experts")
        if min(self.rope_theta, self.norm_eps, self.initializer_range) <= 0:
            raise ValueError("rope_theta, norm_eps and initializer_range must be positive")


@dataclass
class ModelOutput:
    logits: Tensor
    aux_loss: Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean_square = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = einsum(positions, inv_freq, "seq, dim -> seq dim")
        self.register_buffer("cos", rearrange(angles.cos(), "seq dim -> 1 1 seq dim"), persistent=False)
        self.register_buffer("sin", rearrange(angles.sin(), "seq dim -> 1 1 seq dim"), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    even, odd = rearrange(x, "... (dim pair) -> pair ... dim", pair=2)
    rotated = [even * cos - odd * sin, even * sin + odd * cos]
    return rearrange(rotated, "pair ... dim -> ... (dim pair)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.qkv_proj(x), "batch seq (qkv heads dim) -> qkv batch heads seq dim",
            qkv=3, heads=self.num_heads,
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = rearrange(attended, "batch heads seq dim -> batch seq (heads dim)")
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.expert_top_k
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.expert_d_ffn)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = rearrange(x, "batch seq dim -> (batch seq) dim")
        probabilities = F.softmax(self.router(tokens), dim=-1, dtype=torch.float32)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(tokens.dtype)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_indices, slots = torch.where(indices == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(tokens.index_select(0, token_indices))
            weighted = einsum(expert_output, weights[token_indices, slots], "tokens dim, tokens -> tokens dim")
            output.index_add_(0, token_indices, weighted.to(output.dtype))

        # Normalize over all top-k assignments so balanced routing has loss 1.
        fractions = reduce(
            F.one_hot(indices, self.num_experts).float(), "tokens selected experts -> experts", "mean",
        )
        importance = reduce(probabilities, "tokens experts -> experts", "mean")
        aux_loss = self.num_experts * (fractions * importance).sum(dim=-1)
        return rearrange(output, "(batch seq) dim -> batch seq dim", batch=x.size(0)), aux_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = (
            SparseMoE(config) if use_moe
            else SwiGLU(config.d_model, config.d_ffn)
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        x = x + self.attention(self.attn_norm(x), cos, sin)
        normalized = self.ffn_norm(x)
        if isinstance(self.ffn, SparseMoE):
            update, aux_loss = self.ffn(normalized)
        else:
            update = self.ffn(normalized)
            aux_loss = x.new_zeros((), dtype=torch.float32)
        return x + update, aux_loss


class CimiModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.gradient_checkpointing = False
        config = self.config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            config.d_model // config.num_heads, config.max_seq_len, config.rope_theta,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_moe=(layer_id % 2 == 1))
            for layer_id in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: Tensor) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.size(0) == 0:
            raise ValueError("input_ids must have shape (batch, sequence) with a nonempty batch")
        seq_len = input_ids.size(1)
        if not 1 <= seq_len <= self.config.max_seq_len:
            raise ValueError(f"sequence length must be between 1 and {self.config.max_seq_len}")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(seq_len)
        aux_loss = x.new_zeros((), dtype=torch.float32)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x, layer_aux_loss = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x, layer_aux_loss = layer(x, cos, sin)
            aux_loss = aux_loss + layer_aux_loss
        aux_loss = aux_loss / max(1, self.config.num_layers // 2)
        return ModelOutput(logits=self.lm_head(self.norm(x)), aux_loss=aux_loss)
