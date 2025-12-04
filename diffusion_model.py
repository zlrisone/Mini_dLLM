import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionModelConfig:
    """
    一个参考 Qwen-2.5-Coder 思路的轻量配置，参数量约 ~0.5B。
    你可以根据需要微调以下几个关键参数：
      - vocab_size
      - hidden_size
      - num_hidden_layers
      - intermediate_size
      - max_position_embeddings
    """

    vocab_size: int = 6400
    hidden_size: int = 768            # 模型维度 d_model
    num_hidden_layers: int = 16       # Transformer 层数
    num_attention_heads: int = 8
    num_key_value_heads: int = 2      # GQA：k/v 头数
    intermediate_size: int = 5504     # FFN 扩展维度（非 4x 方便卡参数）
    max_position_embeddings: int = 4096
    layer_norm_epsilon: float = 1e-6
    rotary_pct: float = 1.0
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_parallel_residual: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        return x * self.weight / (norm / math.sqrt(x.shape[-1]) + self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [b, n_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]（通过广播作用在 seq_len 维度）
    q_cos = q * cos
    q_sin = rotate_half(q) * sin
    k_cos = k * cos
    k_sin = rotate_half(k) * sin
    return q_cos + q_sin, k_cos + k_sin


class RotaryEmbedding(nn.Module):
    """
    简化版 RoPE，与 Qwen/LLama 类似。
    """

    def __init__(self, dim: int, theta: float = 10000.0, max_position_embeddings: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        # 原始缓存形状是 [1, max_seq, 1, dim]，这里变换成 [1, 1, seq_len, dim]
        cos = self.cos_cached[:, :seq_len, :, :].to(x.device).permute(0, 2, 1, 3)
        sin = self.sin_cached[:, :seq_len, :, :].to(x.device).permute(0, 2, 1, 3)
        return cos, sin


class MultiHeadAttention(nn.Module):
    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert (
            self.hidden_size % self.num_heads == 0
        ), "hidden_size 必须能被 num_attention_heads 整除"

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        rotary_dim = int(self.head_dim * config.rotary_pct)
        self.rotary_dim = rotary_dim
        self.rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int, n_heads: int):
        return x.view(bsz, seq_len, n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self._shape(q, seq_len, bsz, self.num_heads)
        k = self._shape(k, seq_len, bsz, self.num_kv_heads)
        v = self._shape(v, seq_len, bsz, self.num_kv_heads)

        # 应用 RoPE 到前 rotary_dim 维度
        cos, sin = self.rotary_emb(q, seq_len)  # [1, seq_len, 1, dim]

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # GQA: 扩展 kv 头到全部 q 头
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [b, h, s, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class FeedForward(nn.Module):
    """
    近似 Qwen 的 SwiGLU FFN 结构：W1(x) * swish(W2(x)) 投到 W3。
    """

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.w3(F.silu(x2) * x1)


class TransformerBlock(nn.Module):
    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.use_parallel_residual = config.use_parallel_residual
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_parallel_residual:
            normed = self.input_layernorm(hidden_states)
            attn_out = self.attn(normed, attention_mask)
            ffn_out = self.ffn(normed)
            hidden_states = hidden_states + self.dropout(attn_out) + self.dropout(ffn_out)
        else:
            attn_input = self.input_layernorm(hidden_states)
            attn_out = self.attn(attn_input, attention_mask)
            hidden_states = hidden_states + self.dropout(attn_out)

            ffn_input = self.post_attn_layernorm(hidden_states)
            ffn_out = self.ffn(ffn_input)
            hidden_states = hidden_states + self.dropout(ffn_out)

        return hidden_states


class DiffusionModel(nn.Module):
    """
    参考 Qwen-2.5-Coder 的 decoder-only Transformer，
    但缩小为 ~0.5B 参数、命名为 `DiffusionModel`。
    """

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重绑定：与 Qwen/LLaMA 风格一致
        self.lm_head.weight = self.embed_tokens.weight

    def _build_causal_mask(
        self, bsz: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # [1, 1, s, s] 下三角，未来位置为 -inf
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [batch, seq]
        attention_mask: [batch, seq]，可选，1 表示保留，0 表示 padding。
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)

        causal_mask = self._build_causal_mask(
            bsz, seq_len, device=device, dtype=hidden_states.dtype
        )

        if attention_mask is not None:
            # [b, 1, 1, s]
            attn = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * -1e9
            causal_mask = causal_mask + attn

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 简单测试：构建模型并打印参数量
    cfg = DiffusionModelConfig()
    model = DiffusionModel(cfg)
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # 造一个假的输入跑一遍前向
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(x)
    print("logits shape:", logits.shape)

