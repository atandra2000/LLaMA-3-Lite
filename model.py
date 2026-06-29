import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class RoPE(nn.Module):
    """Rotary Position Embeddings with precomputed cos/sin buffers."""
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 500000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE and Flash Attention 2."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.rope = RoPE(head_dim, max_seq_len, rope_theta)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, self.n_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, self.n_heads, S, self.head_dim)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with fused gate+up projection."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.d_ff = d_ff

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, d_ff: int, max_seq_len: int, rope_theta: float):
        super().__init__()
        self.attention = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, head_dim, max_seq_len, rope_theta)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.attention_norm = RMSNorm(d_model, eps=1e-5)
        self.ffn_norm = RMSNorm(d_model, eps=1e-5)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, n_kv_heads: int, head_dim: int, d_ff: int,
                 max_seq_len: int, rope_theta: float = 500000.0,
                 rms_norm_eps: float = 1e-5, gradient_checkpointing: bool = False):
        super().__init__()
        self.input_embedding = InputEmbedding(d_model, vocab_size)

        decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, head_dim,
                         d_ff, max_seq_len, rope_theta)
            for _ in range(n_layers)
        ])
        self.decoder = Decoder(decoder_layers, d_model, eps=rms_norm_eps)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.input_embedding(x)
        if self.gradient_checkpointing and self.training:
            for layer in self.decoder.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = self.decoder(x)
        logits = self.output_proj(x)
        return logits

    def get_num_params(self, non_embedding=True):
        """Return the parameter count (subtracts input embedding and output proj when non_embedding=True)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.input_embedding.embedding.weight.numel()
            n_params -= self.output_proj.weight.numel()
        return n_params

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False


def chunked_cross_entropy(logits, targets, chunk_size=65536, ignore_index=-100):
    """Memory-efficient cross-entropy processing logits in chunks."""
    total_loss = torch.tensor(0.0, device=logits.device)
    total_count = torch.tensor(0, device=logits.device, dtype=torch.long)

    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=ignore_index, reduction='none')
        mask = chunk_targets != ignore_index
        total_loss = total_loss + chunk_loss[mask].sum()
        total_count = total_count + mask.sum()

    if total_count > 0:
        return total_loss / total_count.float()
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def build_transformer(
    vocab_size: int = 128256,
    d_model: int = 1024,
    n_layers: int = 16,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    head_dim: int = 128,
    d_ff: int = 4096,
    max_seq_len: int = 2048,
    rope_theta: float = 500000.0,
    rms_norm_eps: float = 1e-5,
    gradient_checkpointing: bool = False,
) -> Transformer:
    """Build LLaMA 3 model with specified architecture."""
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        d_ff=d_ff, max_seq_len=max_seq_len, rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        gradient_checkpointing=gradient_checkpointing,
    )
    num_params = sum(p.numel() for p in model.parameters())
    non_embed = num_params - model.input_embedding.embedding.weight.numel() - model.output_proj.weight.numel()
    print(f"Total params: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Non-embedding params: {non_embed:,} ({non_embed/1e6:.1f}M)")
    if gradient_checkpointing:
        print(f"Gradient checkpointing: ENABLED")
    return model