"""LLaMA-3-Lite decoder: RoPE, GQA, RMSNorm, SwiGLU, optional Triton paths.

Pure PyTorch by default; ``*_impl='triton'`` swaps in fused kernels.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from kernels.cross_entropy_triton import triton_chunked_cross_entropy_with_z
from kernels.rmsnorm_triton import triton_rmsnorm
from kernels.swiglu_triton import triton_swiglu


class RoPE(nn.Module):
    """Apply rotary position embeddings from cached frequencies."""
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
    """RMSNorm with optional Triton fused-path opt-in.

    The ``triton`` path is reached only when ``train.py`` has already gated on
    ``ENABLE_TRITON_KERNELS=1``; failures surface as the kernel's own
    ImportError/ValueError (no silent fallback — see AGENTS.md hard rule 7).
    """
    def __init__(self, d_model: int, eps: float = 1e-5, impl: str = "pytorch"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.impl = impl

    def forward(self, x):
        if self.impl == "triton":
            return triton_rmsnorm(x, self.weight, self.eps)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class GroupedQueryAttention(nn.Module):
    """Causal GQA with RoPE, optional QK normalization, and SDPA."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, max_seq_len: int, rope_theta: float,
                 qknorm: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        # Normalize each projected head before RoPE to bound logit growth.
        if qknorm:
            self.q_norm = RMSNorm(head_dim, eps=1e-5)
            self.k_norm = RMSNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rope = RoPE(head_dim, max_seq_len, rope_theta)

    def forward(self, x):
        B, S, _ = x.shape

        # Normalize while head_dim is the last axis; transpose afterward.
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, S)
        k = self.rope(k, S)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, self.n_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, S, self.head_dim).reshape(B, self.n_heads, S, self.head_dim)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block with a fused gate/up projection.

    Triton is explicit opt-in; kernel failures are intentionally propagated.
    """
    def __init__(self, d_model: int, d_ff: int, swiglu_impl: str = "pytorch"):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.d_ff = d_ff
        self.swiglu_impl = swiglu_impl

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        if self.swiglu_impl == "triton":
            return self.down_proj(triton_swiglu(gate_up, self.d_ff))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderBlock(nn.Module):
    """Pre-norm attention and SwiGLU residual block."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, d_ff: int, max_seq_len: int, rope_theta: float,
                 qknorm: bool = True, rmsnorm_impl: str = "pytorch",
                 swiglu_impl: str = "pytorch"):
        super().__init__()
        self.attention = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, head_dim, max_seq_len, rope_theta,
            qknorm=qknorm)
        self.ffn = SwiGLUFFN(d_model, d_ff, swiglu_impl=swiglu_impl)
        self.attention_norm = RMSNorm(d_model, eps=1e-5, impl=rmsnorm_impl)
        self.ffn_norm = RMSNorm(d_model, eps=1e-5, impl=rmsnorm_impl)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Decoder(nn.Module):
    """Stack decoder blocks and apply the final RMSNorm."""

    def __init__(self, layers: nn.ModuleList, d_model: int, eps: float = 1e-5,
                 rmsnorm_impl: str = "pytorch"):
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(d_model, eps=eps, impl=rmsnorm_impl)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Transformer(nn.Module):
    """Decoder-only language model with a separate output projection."""

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, n_kv_heads: int, head_dim: int, d_ff: int,
                 max_seq_len: int, rope_theta: float = 500000.0,
                 rms_norm_eps: float = 1e-5, gradient_checkpointing: bool = False,
                 qknorm: bool = True, rmsnorm_impl: str = "pytorch",
                 swiglu_impl: str = "pytorch"):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)

        decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, head_dim,
                         d_ff, max_seq_len, rope_theta, qknorm=qknorm,
                         rmsnorm_impl=rmsnorm_impl, swiglu_impl=swiglu_impl)
            for _ in range(n_layers)
        ])
        self.decoder = Decoder(decoder_layers, d_model, eps=rms_norm_eps,
                                rmsnorm_impl=rmsnorm_impl)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_hidden: bool = False):
        x = self.input_embedding(x)
        if self.gradient_checkpointing and self.training:
            for layer in self.decoder.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            x = self.decoder.norm(x)
        else:
            x = self.decoder(x)
        if return_hidden:
            return x
        logits = self.output_proj(x)
        return logits

    def get_num_params(self, non_embedding=True):
        """Parameter count; subtracts embeddings when ``non_embedding=True``."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.input_embedding.weight.numel()
            n_params -= self.output_proj.weight.numel()
        return n_params



def chunked_cross_entropy_with_z(
    logits, targets, chunk_size=256, ignore_index=-100, z_loss_weight=1e-4,
    cross_entropy_impl: str = "pytorch",
):
    """CE + z-loss (PaLM / Gemma2) over an already-materialized logits tensor.

    Chunks along the token axis so the FP32 loss chain never sees more than
    ``chunk_size`` rows at once; z-loss is averaged over non-ignored tokens
    only. ``cross_entropy_impl='triton'`` swaps the per-chunk FP32 chain for
    the fused kernel; the kernel raises ImportError/ValueError if triton is
    unavailable or the vocab is too large, so opt-in is explicit (no silent
    fallback — see AGENTS.md hard rule 7). Prefer
    :func:`chunked_head_cross_entropy_with_z` when the logits tensor itself
    would not fit in memory — this function still receives full logits.
    """
    if cross_entropy_impl == "triton":
        return triton_chunked_cross_entropy_with_z(
            logits, targets,
            ignore_index=ignore_index,
            z_loss_weight=z_loss_weight,
        )

    total_ce = torch.tensor(0.0, device=logits.device)
    total_count = torch.tensor(0, device=logits.device, dtype=torch.long)
    z_accum = torch.tensor(0.0, device=logits.device)
    n_z = 0

    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        # One promotion keeps logsumexp and CE on the same FP32 values.
        cl = logits[start:end].float()
        ct = targets[start:end]

        mask = ct != ignore_index
        log_z = torch.logsumexp(cl, dim=-1)
        z_accum = z_accum + log_z[mask].pow(2).sum()
        n_z += mask.sum()

        ce = F.cross_entropy(cl, ct, ignore_index=ignore_index, reduction='none')
        total_ce = total_ce + ce[mask].sum()
        total_count = total_count + mask.sum()

    ce_loss = (total_ce / total_count.float()) if total_count > 0 else torch.tensor(0.0, device=logits.device)
    z_loss = z_accum / max(int(n_z), 1)
    return ce_loss + z_loss_weight * z_loss


def chunked_head_cross_entropy_with_z(
    hidden, head_weight, targets, chunk_size=256, ignore_index=-100,
    z_loss_weight=1e-4, cross_entropy_impl: str = "pytorch",
):
    """Memory-bounded LM head + CE + z-loss: never materializes full logits.

    Computes ``hidden @ head_weight.T`` in ``chunk_size``-row slices and
    applies the FP32 CE + z-loss chain per chunk. Each chunk runs inside
    ``checkpoint`` so only one chunk's logits are alive at a time — this is
    what bounds the loss memory to ~0.3 GB at ``chunk_size=256`` instead of
    the ~50 GB a full ``[N, V]`` logits tensor would need. Gradients flow to
    both ``hidden`` and ``head_weight``. ``cross_entropy_impl='triton'`` swaps
    the per-chunk loss for the fused kernel; per-chunk losses are then
    averaged (equal-size chunks ⇒ exact). When ``cross_entropy_impl='triton'``
    but triton is unavailable, a clear ImportError is raised at function
    entry (no silent fallback — see AGENTS.md hard rule 7).
    """
    from kernels.cross_entropy_triton import HAS_TRITON

    if cross_entropy_impl == "triton" and not HAS_TRITON:
        raise ImportError(
            "cross_entropy_impl='triton' requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only) and set "
            "ENABLE_TRITON_KERNELS=1, or use cross_entropy_impl='pytorch'."
        )
    use_triton = cross_entropy_impl == "triton"

    def _chunk(hidden_c, w, targets_c):
        logits = F.linear(hidden_c, w)
        if use_triton:
            return triton_chunked_cross_entropy_with_z(
                logits, targets_c, ignore_index=ignore_index,
                z_loss_weight=z_loss_weight,
            )
        cl = logits.float()
        log_z = torch.logsumexp(cl, dim=-1)
        ce = F.cross_entropy(cl, targets_c, ignore_index=ignore_index,
                             reduction='none')
        mask = targets_c != ignore_index
        return ce[mask].sum(), mask.sum().float(), log_z[mask].pow(2).sum()

    total_ce = torch.tensor(0.0, device=hidden.device)
    total_count = torch.tensor(0, device=hidden.device, dtype=torch.long)
    z_accum = torch.tensor(0.0, device=hidden.device)
    n_z = 0
    n_chunks = 0
    triton_acc = torch.tensor(0.0, device=hidden.device)

    for start in range(0, hidden.shape[0], chunk_size):
        end = min(start + chunk_size, hidden.shape[0])
        out = checkpoint(_chunk, hidden[start:end], head_weight,
                         targets[start:end], use_reentrant=False)
        n_chunks += 1
        if use_triton:
            triton_acc = triton_acc + out
            continue
        total_ce = total_ce + out[0]
        total_count = total_count + out[1].long()
        z_accum = z_accum + out[2]
        n_z += int(out[1])

    if use_triton:
        return triton_acc / max(n_chunks, 1)

    ce_loss = (total_ce / total_count.float()) if total_count > 0 else torch.tensor(0.0, device=hidden.device)
    z_loss = z_accum / max(n_z, 1)
    return ce_loss + z_loss_weight * z_loss


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
    qknorm: bool = True,
    rmsnorm_impl: str = "pytorch",
    swiglu_impl: str = "pytorch",
) -> Transformer:
    """Construct a LLaMA-style decoder from explicit architecture settings."""
    model = Transformer(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        d_ff=d_ff, max_seq_len=max_seq_len, rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        gradient_checkpointing=gradient_checkpointing,
        qknorm=qknorm,
        rmsnorm_impl=rmsnorm_impl,
        swiglu_impl=swiglu_impl,
    )
    num_params = sum(p.numel() for p in model.parameters())
    non_embed = model.get_num_params(non_embedding=True)
    print(f"Total params: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Non-embedding params: {non_embed:,} ({non_embed/1e6:.1f}M)")
    if gradient_checkpointing:
        print("Gradient checkpointing: ENABLED")
    if rmsnorm_impl == "triton" or swiglu_impl == "triton":
        active = [k for k, v in (("rmsnorm", rmsnorm_impl), ("swiglu", swiglu_impl)) if v == "triton"]
        print(f"Triton kernels active: {', '.join(active)}")
    return model
