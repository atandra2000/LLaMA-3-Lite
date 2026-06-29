# Training Script Walkthrough — `train.py`

> **File**: [`train.py`](../train.py)  
> **Entry point**: `python train.py`  
> **Purpose**: End-to-end pretraining orchestration for the LLaMA-3-Lite 515M model on a single NVIDIA A100 80GB SXM GPU.

This document is a line-by-line conceptual walkthrough of the training script. It explains *what* every piece of code does, *why* each decision was made, and *what tradeoffs* are involved. It is written to be useful both as a learning resource and as a reference when modifying the training pipeline.

---

## Table of Contents

1. [Module-Level Setup](#1-module-level-setup)
2. [GPU Optimization Setup — `setup_gpu_optimizations`](#2-gpu-optimization-setup)
3. [Learning Rate Scheduler — `CosineWithWarmup`](#3-learning-rate-scheduler)
4. [Sampling — `top_k_top_p_sampling`](#4-sampling)
5. [Text Generation — `generate_samples`](#5-text-generation)
6. [Validation Loop — `validate`](#6-validation-loop)
7. [Checkpointing — `save_checkpoint` & `load_checkpoint`](#7-checkpointing)
8. [Main Training Function — `train_model`](#8-main-training-function)
   - [8.1 Device and GPU Setup](#81-device-and-gpu-setup)
   - [8.2 Model Construction](#82-model-construction)
   - [8.3 torch.compile()](#83-torchcompile)
   - [8.4 Optimizer — AdamW with Selective Weight Decay](#84-optimizer)
   - [8.5 LR Scheduler and GradScaler](#85-lr-scheduler-and-gradscaler)
   - [8.6 Checkpoint Resume](#86-checkpoint-resume)
   - [8.7 W&B Initialization](#87-wb-initialization)
   - [8.8 Signal Handlers — Emergency Save](#88-signal-handlers)
   - [8.9 CUDA Streams for Async Prefetch](#89-cuda-streams-for-async-prefetch)
   - [8.10 The Training Loop](#810-the-training-loop)
   - [8.11 Gradient Accumulation](#811-gradient-accumulation)
   - [8.12 Logging](#812-logging)
   - [8.13 Periodic Actions — Val, Generation, Checkpointing](#813-periodic-actions)
   - [8.14 Post-Training Teardown](#814-post-training-teardown)
9. [End-to-End Data Flow Diagram](#9-end-to-end-data-flow-diagram)
10. [Key Design Decisions Summary](#10-key-design-decisions-summary)

---

## 1. Module-Level Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
import wandb
import signal, threading, math, time, random, numpy

from dataset import build_training_data
from model import build_transformer, chunked_cross_entropy
from config import get_config, cleanup_old_checkpoints

os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

**`TOKENIZERS_PARALLELISM = "false"`** — HuggingFace tokenizers spawn a Rust-backed thread pool. When combined with PyTorch's multiprocessing DataLoader workers, this causes a deadlock on fork. Disabling parallelism in the tokenizer is the standard fix; the DataLoader workers themselves handle the parallelism at a higher level.

**Import separation** — `dataset`, `model`, and `config` are kept as separate modules. This means `train.py` can be tested or replaced independently, and smoke tests (`test_pipeline.py`) can inject synthetic data without touching the training script.

---

## 2. GPU Optimization Setup

```python
def setup_gpu_optimizations(config):
```

This function is called once at startup if a CUDA device is available. It configures several A100-specific settings before any tensors are allocated.

### TF32 Tensor Cores

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

**What is TF32?** TensorFloat-32 is a format unique to Ampere GPUs (A100, RTX 30xx). It uses 10 bits of mantissa for FP32 matmuls instead of 23, which is enough for neural network precision but enables the hardware's tensor cores to run at full speed — roughly **3× faster** than full FP32 matmuls.

`allow_tf32` enables this for both `torch.matmul` and cuDNN operations (convolutions). `set_float32_matmul_precision('high')` is the newer unified API that does the same. Both are set for belt-and-suspenders compatibility with different PyTorch versions.

**Why it's safe here** — The model trains in BF16 mixed precision (see §8.5). TF32 applies to any remaining FP32 matmuls (e.g., optimizer state updates). The precision loss is negligible for SGD-style dynamics.

### cuDNN Benchmark Mode

```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

When `benchmark=True`, cuDNN profiles several convolution algorithms on the first batch and picks the fastest one. For transformers (which are mostly matmuls), the benefit is small but nonzero. `deterministic=False` is required to enable `benchmark`, and is acceptable since training is not expected to be bit-exact across runs anyway.

### CUDA Allocator Configuration

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

PyTorch's CUDA memory allocator, by default, holds on to large blocks once allocated and reuses them. `expandable_segments` allows the allocator to grow segments dynamically rather than pre-allocating large contiguous blocks. This reduces fragmentation during the early training steps when memory usage climbs to its peak, making OOM errors from fragmentation less likely.

---

## 3. Learning Rate Scheduler

```python
class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr, peak_lr):
```

The scheduler is implemented from scratch rather than using `torch.optim.lr_scheduler` for two reasons:

1. **Resumability** — `torch.optim.lr_scheduler` schedulers tie their state to optimizer step counts in a way that makes resuming from mid-run checkpoints fragile. A hand-rolled scheduler stores a single `_step` counter, which serializes cleanly into the checkpoint dict.
2. **Transparency** — The LR formula is visible in the code, not buried inside a library's `get_last_lr()` abstraction.

### Warmup Phase

```python
def get_lr(self):
    if self._step < self.warmup_steps:
        return self.peak_lr * self._step / self.warmup_steps
```

During the first `warmup_steps` (2,000 steps by default), the learning rate increases **linearly** from 0 to `peak_lr` (3e-4).

**Why warm up?** At the start of training, the model's weights are randomly initialized and the gradients are large and noisy. A full learning rate would cause the optimizer to take massive steps, potentially destabilizing the loss. Warming up gradually lets the optimizer find a reasonable direction before committing to large updates. This is especially important for large-batch training where gradient noise is lower but step magnitudes are high.

### Cosine Decay Phase

```python
    else:
        progress = (self._step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine
```

After warmup, the LR follows a **cosine annealing** curve from `peak_lr` down to `min_lr` (3e-5 — exactly 10× below peak).

**Why cosine over linear decay?** Cosine decay spends more time near the peak LR (fast exploration) and more time near the minimum LR (careful refinement at the end). The smooth S-curve also avoids the abrupt "cliff" that linear decay produces when the rate drops sharply.

**Why not decay to 0?** The `min_lr` floor of `3e-5` prevents the optimizer from taking micro-steps near the end of training. Very small learning rates stall progress without meaningfully improving the loss, and completely zeroing the LR can cause gradient accumulators (Adam's `m` and `v`) to stagnate.

### The `step()` Method

```python
def step(self):
    self._step += 1
    lr = self.get_lr()
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
```

The scheduler directly mutates `optimizer.param_groups[*]['lr']` rather than using `optimizer.step()` hooks. This is the same pattern used by PyTorch schedulers internally. Notice it updates **all** param groups — this means both the `decay_params` and `no_decay_params` groups (see §8.4) track the same LR, which is correct: weight decay is the only difference between them.

---

## 4. Sampling

```python
def top_k_top_p_sampling(logits, top_k, top_p, temperature):
```

This function implements combined **temperature + top-k + nucleus (top-p) sampling** for autoregressive text generation. It operates on a single-token logit vector `(vocab_size,)`.

### Temperature Scaling

```python
logits = logits / temperature
```

Divides all logits by a scalar before softmax. A temperature of `1.0` leaves the distribution unchanged. Values `< 1.0` make the distribution sharper (more confident, repetitive). Values `> 1.0` flatten it (more random, creative). The default is `0.8` — slightly conservative to avoid incoherent output during early training.

**Mathematical note**: Dividing logits by `T` before softmax is equivalent to raising probabilities to the power `1/T`. This is why high temperature "spreads" the distribution.

### Top-k Filtering

```python
if top_k > 0:
    top_k_vals, top_k_indices = logits.topk(top_k, dim=-1)
    logits = torch.full_like(logits, float('-inf')).scatter_(-1, top_k_indices, top_k_vals)
```

Zeros out (sends to `-inf`) all logits except the `top_k` highest. After softmax, this means the model can only sample from the 50 most probable tokens. This eliminates tokens with negligible probability that would otherwise contribute noise.

### Top-p (Nucleus) Filtering

```python
if top_p > 0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
```

Top-p sampling keeps the smallest set of tokens whose cumulative probability exceeds `top_p` (0.9 by default). Unlike top-k which always keeps exactly `k` tokens, top-p adapts to the shape of the distribution: when the model is confident, very few tokens are kept; when it's uncertain, many are included.

The `[..., 1:] = [..., :-1]` shift is a subtle but critical detail: the cumulative sum exceeds the threshold *after* adding a token, so we shift the mask one position right to include the token that crossed the threshold rather than excluding it.

**Why combine top-k and top-p?** Top-k ensures we never sample from a very long tail of low-probability tokens. Top-p then further restricts based on the actual probability distribution. Together, they cover both the case where the distribution is flat (top-k caps the candidates) and where it's peaked (top-p tightly focuses on the high-probability region).

---

## 5. Text Generation

```python
@torch.no_grad()
def generate_samples(model, tokenizer, device, step, config):
```

The `@torch.no_grad()` decorator disables PyTorch's autograd engine for the entire function, which avoids allocating gradient computation graphs for inference — a significant memory saving.

### Autoregressive Decoding

```python
for _ in range(config['generation_max_tokens']):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, ...):
        logits = model(generated)
    next_token = top_k_top_p_sampling(logits[:, -1, :], ...)
    generated = torch.cat([generated, next_token], dim=1)
    if next_token.item() == tokenizer.eos_token_id:
        break
```

Each iteration runs a full forward pass on the entire accumulated sequence. Only the **last token's logit** (`logits[:, -1, :]`) is used for sampling — the others were needed by the causal attention mechanism to compute the last position, but their logits are discarded.

**Efficiency note**: This is the naive autoregressive approach (re-computing all previous positions on each step). A production inference system would use a **KV cache** to store past key/value projections and only compute the new token's attention. For diagnostic generation (128 tokens, 5 prompts, every 20K training steps), the overhead is acceptable.

The `model.eval()` / `model.train()` pair around generation is critical: `eval()` disables gradient checkpointing's recomputation on the forward pass (since there's no backward), and it would also disable dropout if it were used.

### W&B Logging of Generated Text

```python
table = wandb.Table(columns=["prompt", "generated", "step"])
# ... populate ...
wandb.log({"gen/samples": table}, step=step)
```

Generated samples are logged as a W&B `Table`, which allows browsing them in the W&B UI across steps. This is the primary human-readable signal for whether the model is learning language structure during training.

---

## 6. Validation Loop

```python
@torch.no_grad()
def validate(model, val_dataloader, pad_id, device, step, config):
```

### Why Validate During Pretraining?

Validation loss estimates generalization and catches overfitting or training instability. In pretraining, overfitting is rare (the dataset is enormous relative to model capacity), but validation loss is still valuable as a regression check — a validation loss spike indicates NaN gradients, a bad checkpoint, or a data pipeline bug.

### Chunked Cross-Entropy in Validation

```python
if use_chunked_ce:
    loss = chunked_cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        chunk_size=65536,
        ignore_index=pad_id,
    )
```

The same chunked cross-entropy used in training (see §8.10) is applied during validation, even though `@torch.no_grad()` means there's no gradient tape. This is intentional: without chunking, a single validation batch with `batch_size=96, seq_len=2048` produces a logits tensor of shape `(96*2048, 128000)` ≈ **50 GB**. Chunking reduces this to ~0.3 GB.

### Perplexity

```python
perplexity = math.exp(min(avg_loss, 20))
```

Perplexity is defined as `exp(cross_entropy_loss)`. The `min(avg_loss, 20)` guard prevents `math.exp` from overflowing on early training steps when the loss can be very high (e.g., a freshly initialized model has loss ≈ `log(vocab_size)` ≈ `log(128000)` ≈ 11.8).

Perplexity is more interpretable than raw loss: a perplexity of 50 means the model is as uncertain as if it were choosing uniformly among 50 equally likely tokens at each position.

---

## 7. Checkpointing

### `save_checkpoint`

```python
def save_checkpoint(model, optimizer, scheduler, step, config,
                    best_val_loss=None, is_final=False, async_save=True):
```

#### What Gets Saved

```python
checkpoint = {
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step':                 step,
    'tokens_seen':          step * batch_size * seq_len * grad_accum,
    'best_val_loss':        best_val_loss,
    'rng_torch':            torch.random.get_rng_state(),
    'rng_numpy':            numpy.random.get_state(),
    'rng_python':           random.getstate(),
    'rng_cuda':             torch.cuda.get_rng_state(),  # if CUDA
    'config':               config,
}
```

**Saving all RNG states** is the key enabler of "exact reproducibility." When training is resumed, every source of randomness (weight init RNG, data shuffling RNG, dropout RNG if used) is restored to the exact state it was in at the checkpoint step. This means the resumed run produces bit-identical outputs to a hypothetical uninterrupted run.

**Why save `config`?** The `config` dict is snapshotted inside the checkpoint so that, when loading a checkpoint weeks later, you can always reconstruct the exact hyperparameters used, even if `config.py` has since been modified.

#### Async Checkpoint I/O

```python
if async_save and config.get('async_checkpoint', True):
    thread = threading.Thread(
        target=_save_checkpoint_to_disk,
        args=(checkpoint_copy, path),
        daemon=True
    )
    thread.start()
    return thread
```

Saving a 515M-parameter model checkpoint to disk can take **several seconds** (the state dict is ~2 GB at FP32). If done synchronously, this stalls training. By offloading the `torch.save` call to a daemon thread, the training loop continues computing the next batch while the I/O happens concurrently.

**The `checkpoint_copy` shallow copy** ensures the background thread works on its own reference to the state dict. Without this, if the optimizer state is modified (e.g., `zero_grad()`) before the thread finishes writing, the saved checkpoint could be corrupted.

**`daemon=True`** ensures the thread does not prevent the Python process from exiting. If the main process is killed (e.g., by a scheduler), the daemon thread is also killed — the emergency signal handler (§8.8) handles graceful saves in that scenario.

#### Checkpoint Filename and Cleanup

```python
path = model_folder / f"{config['model_filename']}_step_{step}.pt"
```

Files are named with their step number. The `cleanup_old_checkpoints` function in `config.py` sorts by the integer step suffix (not lexicographically, to avoid `step_10.pt` < `step_9.pt`) and deletes all but the `keep_last_n_checkpoints` (default 3) most recent.

#### Final Checkpoint

```python
if is_final:
    torch.save(checkpoint, final_path)           # Full state
    torch.save(model.state_dict(), model_only_path)  # Weights only
```

At the end of training, two files are written: the full checkpoint (for exact reproducibility) and a weights-only file (for inference, fine-tuning, or sharing — without 3× overhead of optimizer states).

### `load_checkpoint`

```python
checkpoint = torch.load(latest, map_location=device, weights_only=False)
```

**`map_location=device`** ensures that tensors in the checkpoint are loaded directly onto the target device, without first staging on CPU. Without this, a checkpoint saved on GPU would load to GPU even when `device='cpu'`.

#### The RNG State Bug Fix

```python
rng_torch = checkpoint['rng_torch']
if isinstance(rng_torch, torch.Tensor):
    rng_torch = rng_torch.cpu().to(torch.uint8)
torch.random.set_rng_state(rng_torch)
```

**This is a subtle and important fix.** `torch.load(map_location=device)` moves *all* tensors in the checkpoint to the specified device — including the RNG state tensor, which `torch.random.set_rng_state()` expects to be a **CPU ByteTensor**. If the checkpoint was saved on GPU and is loaded onto GPU, the RNG state tensor would be on GPU and `set_rng_state()` would raise a cryptic error. The explicit `.cpu().to(torch.uint8)` ensures the RNG state is always in the correct form regardless of the load device.

The same fix is applied to `rng_cuda` — CUDA's `set_rng_state` expects a CPU ByteTensor and dispatches the state to the GPU device internally.

---

## 8. Main Training Function

```python
def train_model(config, train_dataloader=None, val_dataloader=None, tokenizer=None):
```

The optional `train_dataloader/val_dataloader/tokenizer` parameters allow the function to be called from tests with pre-built synthetic dataloaders, bypassing the HuggingFace download. When called from `__main__`, these are `None` and the function builds them itself.

### 8.1 Device and GPU Setup

```python
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
if device.type == 'cuda':
    setup_gpu_optimizations(config)
```

`torch.cuda.is_available()` is the standard guard. The script runs on CPU (for smoke tests, development on laptops) and on GPU (for actual training) without code changes. GPU-specific features like CUDA streams and mixed-precision autocast are gated on `device.type == 'cuda'`.

### 8.2 Model Construction

```python
real_vocab_size = max(config['vocab_size'], len(tokenizer))
model = build_transformer(
    vocab_size=real_vocab_size,
    ...
    gradient_checkpointing=gradient_checkpointing,
).to(device)
```

**`max(config['vocab_size'], len(tokenizer))`** — The config specifies `vocab_size=128000`, but the actual LLaMA 3 tokenizer has `128256` tokens (the extra 256 are reserved for control tokens and special use). Using `max()` ensures the output projection matrix is always large enough to cover all tokenizer IDs. Attempting to embed or project token ID 128001 into a matrix of size 128000 would cause a runtime index error.

#### Memory Estimation

```python
model_mem_gb = num_params * 2 / 1e9  # BF16 = 2 bytes/param
```

This prints an estimated peak GPU memory before training starts, helping the user anticipate OOM errors on smaller GPUs. The formula `num_params * 2` gives model weights in BF16 bytes. The full Adam memory (weights + gradients + m + v) is roughly `8 * num_params * 2` bytes at BF16/FP32 mixed, but the print intentionally shows just the weights as a lower bound.

### 8.3 torch.compile()

```python
if config.get('compile_model', True) and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)
```

`torch.compile()`, introduced in PyTorch 2.0, uses **TorchDynamo** to trace the model's computation graph and then passes it to **TorchInductor**, which generates optimized C++/CUDA kernels via Triton.

**What it does concretely:**
- **Operator fusion**: multiple element-wise ops (e.g., `silu(gate) * up`) are merged into a single kernel, eliminating intermediate tensor allocations and kernel launch overhead.
- **Memory layout optimization**: tensors are laid out contiguously in memory where beneficial.
- **Loop unrolling and vectorization**: for small, repeated ops.

**The compilation penalty**: The first 1–3 training steps are slow (tens of seconds) while TorchDynamo traces the model and TorchInductor compiles the kernels. Subsequent steps run the optimized kernels. For 42,000 training steps, the one-time compile cost is negligible.

**`hasattr(torch, 'compile')`** ensures backward compatibility with PyTorch 1.x where `torch.compile` does not exist.

### 8.4 Optimizer

#### Selective Weight Decay

```python
decay_params = []
no_decay_params = []
for param in model.named_parameters():
    if not param[1].requires_grad:
        continue
    if param[1].dim() >= 2:
        decay_params.append(param[1])
    else:
        no_decay_params.append(param[1])
```

**Why separate decay groups?**

Weight decay (`L2` regularization) encourages smaller weights, which helps prevent overfitting. However, it should **only** be applied to weight matrices (2D+ tensors), not to:
- **Biases** — 1D, and this model has no biases anyway (`bias=False` everywhere).
- **RMSNorm scale parameters** — 1D. Decaying these would collapse the normalization layer's learned scales toward zero, harming training.

The heuristic `dim() >= 2` correctly separates:
- **Decayed**: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `out_proj.weight`, `gate_up_proj.weight`, `down_proj.weight`, `output_proj.weight`, `input_embedding.weight` (all 2D).
- **Not decayed**: all `RMSNorm.weight` parameters (1D, shape `(d_model,)`).

```python
optimizer = torch.optim.AdamW([
    {'params': decay_params,    'weight_decay': config['weight_decay']},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=config['learning_rate'], betas=(0.9, 0.95), eps=1e-8)
```

#### AdamW vs Adam

Standard Adam adds the L2 penalty to the **gradient** before applying the adaptive scaling. This means the effective weight decay depends on the gradient magnitude — parameters with small gradients get less regularization. **AdamW** (Decoupled Weight Decay) applies the decay **directly to the weights** after the adaptive step, making weight decay independent of gradient history. For large language models, AdamW consistently outperforms Adam.

#### Beta Parameters

- **`beta1 = 0.9`** — Exponential decay for the first moment (gradient). A 10-step effective window.
- **`beta2 = 0.95`** — Exponential decay for the second moment (squared gradient). A 20-step effective window. LLaMA 3 uses 0.95 rather than the common 0.999 because a shorter window makes the adaptive scaling more responsive to recent gradient magnitudes, which is beneficial for large batch training where gradient statistics change rapidly.
- **`eps = 1e-8`** — Added to the denominator to prevent division by zero. Rarely matters in practice.

### 8.5 LR Scheduler and GradScaler

```python
scheduler = CosineWithWarmup(optimizer, ...)
scaler = GradScaler(enabled=(device.type == 'cuda'))
```

**`GradScaler`** manages the loss scaling required for numerically stable mixed-precision (BF16/FP16) training. When using half-precision, small gradients can underflow to zero (become 0.0 when their true value is too small to represent). Loss scaling multiplies the loss by a large scalar before backward, which shifts gradients into representable range, then divides the optimizer step by the same scalar.

**Why is BF16 more stable than FP16?** BF16 has the same exponent range as FP32 (8 bits vs FP16's 5 bits), so gradient underflow is very rare. The GradScaler is kept enabled for correctness on mixed systems but rarely needs to actually scale in BF16 training. Its overhead is negligible.

### 8.6 Checkpoint Resume

```python
initial_step, best_val_loss = 0, float('inf')
if config.get('preload') is not None:
    initial_step, best_val_loss = load_checkpoint(
        model, optimizer, scheduler, config, device)
```

When `preload` is set, `load_checkpoint` restores the full training state: model weights, optimizer momentum buffers, scheduler step counter, and all RNG states. The training loop then starts from `initial_step`, and the LR scheduler correctly continues from the resumed LR (because `_step` is restored).

The `best_val_loss` is also restored so that the "save best model" logic (§8.13) correctly tracks whether the resumed run improves on the pre-resume best.

### 8.7 W&B Initialization

```python
wandb.init(
    project=config['wandb_project'],
    name=f"llama3-515M-{device}-{int(time.time())}",
    config={ ... },
    tags=config.get('wandb_tags', []),
)
```

W&B is initialized **after** the model and optimizer are built but **before** the training loop starts. The full hyperparameter dict is passed as `config=`, which W&B stores and makes searchable. The run `name` includes a Unix timestamp so that multiple runs are distinguishable even if they share the same config.

**`wandb_entity=None`** defaults to the user's default entity. Set this to a team name if logging to a shared W&B team workspace.

### 8.8 Signal Handlers

```python
global_state = {'step': initial_step, 'model': model, 'optimizer': optimizer,
                'scheduler': scheduler, 'config': config, 'best_val_loss': best_val_loss}

def emergency_save_handler(signum, frame):
    save_checkpoint(global_state['model'], ...)
    wandb.finish()
    sys.exit(1)

signal.signal(signal.SIGTERM, emergency_save_handler)
signal.signal(signal.SIGINT, emergency_save_handler)
```

On HPC clusters and cloud preemptible instances, training jobs are killed with `SIGTERM` (or `SIGINT` from Ctrl+C) before their time allocation expires. Without a signal handler, the job exits immediately and all in-progress training since the last checkpoint is lost.

**`global_state` dict** acts as a closure over the current training state. The signal handler uses it to save the latest model/optimizer/scheduler state and cleanly finish the W&B run (uploading any pending logs). The step stored in `global_state['step']` is updated at the top of each training loop iteration, so at most one step's worth of computation is lost.

**`sys.exit(1)`** exits with a non-zero code, signaling to the job scheduler that the process was interrupted rather than completed successfully.

### 8.9 CUDA Streams for Async Prefetch

```python
if device.type == 'cuda':
    data_stream = torch.cuda.Stream()
```

A CUDA stream is an ordered sequence of GPU operations. PyTorch uses a **default stream** for all compute operations. By creating a separate `data_stream`, we can overlap **CPU→GPU data transfer** with **GPU compute** from the previous batch.

#### The Double-Buffer Pattern

Before the loop, one batch is pre-fetched:

```python
next_batch = next(step_iterator)
with torch.cuda.stream(data_stream):
    next_input = next_batch['input'].to(device, non_blocking=True)
    next_target = next_batch['target'].to(device, non_blocking=True)
torch.cuda.current_stream().wait_stream(data_stream)
```

Inside the loop:

```python
for step in pbar:
    # Use the batch that was pre-fetched last iteration
    input_ids = next_input
    target_ids = next_target

    # Fetch the NEXT batch from CPU while GPU runs the current batch
    batch = next(step_iterator)
    if device.type == 'cuda':
        with torch.cuda.stream(data_stream):
            next_input = batch['input'].to(device, non_blocking=True)
            next_target = batch['target'].to(device, non_blocking=True)

    # GPU compute on current batch (overlaps with DMA transfer above)
    logits = model(input_ids)
    ...

    # Synchronize: ensure the transfer is complete before next iteration
    torch.cuda.current_stream().wait_stream(data_stream)
```

**`non_blocking=True`** issues a DMA (Direct Memory Access) transfer from pinned CPU memory to GPU memory without blocking the CPU thread. **Pinned memory** (`pin_memory=True` in the DataLoader) is a prerequisite — page-locked host memory enables asynchronous DMA transfers.

**The timeline looks like:**

```
Step N:  [Transfer N+1 ──────] [Compute N ─────────────────]
Step N+1:                       [Transfer N+2 ──────] [Compute N+1 ──────────]
```

Without double-buffering, there would be a data-wait stall at the start of every step. This optimization is worth **5–15% throughput** depending on batch size and data loading speed. The `train/data_wait_ms` W&B metric (§8.12) measures any residual stall.

### 8.10 The Training Loop

```python
for step in pbar:
```

#### Forward Pass with Mixed Precision

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                    enabled=(device.type == 'cuda')):
    logits = model(input_ids)
    if use_chunked_ce:
        loss = chunked_cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            chunk_size=65536,
            ignore_index=pad_id,
        )
    else:
        loss = F.cross_entropy(...)
    loss = loss / grad_accum_steps
```

`torch.autocast` wraps the forward pass in a context that automatically promotes or demotes operations to the most efficient dtype:
- Matrix multiplications (`nn.Linear`, attention scores) run in **BF16** on A100 tensor cores.
- Reductions (softmax, RMSNorm) run in **FP32** for numerical stability.
- `autocast` maintains an internal registry of which ops should use which dtype.

**The loss division by `grad_accum_steps`** is the normalization required for gradient accumulation (see §8.11).

#### Chunked Cross-Entropy

```python
logits.view(-1, logits.size(-1))  # (B*S, V)
target_ids.view(-1)                # (B*S,)
```

With `batch_size=96, seq_len=2048, vocab_size=128256`, the full logits tensor has shape `(196608, 128256)` — approximately **50 GB in FP32** or **25 GB in BF16**. This alone would OOM the A100.

`chunked_cross_entropy` processes this in chunks of 65,536 tokens at a time. For each chunk:
1. Slice `logits[start:end]` — only ~0.3 GB
2. Compute `F.cross_entropy` with `reduction='none'`
3. Accumulate the sum of non-ignored losses and the count of non-ignored tokens
4. Divide at the end to get the mean

The result is numerically identical to computing `F.cross_entropy` over the full tensor (the mean is associative). See `tests/test_model.py::TestChunkedCrossEntropy` for the formal equivalence proof.

#### Backward Pass

```python
scaler.scale(loss).backward()
```

`scaler.scale(loss)` multiplies the loss by the current loss scale factor (e.g., 65536). Backpropagation then computes `d(scale * loss)/dθ = scale * d(loss)/dθ`, giving scaled gradients. The unscaling step (§8.11) divides them back.

### 8.11 Gradient Accumulation

```python
if (step + 1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

With `grad_accum_steps=1` (the default), this block runs on every step. When `grad_accum_steps > 1`, gradients from consecutive micro-batches are **accumulated** (added together via repeated `.backward()` calls) before a single optimizer step. This effectively multiplies the batch size without requiring more GPU memory simultaneously.

**The scaler sequence:**

1. `scaler.unscale_(optimizer)` — divides all gradients by the current scale factor, restoring the true gradient magnitudes.
2. `nn.utils.clip_grad_norm_` — computes the global gradient L2 norm and clips to `max_grad_norm=1.0` if exceeded. Gradient clipping is a hard guard against "gradient explosions" where a single bad batch produces enormous gradients that would otherwise destabilize training.
3. `scaler.step(optimizer)` — runs the AdamW update. If any gradient is `inf` or `NaN` (overflow in BF16), the scaler **skips** the optimizer step for this iteration, preventing the corruption from propagating.
4. `scaler.update()` — adjusts the scale factor for the next iteration (increases it if no overflow was detected, decreases it if overflow was detected).
5. `optimizer.zero_grad(set_to_none=True)` — resets gradients. `set_to_none=True` sets `param.grad = None` instead of zeroing the tensor, which saves one memory write per parameter and slightly reduces memory usage.
6. `scheduler.step()` — advances the LR schedule.

**Why clip gradients?** Without clipping, a batch with anomalous data (very long documents, unusual token combinations) can produce gradients orders of magnitude larger than normal. A single such step can move the model far from a good region of parameter space and take many subsequent steps to recover. Clipping at norm `1.0` provides a consistent "maximum step size" guarantee.

### 8.12 Logging

```python
if step % config['log_interval'] == 0 and step > initial_step:
    if device.type == 'cuda':
        torch.cuda.synchronize()
```

**`torch.cuda.synchronize()`** flushes all pending CUDA kernels before measuring step time. Without it, the Python timer would include only the time to *launch* asynchronous CUDA kernels (microseconds), not the time to *execute* them (milliseconds). This is a common source of misleading throughput measurements.

```python
log_dict = {
    'train/loss':             loss.item() * grad_accum_steps,
    'train/lr':               scheduler.get_lr(),
    'train/grad_norm':        grad_norm.item(),
    'train/step_time_ms':     step_time * 1000,
    'train/tokens_per_sec':   tokens_per_step / step_time,
    'train/tokens_seen':      step * tokens_per_step,
    'train/effective_batch':  batch_size * grad_accum_steps,
    'train/data_wait_ms':     data_wait_time * 1000,
    'gpu/memory_used_mb':     torch.cuda.memory_allocated() / 1e6,
    'gpu/memory_peak_mb':     torch.cuda.max_memory_allocated() / 1e6,
    'gpu/memory_reserved_mb': torch.cuda.memory_reserved() / 1e6,
    'gpu/utilization_pct':    torch.cuda.utilization(),
}
```

**`loss.item() * grad_accum_steps`** — The loss was divided by `grad_accum_steps` before backward. For logging, we multiply back to recover the true per-batch loss for interpretability.

**`memory_allocated` vs `memory_reserved`** — PyTorch's allocator reserves large blocks from CUDA and sub-allocates from them. `memory_allocated` is the memory actually holding tensor data. `memory_reserved` is the total block size reserved by the allocator (always >= allocated). The gap between them is the allocator's internal fragmentation.

**`data_wait_ms`** — accumulates the time the training loop spent blocked waiting for the CPU DataLoader workers to produce a batch (across all steps since the last log). A consistently nonzero value indicates the data pipeline is the bottleneck. If this is >5% of step time, consider increasing `num_workers` or `prefetch_factor`.

```python
pbar.set_postfix({
    "loss":    f"{loss.item() * grad_accum_steps:.4f}",
    "lr":      f"{current_lr:.2e}",
    "tok/s":   f"{tokens_per_sec/1e6:.2f}M",
    "data_ms": f"{data_wait_time * 1000:.1f}",
})
data_wait_time = 0.0
```

The tqdm progress bar postfix shows the four most actionable real-time signals: loss trajectory, current LR, throughput, and data stall. `data_wait_time` is reset to 0.0 after each log so it represents the average wait *per log interval*, not cumulative.

### 8.13 Periodic Actions

#### Validation (every 2,000 steps)

```python
if step > 0 and step % config['val_interval'] == 0:
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    val_loss = validate(model, val_dataloader, pad_id, device, step, config)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
```

**`torch.cuda.reset_peak_memory_stats()`** clears the peak memory counter before validation, so the logged peak during validation reflects only the validation pass (not the accumulated peak from training). This makes it easier to separately profile training vs. validation memory usage.

**Best model tracking** — independently of the periodic checkpoint (which just saves the latest), this saves a separate `_best.pt` file whenever a new validation loss minimum is achieved. This is the model to use for inference/evaluation, as it represents the point of best generalization rather than just the latest training state.

**`step > 0`** guard — skips the action at step 0 (before any training has happened) to avoid a meaningless initial validation and to avoid overwriting a just-loaded checkpoint.

#### Text Generation (every 20,000 steps)

```python
if step > 0 and step % config['generation_interval'] == 0:
    generate_samples(model, tokenizer, device, step, config)
```

Sample generation is deliberately infrequent (every 20K steps, about every ~10% of training). It's expensive (5 x 128-token autoregressive decodes without a KV cache) and only useful once the model has learned enough to produce readable text (typically after several thousand steps).

#### Checkpointing (every 5,000 steps)

```python
if step > 0 and step % config['checkpoint_interval'] == 0:
    save_thread = save_checkpoint(model, optimizer, scheduler, step, config,
                                  best_val_loss, async_save=True)
    if save_thread is not None:
        active_save_threads.add(save_thread)
    if config.get('keep_last_n_checkpoints', 0) > 0:
        cleanup_old_checkpoints(config, step)
```

```python
for t in list(active_save_threads):
    if not t.is_alive():
        active_save_threads.discard(t)
```

The `active_save_threads` set is polled at the **end of every step** (not just on checkpoint steps) to clean up completed threads. This ensures the set doesn't grow unboundedly and keeps a handle on in-flight I/O. If a checkpoint thread is still running when the next checkpoint interval arrives, a second thread is launched — the OS handles concurrent file writes to different paths safely.

### 8.14 Post-Training Teardown

```python
total_time = time.time() - training_start_time
print(f"Training completed in {total_time/3600:.2f} hours!")
print(f"Average throughput: {... / 1e6:.2f}M tokens/sec")

save_checkpoint(model, optimizer, scheduler, config['max_steps'], config,
                best_val_loss, is_final=True)
wandb.finish()
```

`wandb.finish()` uploads any buffered logs, marks the run as complete in the W&B UI, and closes the network connection. Without this call, the W&B run remains in "running" state and logs from the end of training may not be uploaded.

The final checkpoint is saved **synchronously** (`is_final=True` bypasses async I/O) to guarantee it completes before the process exits.

---

## 9. End-to-End Data Flow Diagram

```
                    +------------------------------------------+
                    |              train.py                    |
                    +------------------------------------------+
                                        |
                                        v
                    +------------------------------------------+
                    |         build_training_data()            |  dataset.py
                    |   HuggingFace -> tokenize -> memmap disk |
                    +--------------------+---------------------+
                                         |  PackedDataset (uint32 memmap)
                                         |  ShuffledRangeSampler
                                         |  DataLoader (num_workers=6)
                                         v
                    +------------------------------------------+
                    |         CPU DataLoader Workers           |
                    |   Slice chunk -> torch.LongTensor        |
                    +--------------------+---------------------+
                                         |  pinned CPU memory
                                         |  (pin_memory=True)
                                         v
           +-------------------------------------------------------------+
           |               CUDA data_stream (DMA)                        |
           |   non_blocking=True: CPU->GPU transfer runs async           |
           +-----------------------------+-------------------------------+
                                         |  GPU HBM2e
                                         v
           +-------------------------------------------------------------+
           |              Transformer.forward()                          |
           |   BF16 autocast | gradient checkpointing                   |
           |                 |                                           |
           |   InputEmbedding (BF16)                                     |
           |        |                                                    |
           |   x 16 DecoderBlock (gradient checkpointed):               |
           |        +-- RMSNorm -> GQA (Flash Attn, RoPE) -> +x         |
           |        +-- RMSNorm -> SwiGLU (fused) -> +x                 |
           |                 |                                           |
           |   Final RMSNorm                                             |
           |        |                                                    |
           |   output_proj (BF16) -> logits (B, S, V)                   |
           +-----------------------------+-------------------------------+
                                         |
                                         v
           +-------------------------------------------------------------+
           |           chunked_cross_entropy()                           |
           |   Processes (B*S, V) in 65K-token chunks                   |
           |   Peak logits memory: 0.3 GB (vs 50 GB unchunked)          |
           +-----------------------------+-------------------------------+
                                         |  scalar loss
                                         v
           +-------------------------------------------------------------+
           |         scaler.scale(loss).backward()                       |
           |   Gradient checkpointing: recomputes activations            |
           |   per layer on backward pass                                |
           +-----------------------------+-------------------------------+
                                         |  gradients in GPU memory
                                         v
           +-------------------------------------------------------------+
           |  scaler.unscale_() -> clip_grad_norm_() -> AdamW.step()    |
           |  scheduler.step() -> zero_grad(set_to_none=True)           |
           +-------------------------------------------------------------+
```

---

## 10. Key Design Decisions Summary

| Decision | Rationale | Alternative Considered |
|---|---|---|
| **Hand-rolled LR scheduler** | Clean checkpoint serialization, no framework coupling | `torch.optim.lr_scheduler` — harder to resume mid-run |
| **Selective weight decay (dim >= 2)** | RMSNorm scales must not be decayed to zero | Apply decay to all params — would degrade normalization |
| **BF16 over FP16** | A100 native BF16 tensor cores; wider dynamic range; no grad overflow | FP16 — overflow issues at batch_size=96 with large logits |
| **Chunked cross-entropy** | 50 GB -> 0.3 GB logits memory, identical numerics | Standard CE — OOM at batch_size=96 |
| **Gradient checkpointing** | 70 GB -> 3.2 GB activation memory, enables 2x batch size | No checkpointing — OOM, forces batch_size=48 |
| **Async checkpoint I/O** | Eliminates ~5-10s training stall per checkpoint | Synchronous save — blocks training loop |
| **Full RNG state in checkpoint** | Bit-exact resume guaranteed | No RNG save — resumed run diverges from uninterrupted run |
| **SIGTERM/SIGINT handler** | No training loss on preemption | No handler — last 5K steps lost on preemption |
| **`set_to_none=True` in zero_grad** | Saves one memory write per parameter per step | In-place zero — slightly more memory bandwidth |
| **Double-buffer CUDA stream prefetch** | Hides 5-15% CPU->GPU transfer latency | Synchronous transfer — blocks forward pass start |
| **torch.compile()** | Kernel fusion, layout optimization, ~5-10% throughput gain | No compile — more portable but slower |
| **`beta2=0.95` instead of 0.999** | More responsive to recent gradient statistics in large-batch regime | 0.999 — slower adaptation, less stable early training |
| **No weight tying** | Independent output projection can learn different representations | Tied weights — saves 128K x 1024 parameters but constrains the model |
| **Emergency save via `global_state` closure** | Signal handlers cannot take custom arguments; closure captures live refs | Class-based trainer — more complex refactor |

---

*This document covers `train.py` at commit-time. For the model architecture, see [`model_architecture.md`](model_architecture.md). For the data pipeline, see [`data_prep.md`](data_prep.md). For the RoPE implementation, see [`rope.md`](rope.md).*

---

## Appendix — Extracted rationale (from inline comments)

### Optimizer: AdamW with selective weight decay

- **Decoupled weight decay** (AdamW, not Adam) — decay is applied directly
  to the weights after the adaptive step, making it independent of gradient
  history. Consistently outperforms Adam for LLMs.
- **2D+ params only** — weight decay is restricted to weight matrices
  (`dim() >= 2`); 1D params (RMSNorm scales, biases) are exempt. Decaying
  RMSNorm scales toward zero would collapse the normalization layer's
  learned gains and harm training.
- **`beta1 = 0.9`, `beta2 = 0.95`** — LLaMA-3 uses `0.95` rather than the
  common `0.999` for `beta2` (a 20-step effective window vs ~1000). A
  shorter window makes the adaptive scaling more responsive to recent
  gradient magnitudes — beneficial in the large-batch regime where gradient
  statistics change rapidly.
- **`eps = 1e-8`** — denominator guard, rarely matters in practice.

### LR schedule: cosine with linear warmup (3e-4 → 3e-5, 2000 warmup)

- **Linear warmup (steps 0–2000)**: from ~0 to `peak_lr = 3e-4`. At start,
  weights are random and gradients are large/noisy; a full LR would take
  massive steps and destabilize the loss. Warmup lets the optimizer find a
  reasonable direction before committing to large updates.
- **Cosine decay (steps 2000–42000)**: from `peak_lr` down to `min_lr =
  3e-5` (exactly 10× below peak). Cosine spends more time near the peak
  (fast exploration) and more time near the minimum (careful refinement),
  avoiding the abrupt cliff that linear decay produces.
- **`min_lr` floor (3e-5)**: never decays to 0. Very small LRs stall
  progress without improving loss, and zeroing the LR can stagnate Adam's
  `m`/`v` accumulators.
- **Hand-rolled scheduler** (`CosineWithWarmup`): a single `_step` counter
  serializes cleanly into the checkpoint dict, avoiding the
  resume-from-mid-run fragility of `torch.optim.lr_scheduler`. The
  scheduler directly mutates `optimizer.param_groups[*]['lr']` so both the
  `decay` and `no_decay` groups track the same LR (weight decay is the only
  difference between them).

### Mixed precision: BF16 + GradScaler + TF32 + torch.compile + FA2

- **BF16 autocast** via `torch.autocast(dtype=torch.bfloat16)`: matmuls
  run in BF16 on A100 tensor cores; reductions (softmax, RMSNorm) stay in
  FP32 for numerical stability. BF16 is preferred over FP16 because it
  shares FP32's 8-bit exponent range, so gradient underflow is very rare.
- **`GradScaler`** kept enabled for correctness on mixed systems; rarely
  needs to actually scale in BF16 training. Overhead is negligible.
- **TF32** (`allow_tf32 = True`): ~3× faster FP32 matmuls on Ampere (10
  mantissa bits). Safe because the model trains in BF16; TF32 only
  affects residual FP32 matmuls (e.g. optimizer state updates).
- **`torch.compile()`** (PyTorch 2.0+): TorchDynamo traces the graph and
  TorchInductor generates fused Triton kernels. Operator fusion merges
  elementwise ops (e.g. `silu(gate) * up`), eliminating intermediate
  allocations and kernel-launch overhead. One-time compile penalty on
  the first 1–3 steps; negligible over 42 000 steps.
- **Flash-Attention 2**: `F.scaled_dot_product_attention(is_causal=True)`
  dispatches to the FA2 / memory-efficient kernel on A100 — O(S) memory
  instead of O(S²), 2–3× speedup.

### Async CPU→GPU transfer (double-buffered CUDA stream)

A separate `data_stream` overlaps CPU→GPU H2D copy with GPU compute on the
previous batch. `non_blocking=True` issues a DMA transfer from pinned host
memory (`pin_memory=True` in the DataLoader) without blocking the CPU
thread; sync happens when the GPU needs the tensor. Worth **5–15%**
throughput; the `train/data_wait_ms` W&B metric measures residual stall.

### Checkpointing: full RNG-state restore, async I/O

- **Full RNG state** saved in every checkpoint: `rng_torch`, `rng_numpy`,
  `rng_python`, and `rng_cuda` (when CUDA is available). Restoring all
  four sources of randomness is the key enabler of **exact reproducibility**
  — the resumed run is bit-identical to a hypothetical uninterrupted run.
- **`config` snapshot** saved inside the checkpoint so hyperparameters can
  be reconstructed weeks later even if `config.py` changes.
- **Async I/O**: `torch.save` offloaded to a daemon thread so the training
  loop keeps computing the next batch while I/O happens concurrently.
  `checkpoint_copy` (shallow copy of the state dict) ensures the background
  thread works on its own reference; without it, `zero_grad()` could
  corrupt the saved checkpoint before the thread finishes.
- **CPU-ByteTensor fix for RNG state**: `torch.load(map_location=device)`
  moves *all* tensors to the load device — including the RNG state tensor,
  which `torch.random.set_rng_state()` expects to be a CPU ByteTensor. The
  explicit `.cpu().to(torch.uint8)` coercion in `load_checkpoint` makes
  cross-device resume work (regression test: `test_train.py::
  test_load_restores_rng_state_cross_device`).
- **Final checkpoint** writes two files: a full-state checkpoint (for
  exact reproducibility) and a weights-only file (for inference / sharing
  without 3× optimizer-state overhead).
- **`cleanup_old_checkpoints`** sorts by the **integer** step suffix (not
  lexicographically) to avoid the `step_10.pt < step_9.pt` trap; keeps the
  last `keep_last_n_checkpoints` (default 3).

### Periodic actions

- **Validation every 2 000 steps** (100 batches): `torch.cuda.reset_peak_memory_stats()` is called first so the logged peak reflects only the validation pass. A separate `_best.pt` is saved whenever a new validation-loss minimum is achieved.
- **Sample generation every 20 000 steps**: 5 prompts × 128 tokens with top-k/top-p sampling. Deliberately infrequent — expensive (no KV cache) and only useful once the model has learned enough to produce readable text.
- **Checkpointing every 5 000 steps** (keep 3): `active_save_threads` set polled at the end of every step to reap completed threads.

### Sampling: temperature + top-k + top-p

- **Temperature**: `logits / temperature`. `< 1.0` sharpens, `> 1.0`
  flattens. Default `0.8`.
- **Top-k**: zero out (to `-inf`) all but the `top_k` highest logits.
  Eliminates the long tail of negligible-probability tokens.
- **Top-p (nucleus)**: keep the smallest set whose cumulative probability
  exceeds `top_p` (default 0.9). Adapts to the distribution shape — few
  tokens when confident, many when uncertain. The
  `[..., 1:] = [..., :-1]` shift is critical: cumsum exceeds the threshold
  *after* adding a token, so we shift the mask one position right to
  include the token that crossed the threshold.

### Signal handlers

`SIGTERM` / `SIGINT` handlers save an emergency checkpoint via a
`global_state` closure and `wandb.finish()` before `sys.exit(1)`. Without
this, preemption (HPC scheduler, Ctrl+C) loses all training since the last
checkpoint. The step in `global_state['step']` is updated at the top of
each loop iteration, so at most one step's worth of computation is lost.
