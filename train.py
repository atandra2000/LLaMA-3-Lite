import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import sys
import time
import math
import random
import signal
import threading
import numpy

import wandb

from dataset import build_training_data
from model import build_transformer, chunked_cross_entropy
from config import get_config, cleanup_old_checkpoints

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_gpu_optimizations(config):
    """Configure A100 GPU for maximum throughput (TF32, BF16, cuDNN benchmark)."""
    if config.get('tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', True)
    torch.backends.cudnn.deterministic = False

    if 'cuda_alloc_conf' in config:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['cuda_alloc_conf']

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name} ({total_mem:.1f} GB)")
        if hasattr(torch.cuda, 'get_device_capability'):
            cap = torch.cuda.get_device_capability(0)
            print(f"CUDA Compute Capability: {cap[0]}.{cap[1]}")


class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float, peak_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.peak_lr = peak_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self._step < self.warmup_steps:
            return self.peak_lr * self._step / self.warmup_steps
        else:
            progress = (self._step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']


def top_k_top_p_sampling(logits, top_k, top_p, temperature):
    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_vals, top_k_indices = logits.topk(top_k, dim=-1)
        logits = torch.full_like(logits, float('-inf')).scatter_(-1, top_k_indices, top_k_vals)

    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        logits = torch.full_like(logits, float('-inf')).scatter(
            -1, sorted_indices, sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
        )

    probs = logits.softmax(dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate_samples(model, tokenizer, device, step, config):
    model.eval()
    prompts = [
        "The history of artificial intelligence began in the",
        "In a surprising discovery, researchers found that",
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    ",
        "class BinaryTree:\n    def __init__(self, value):\n        ",
        "import numpy as np\n\ndef calculate_mean(data):\n    ",
    ]

    table = wandb.Table(columns=["prompt", "generated", "step"])
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        generated = input_ids
        for _ in range(config['generation_max_tokens']):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(generated)
            next_token = top_k_top_p_sampling(
                logits[:, -1, :],
                config['generation_top_k'],
                top_p=0.9,
                temperature=config['generation_temperature']
            )
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated[0].tolist())
        table.add_data(prompt, text, step)

    wandb.log({"gen/samples": table}, step=step)
    model.train()


@torch.no_grad()
def validate(model, val_dataloader, pad_id, device, step, config):
    """Validation loop with chunked cross-entropy for memory efficiency."""
    model.eval()
    total_loss = 0
    num_batches = 0

    val_max_batches = config.get('val_max_batches', 200)
    use_chunked_ce = config.get('use_chunked_cross_entropy', True)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in val_dataloader:
            if num_batches >= val_max_batches:
                break
            input_ids = batch['input'].to(device, non_blocking=True)
            target_ids = batch['target'].to(device, non_blocking=True)

            logits = model(input_ids)

            if use_chunked_ce:
                loss = chunked_cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    chunk_size=65536,
                    ignore_index=pad_id,
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=pad_id
                )

            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))

    wandb.log({
        'val/loss': avg_loss,
        'val/perplexity': perplexity,
    }, step=step)

    model.train()
    return avg_loss


def _save_checkpoint_to_disk(checkpoint, path):
    """Background worker for async checkpoint I/O."""
    try:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    except Exception as exc:
        print(f"[checkpoint] ERROR saving {path}: {exc}")


def save_checkpoint(model, optimizer, scheduler, step, config, best_val_loss=None,
                    is_final=False, async_save=True):
    """Save checkpoint with optional async I/O to minimize training pause."""
    model_folder = Path(config['model_folder'])
    model_folder.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'tokens_seen': step * config['batch_size'] * config['seq_len'] * config.get('gradient_accumulation', 1),
        'best_val_loss': best_val_loss,
        'rng_torch': torch.random.get_rng_state(),
        'rng_numpy': numpy.random.get_state(),
        'rng_python': random.getstate(),
        'config': config,
    }
    if torch.cuda.is_available():
        checkpoint['rng_cuda'] = torch.cuda.get_rng_state()

    if is_final:
        final_path = model_folder / f"{config['model_filename']}_final_model_full.pt"
        torch.save(checkpoint, final_path)
        model_only_path = model_folder / f"{config['model_filename']}_final_model_weights.pt"
        torch.save(model.state_dict(), model_only_path)
        print(f"Final model saved: {final_path}")
        print(f"Model weights only: {model_only_path}")
        return None

    path = model_folder / f"{config['model_filename']}_step_{step}.pt"

    if async_save and config.get('async_checkpoint', True):
        checkpoint_copy = {k: v.copy() if hasattr(v, 'copy') else v for k, v in checkpoint.items()}
        for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
            if isinstance(checkpoint_copy[key], dict):
                checkpoint_copy[key] = checkpoint_copy[key].copy()
        thread = threading.Thread(target=_save_checkpoint_to_disk, args=(checkpoint_copy, path), daemon=True)
        thread.start()
        print(f"Checkpoint offloaded to background thread: {path}")
        return thread
    else:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        return None


def load_checkpoint(model, optimizer, scheduler, config, device):
    model_folder = Path(config['model_folder'])
    # Sort by integer step suffix to avoid the lexical-sort trap where
    # "step_10.pt" sorts before "step_9.pt". Mirrors the fix in
    # config.latest_weights_file_path.
    checkpoints = sorted(
        model_folder.glob(f"{config['model_filename']}_step_*.pt"),
        key=lambda x: int(str(x.stem).split('_step_')[-1])
        if str(x.stem).split('_step_')[-1].isdigit() else -1,
    )

    if not checkpoints:
        return 0, float('inf')

    latest = checkpoints[-1]
    checkpoint = torch.load(latest, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # The CPU RNG state must be a CPU ByteTensor. torch.load(map_location=...)
    # moves *all* tensors in the checkpoint to `device`, which corrupts the
    # RNG state when resuming on a different device (e.g. GPU). Move it back
    # to CPU explicitly. Same idea for the CUDA RNG state below.
    rng_torch = checkpoint['rng_torch']
    if isinstance(rng_torch, torch.Tensor):
        rng_torch = rng_torch.cpu().to(torch.uint8)
    torch.random.set_rng_state(rng_torch)
    numpy.random.set_state(checkpoint['rng_numpy'])
    random.setstate(checkpoint['rng_python'])
    if 'rng_cuda' in checkpoint and torch.cuda.is_available():
        rng_cuda = checkpoint['rng_cuda']
        # torch.cuda.set_rng_state expects a CPU ByteTensor (it dispatches the
        # state to the target device internally). torch.load(map_location=...)
        # may have moved it to the load device, so force it back to CPU + uint8.
        if isinstance(rng_cuda, torch.Tensor):
            rng_cuda = rng_cuda.cpu().to(torch.uint8)
        torch.cuda.set_rng_state(rng_cuda)

    print(f"Resumed from step {checkpoint['step']}")
    return checkpoint['step'], checkpoint.get('best_val_loss', float('inf'))


def train_model(config, train_dataloader=None, val_dataloader=None, tokenizer=None):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    if device.type == 'cuda':
        setup_gpu_optimizations(config)

    if train_dataloader is None or val_dataloader is None or tokenizer is None:
        train_dataloader, val_dataloader, tokenizer = build_training_data(config)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    gradient_checkpointing = config.get('gradient_checkpointing', True)
    real_vocab_size = max(config['vocab_size'], len(tokenizer))
    model = build_transformer(
        vocab_size=real_vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        head_dim=config['head_dim'],
        d_ff=config['d_ff'],
        max_seq_len=config['seq_len'],
        rope_theta=config['rope_theta'],
        rms_norm_eps=config['rms_norm_eps'],
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    non_embed_params = (model.get_num_params(non_embedding=True)
                         if hasattr(model, "get_num_params") else None)
    model_mem_gb = num_params * 2 / 1e9
    print(f"\n{'='*60}")
    print(f"Model: {num_params/1e6:.1f}M parameters ({model_mem_gb:.2f} GB in BF16)")
    if non_embed_params is not None:
        print(f"Non-embedding parameters: {non_embed_params/1e6:.1f}M")
    print(f"Gradient checkpointing: {'ON' if gradient_checkpointing else 'OFF'}")
    bs, seq = config['batch_size'], config['seq_len']
    tokens_per_step = bs * seq
    if gradient_checkpointing:
        est_peak = model_mem_gb * 7.2 + tokens_per_step * config['vocab_size'] * 2 / 1e9 + 3
    else:
        est_peak = model_mem_gb * 7.2 + tokens_per_step * 2 * 16 * config['d_model'] * 2 / 1e9 + 3
    print(f"Batch size: {bs} | Seq len: {seq} | Tokens/step: {tokens_per_step:,}")
    print(f"Estimated peak GPU memory: {est_peak:.1f} GB (A100 80GB available)")
    print(f"{'='*60}\n")

    # torch.compile() for kernel fusion and operator optimization on A100
    if config.get('compile_model', True) and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    decay_params = []
    no_decay_params = []
    for param in model.named_parameters():
        if not param[1].requires_grad:
            continue
        if param[1].dim() >= 2:
            decay_params.append(param[1])
        else:
            no_decay_params.append(param[1])

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config['learning_rate'], betas=(config['beta1'], config['beta2']),
       eps=config['eps'])

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: AdamW with decoupled weight decay")
    print(f"  Decay params: {n_decay:,} ({n_decay/1e6:.1f}M)")
    print(f"  No-decay params: {n_no_decay:,} ({n_no_decay/1e6:.1f}M)")

    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        min_lr=config['min_lr'],
        peak_lr=config['learning_rate'],
    )

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    initial_step, best_val_loss = 0, float('inf')
    if config.get('preload') is not None:
        initial_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, config, device)

    wandb.init(
        project=config['wandb_project'],
        entity=config.get('wandb_entity'),
        name=f"llama3-515M-{device}-{int(time.time())}",
        config={
            "architecture": "LLaMA 3",
            "d_model": config['d_model'],
            "n_layers": config['n_layers'],
            "n_heads": config['n_heads'],
            "n_kv_heads": config['n_kv_heads'],
            "d_ff": config['d_ff'],
            "vocab_size": config['vocab_size'],
            "seq_len": config['seq_len'],
            "params_total": num_params,
            "params_non_embed": non_embed_params if non_embed_params is not None
                                 else 0,
            "batch_size": config['batch_size'],
            "gradient_accumulation": config.get('gradient_accumulation', 1),
            "learning_rate": config['learning_rate'],
            "min_lr": config['min_lr'],
            "warmup_steps": config['warmup_steps'],
            "max_steps": config['max_steps'],
            "optimizer": "AdamW",
            "beta1": config['beta1'],
            "beta2": config['beta2'],
            "weight_decay": config['weight_decay'],
            "precision": "bf16",
            "gradient_checkpointing": gradient_checkpointing,
            "chunked_cross_entropy": config.get('use_chunked_cross_entropy', True),
            "torch_compile": config.get('compile_model', True),
        },
        tags=config.get('wandb_tags', []),
    )

    global_state = {'step': initial_step, 'model': model, 'optimizer': optimizer,
                    'scheduler': scheduler, 'config': config, 'best_val_loss': best_val_loss}

    def emergency_save_handler(signum, frame):
        print(f"\nSignal {signum} received. Saving emergency checkpoint...")
        save_checkpoint(
            global_state['model'], global_state['optimizer'], global_state['scheduler'],
            global_state['step'], global_state['config'], global_state['best_val_loss']
        )
        wandb.finish()
        sys.exit(1)

    signal.signal(signal.SIGTERM, emergency_save_handler)
    signal.signal(signal.SIGINT, emergency_save_handler)

    active_save_threads = set()

    print(f"\nStarting training for {config['max_steps']} steps...")
    model.train()

    if device.type == 'cuda':
        data_stream = torch.cuda.Stream()

    grad_accum_steps = config.get('gradient_accumulation', 1)
    tokens_per_step = config['batch_size'] * config['seq_len'] * grad_accum_steps
    use_chunked_ce = config.get('use_chunked_cross_entropy', True)

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    step_iterator = iter(train_dataloader)
    training_start_time = time.time()
    step_start_time = time.time()
    data_wait_time = 0.0
    warmup_steps = 5

    pbar = tqdm(range(initial_step, config['max_steps']), desc="Training", unit="step")

    data_start = time.time()
    try:
        next_batch = next(step_iterator)
    except StopIteration:
        step_iterator = iter(train_dataloader)
        next_batch = next(step_iterator)
    data_wait_time += time.time() - data_start

    if device.type == 'cuda':
        with torch.cuda.stream(data_stream):
            next_input = next_batch['input'].to(device, non_blocking=True)
            next_target = next_batch['target'].to(device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(data_stream)
    else:
        next_input = next_batch['input'].to(device)
        next_target = next_batch['target'].to(device)

    for step in pbar:
        input_ids = next_input
        target_ids = next_target

        data_start = time.time()
        try:
            batch = next(step_iterator)
        except StopIteration:
            step_iterator = iter(train_dataloader)
            batch = next(step_iterator)
        fetch_time = time.time() - data_start
        data_wait_time += fetch_time

        if device.type == 'cuda':
            with torch.cuda.stream(data_stream):
                next_input = batch['input'].to(device, non_blocking=True)
                next_target = batch['target'].to(device, non_blocking=True)

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
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=pad_id
                )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if device.type == 'cuda':
            torch.cuda.current_stream().wait_stream(data_stream)

        if device.type == 'cuda':
            end_event.record()

        step_time = time.time() - step_start_time
        step_start_time = time.time()

        if step % config['log_interval'] == 0 and step > initial_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()

            current_lr = scheduler.get_lr()
            tokens_seen = step * tokens_per_step
            tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0
            effective_batch = config['batch_size'] * grad_accum_steps
            gpu_util = torch.cuda.utilization() if device.type == 'cuda' else None

            log_dict = {
                'train/loss': loss.item() * grad_accum_steps,
                'train/lr': current_lr,
                'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'train/step_time_ms': step_time * 1000,
                'train/tokens_per_sec': tokens_per_sec,
                'train/tokens_seen': tokens_seen,
                'train/effective_batch_size': effective_batch,
                'train/data_wait_ms': data_wait_time * 1000,
            }

            if device.type == 'cuda':
                log_dict['gpu/memory_used_mb'] = torch.cuda.memory_allocated() / 1e6
                log_dict['gpu/memory_peak_mb'] = torch.cuda.max_memory_allocated() / 1e6
                log_dict['gpu/memory_reserved_mb'] = torch.cuda.memory_reserved() / 1e6
                if gpu_util is not None:
                    log_dict['gpu/utilization_pct'] = gpu_util

            wandb.log(log_dict, step=step)

            pbar.set_postfix({
                "loss": f"{loss.item() * grad_accum_steps:.4f}",
                "lr": f"{current_lr:.2e}",
                "tok/s": f"{tokens_per_sec/1e6:.2f}M",
                "data_ms": f"{data_wait_time * 1000:.1f}",
            })
            data_wait_time = 0.0

        if step > 0 and step % config['val_interval'] == 0:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            val_loss = validate(model, val_dataloader, pad_id, device, step, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                global_state['best_val_loss'] = best_val_loss
                best_model_path = Path(config['model_folder']) / f"{config['model_filename']}_best.pt"
                Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved (val_loss: {val_loss:.4f})")
            model.train()

        if step > 0 and step % config['generation_interval'] == 0:
            generate_samples(model, tokenizer, device, step, config)

        if step > 0 and step % config['checkpoint_interval'] == 0:
            save_thread = save_checkpoint(model, optimizer, scheduler, step, config,
                                          best_val_loss, async_save=True)
            if save_thread is not None:
                active_save_threads.add(save_thread)
            if config.get('keep_last_n_checkpoints', 0) > 0:
                cleanup_old_checkpoints(config, step)

        for t in list(active_save_threads):
            if not t.is_alive():
                active_save_threads.discard(t)

    total_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours!")
    print(f"Average throughput: {config['max_steps'] * tokens_per_step / total_time / 1e6:.2f}M tokens/sec")

    save_checkpoint(model, optimizer, scheduler, config['max_steps'], config, best_val_loss, is_final=True)

    wandb.finish()

    print(f"All artifacts saved to: {config['model_folder']}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)