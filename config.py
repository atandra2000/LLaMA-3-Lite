from pathlib import Path


def get_config():
    return {
        # Model Architecture — LLaMA 3 (515M parameters)
        'd_model':              1024,
        'n_layers':             16,
        'n_heads':              8,
        'n_kv_heads':           4,
        'head_dim':             128,
        'd_ff':                 4096,
        'vocab_size':           128000,
        'seq_len':              2048,
        'rope_theta':           500000.0,
        'rms_norm_eps':         1e-5,
        'dropout':              0.0,
        'tie_embeddings':       False,
        'bias':                 False,

        # Training — Optimized for A100 80GB SXM
        # BS=96 with gradient checkpointing + chunked CE fits ~20GB (vs ~92GB without)
        'batch_size':           96,
        'gradient_accumulation': 1,
        'max_steps':            42000,
        'learning_rate':        3e-4,
        'min_lr':               3e-5,
        'warmup_steps':         2000,
        'weight_decay':         0.1,
        'max_grad_norm':        1.0,

        # Optimizer — AdamW with decoupled weight decay (2D+ params only)
        'optimizer':            'AdamW',
        'beta1':                0.9,
        'beta2':                0.95,
        'eps':                  1e-8,

        # LR Schedule — Cosine with linear warmup
        'lr_scheduler':         'cosine',
        'warmup_style':         'linear',

        # A100 Optimizations
        'dtype':                'bfloat16',
        'use_flash_attention':  True,
        'compile_model':        True,
        'gradient_checkpointing': True,  # ~55% memory reduction
        'use_chunked_cross_entropy': True,  # ~100x logits memory reduction

        # GPU System Configuration
        'tf32':                 True,  # ~3x matmul speedup on A100
        'cudnn_benchmark':      True,
        'cuda_alloc_conf':      'expandable_segments:True',

        # Data Sources — Multi-source code + text mix
        'data_sources': {
            'fineweb_edu':           {'weight': 0.5, 'source': 'HuggingFaceFW/fineweb-edu',
                                      'split': 'train'},
            'fineweb_code':          {'weight': 0.1, 'source': 'HuggingFaceFW/fineweb-edu',
                                      'split': 'train',
                                      'filter': 'code',
                                      'filter_mode': 'word'},
            'the_stack_python':      {'weight': 0.2, 'source': 'bigcode/the-stack',
                                      'split': 'train',
                                      'languages': ['Python']},
            'the_stack_multilang':   {'weight': 0.05, 'source': 'bigcode/the-stack',
                                      'split': 'train',
                                      'languages': ['JavaScript', 'TypeScript', 'Rust', 'Go',
                                                     'C', 'C++', 'Java', 'SQL', 'Shell']},
            'wikipedia':             {'weight': 0.05, 'source': 'wikimedia/wikipedia',
                                      'split': '20231101.en'},
            'stackoverflow_qa':      {'weight': 0.05, 'source': 'open-phi/StackOverflow-QA',
                                      'split': 'train'},
        },
        'num_workers':          6,
        'prefetch_factor':       16,
        'pin_memory':            True,
        'document_packing':      True,
        'target_tokens':         4_000_000_000,

        # Data Pipeline — Streaming + Disk-Backed Token Cache
        'data_cache_dir':        'data_cache',
        'data_cache_filename':   'tokens.bin',
        'reuse_data_cache':      True,
        'shuffle_documents':     True,
        'shuffle_seed':          42,
        'dedup':                 True,
        'dedup_hash_bytes':      256,
        'min_doc_tokens':        16,
        'max_doc_tokens':        8192,
        'tokenize_batch_size':   1000,

        # Tokenizer — LLaMA 3 (128K vocab)
        'tokenizer_name':       'NousResearch/Meta-Llama-3-8B',
        'tokenizer_type':       'autotokenizer',
        'tokenizer_cache_dir':  None,

        # Validation & Generation
        'val_interval':         2000,
        'val_max_batches':      100,
        'val_split':            0.05,
        'generation_interval':  20000,
        'generation_max_tokens': 128,
        'generation_temperature': 0.8,
        'generation_top_k':     50,

        # Checkpointing
        'model_folder':         'weights',
        'model_filename':      'llama3-515M',
        'checkpoint_interval':  5000,
        'keep_last_n_checkpoints': 3,
        'async_checkpoint':     True,
        'preload':              None,

        # W&B Logging
        'wandb_project':        'langgpt-llama3-pretrain',
        'wandb_entity':         None,
        'wandb_tags':           ['llama3', '515M', 'a100', 'pretrain', 'code'],
        'log_interval':         50,

        # Sampling
        'top_k':                50,
        'temperature':          0.8,
    }


def get_weights_file_path(config, step: int):
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}_step_{step}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    """Return the path to the highest-numbered checkpoint, or None.

    Sorts by the integer step number embedded in the filename so that
    ``step_20.pt`` is correctly chosen over ``step_9.pt`` (a plain lexical
    sort would pick ``step_9.pt`` because '9' > '2' as characters).
    """
    model_folder = Path(config['model_folder'])
    if not model_folder.exists():
        return None
    checkpoints = list(model_folder.glob(f"{config['model_filename']}_step_*.pt"))
    if not checkpoints:
        return None
    # Sort by the integer step suffix to avoid the lexical-sort trap
    # (e.g. "step_10.pt" vs "step_9.pt"). Mirrors cleanup_old_checkpoints.
    checkpoints.sort(
        key=lambda x: int(str(x.stem).split('_step_')[-1])
        if str(x.stem).split('_step_')[-1].isdigit() else -1
    )
    return str(checkpoints[-1])


def cleanup_old_checkpoints(config, current_step):
    model_folder = Path(config['model_folder'])
    if not model_folder.exists():
        return

    keep_n = config.get('keep_last_n_checkpoints', 3)
    checkpoint_files = sorted(
        model_folder.glob(f"{config['model_filename']}_step_*.pt"),
        key=lambda x: int(str(x.stem).split('_step_')[-1])
    )

    if len(checkpoint_files) > keep_n:
        for old_file in checkpoint_files[:-keep_n]:
            old_file.unlink()
            print(f"Removed old checkpoint: {old_file}")
