# Files

- [Memory, Precision, and Loss Engineering](memory-and-numerics.md) - Explains how checkpointed activations, chunked LM-head loss, mixed precision, SDPA attention, GQA, and fused SwiGLU make the target workload fit while preserving trainable gradients. Defines the numerical-equivalence, stability, compile, and explicit Triton opt-in contracts that govern safe changes.
- [Decoder Model Architecture](model-architecture.md) - Defines the implementation-grounded tensor and execution contract for the decoder-only Transformer, including pre-norm residual blocks, GQA with RoPE, RMSNorm, fused SwiGLU, final normalization, and the untied language-model head. Clarifies the hidden-state API, training loss boundary, configuration invariants, and parameter-count convention.
