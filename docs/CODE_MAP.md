# CODE_MAP — symbol ↔ doc ↔ test map

> **Generated** by `scripts/generate_code_map.py` from the docs' own
> `<module>.py:<symbol>` citations. `tests/test_doc_refs.py` (CI) verifies
> every citation resolves; run this script after doc changes and commit the
> diff. Line-number anchors are banned.

## Modules and the symbols the docs cite

| Module | Cited symbols |
|--------|---------------|
| `benchmark_data.py` | `benchmark`, `build_benchmark_buffer`, `main` |
| `config.py` | `get_config` |
| `data/prepare_data.py` | `LLAMA3_EOS_TOKEN_ID`, `LLAMA3_PAD_TOKEN_ID`, `LLAMA3_TOKENIZER_NAME`, `LLAMA3_VOCAB_SIZE`, `_apply_llama3_defaults`, `main` |
| `data/shared_data/loader.py` | `PackedDataset`, `PackedDataset.__getitem__`, `PackedDataset.__init__`, `ShuffledRangeSampler`, `ShuffledRangeSampler.__iter__`, `ShuffledRangeSampler.set_epoch`, `_SyntheticTokenizerStub`, `_SyntheticTokenizerStub.__len__`, `build_synthetic_data`, `build_tokenizer`, `build_training_data`, `collate_fn` |
| `dataset.py` | `PackedDataset` |
| `kernels/cross_entropy_triton.py` | `HAS_TRITON`, `_MAX_VOCAB_BLOCK`, `_TritonCEWithZ`, `_TritonCEWithZ.backward`, `_triton_ce_z_forward`, `cross_entropy_with_z_pytorch`, `triton_chunked_cross_entropy_with_z` |
| `kernels/rmsnorm_triton.py` | `HAS_TRITON`, `_MAX_BLOCK_SIZE`, `_TritonRMSNorm`, `_triton_rmsnorm_forward`, `rmsnorm_pytorch`, `triton_rmsnorm` |
| `kernels/swiglu_triton.py` | `_TritonSwiGLU`, `_triton_swiglu_forward`, `swiglu_pytorch`, `triton_swiglu` |
| `model.py` | `Decoder`, `Decoder.__init__`, `Decoder.forward`, `DecoderBlock`, `DecoderBlock.__init__`, `DecoderBlock.forward`, `GroupedQueryAttention`, `GroupedQueryAttention.__init__`, `GroupedQueryAttention.forward`, `RMSNorm`, `RMSNorm.__init__`, `RMSNorm.forward`, `RoPE`, `RoPE.__init__`, `RoPE.forward`, `SwiGLUFFN`, `SwiGLUFFN.__init__`, `SwiGLUFFN.forward`, `Transformer`, `Transformer.__init__`, `Transformer._init_weights`, `Transformer.forward`, `Transformer.get_num_params`, `build_transformer`, `chunked_cross_entropy_with_z`, `chunked_head_cross_entropy_with_z` |
| `tests/conftest.py` | `count_params`, `device`, `dtype`, `full_config`, `make_token_stream`, `pytest_addoption`, `pytest_collection_modifyitems`, `seed_everything`, `tiny_config`, `tiny_model`, `weights_dir` |
| `tests/e2e_gpu_smoke.py` | `build_model`, `build_tiny_config`, `check_checkpoint_roundtrip`, `check_chunked_ce`, `check_data_pipeline`, `check_environment`, `check_triton_kernels`, `check_validate`, `device_supports_tf32`, `main`, `train_steps` |
| `tests/test_config.py` | `REQUIRED_KEYS`, `TestGetConfig.test_data_source_weights_positive`, `TestGetConfig.test_has_all_required_keys`, `TestGetConfig.test_known_values`, `TestGetConfig.test_learning_rate_schedule_invariants`, `TestGetConfig.test_no_extra_unknown_keys` |
| `tests/test_model.py` | `TestChunkedCrossEntropyWithZ`, `TestChunkedCrossEntropyWithZ.test_gradients_flow`, `TestChunkedCrossEntropyWithZ.test_matches_ce_plus_zpen_reference`, `TestChunkedCrossEntropyWithZ.test_z_loss_grows_with_logit_magnitude`, `TestChunkedCrossEntropyWithZ.test_z_loss_ignores_ignore_index_positions`, `TestChunkedCrossEntropyWithZ.test_z_weight_zero_matches_pure_ce`, `TestChunkedHeadCrossEntropyWithZ`, `TestChunkedHeadCrossEntropyWithZ.test_gradients_flow_to_hidden_and_head`, `TestChunkedHeadCrossEntropyWithZ.test_matches_dense_ce_plus_z`, `TestChunkedHeadCrossEntropyWithZ.test_matches_dense_ce_with_zero_z`, `TestChunkedHeadCrossEntropyWithZ.test_return_hidden_skips_head`, `TestGroupedQueryAttention`, `TestGroupedQueryAttention.test_causality`, `TestGroupedQueryAttention.test_invalid_n_kv_heads_raises`, `TestGroupedQueryAttention.test_n_rep_consistency`, `TestQKNorm`, `TestQKNorm.test_disabled_attention_is_bit_identical`, `TestQKNorm.test_enabled_attention_does_not_crash`, `TestQKNorm.test_param_count_increases_when_enabled`, `TestRMSNorm`, `TestRMSNorm.test_matches_reference`, `TestRMSNorm.test_output_shape`, `TestRMSNorm.test_scale_invariance`, `TestRMSNorm.test_zero_input_yields_weight`, `TestRoPE`, `TestRoPE.test_buffer_shapes`, `TestRoPE.test_inv_freq_monotonic`, `TestRoPE.test_position_zero_is_identity`, `TestRoPE.test_relative_position_property`, `TestRoPE.test_rotation_is_orthogonal`, `TestSwiGLUFFN`, `TestSwiGLUFFN.test_fused_equals_unfused_reference`, `TestSwiGLUFFN.test_gate_up_proj_has_2x_d_ff_rows`, `TestSwiGLUFFN.test_output_shape`, `TestTransformerForward`, `TestTransformerForward.test_forward_output_shape`, `TestTransformerForward.test_gradient_checkpointing_matches_normal`, `TestTransformerForward.test_gradient_checkpointing_matches_normal_in_training`, `TestTransformerParamCount`, `TestTransformerParamCount.test_full_model_total_params`, `TestTransformerParamCount.test_get_num_params_definition_mismatch` |
| `tests/test_smoke.py` | `TestEndToEndSmoke`, `TestEndToEndSmoke.test_validate_runs_and_returns_finite_loss`, `tiny_dataloaders` |
| `tests/test_train.py` | `TestCheckpointRoundTrip`, `TestCheckpointRoundTrip.test_async_save_returns_thread`, `TestCheckpointRoundTrip.test_load_restores_model_weights`, `TestCheckpointRoundTrip.test_load_restores_rng_state`, `TestCheckpointRoundTrip.test_load_restores_rng_state_cross_device`, `TestSetupGpuOptimizations`, `TestSetupGpuOptimizations.test_idempotent_on_cpu`, `TestTopKTopPSampling`, `TestTopKTopPSampling.test_handles_neg_inf_logits`, `make_tiny_scheduler` |
| `train.py` | `_head_weight`, `_next_batch`, `generate_samples`, `load_checkpoint`, `save_checkpoint`, `setup_gpu_optimizations`, `top_k_top_p_sampling`, `train_model`, `validate` |

