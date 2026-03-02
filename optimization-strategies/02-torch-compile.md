# Strategy 2: torch.compile() on S3Gen Components

## Overview
The S3Gen vocoder consists of several neural network components that run in eager mode (no compilation). Applying torch.compile() can fuse operations, reduce Python dispatch overhead, and enable kernel-level optimizations. No torch.compile() is used anywhere in the current codebase.

## Current Inference Call Graph
```
S3Token2Wav.inference()
  |-- flow_inference() -> CausalMaskedDiffWithXvec.inference()
  |     |-- UpsampleConformerEncoder (6 blocks, multi-head attention)
  |     |-- CausalConditionalCFM.forward()
  |     |     \-- solve_euler() [n_timesteps x ConditionalDecoder.forward()]
  |     |          \-- ConditionalDecoder: U-Net with ResBlocks + Transformers
  |     \-- ConditionalDecoder estimator
  \-- hift_inference() -> HiFTGenerator.inference()
        |-- ConvRNNF0Predictor.forward()
        |-- SineGen / SourceModuleHnNSF
        \-- decode() [HiFiGAN ResBlocks + ISTFT]
```

## Key Files
- `src/chatterbox_vllm/models/s3gen/s3gen.py` -- S3Token2Wav entry points
- `src/chatterbox_vllm/models/s3gen/flow_matching.py` -- CausalConditionalCFM, solve_euler(), ConditionalDecoder
- `src/chatterbox_vllm/models/s3gen/hifigan.py` -- HiFTGenerator (mel to waveform)
- `src/chatterbox_vllm/models/s3gen/f0_predictor.py` -- ConvRNNF0Predictor
- `src/chatterbox_vllm/models/s3gen/xvector.py` -- CAMPPlus speaker encoder
- `src/chatterbox_vllm/models/s3gen/decoder.py` -- ConditionalDecoder (U-Net)
- `src/chatterbox_vllm/models/s3gen/utils/mask.py` -- Dynamic chunk masking

## Blockers and Required Fixes

### 1. Dynamic Audio Lengths (SEVERE)
Every unique sequence length combination triggers graph recompilation (2-5 seconds each). Variable-length speech tokens are fundamental to this system.

Fix: Implement shape bucketing -- group sequences into length ranges (e.g., 0-100, 100-300, 300-1000 tokens) and pad to bucket boundaries. Pre-compile one graph per bucket during warmup. Configure `torch._dynamo.config.cache_size_limit` appropriately.

### 2. Data-Dependent Control Flow (HIGH)
- `flow_matching.py`: `if B == 1` branch for noise generation
- `mask.py`: `torch.randint(...).item()` causes graph break (`.item()` exits the compiled graph)
- `flow_matching.py`: `x_in = torch.zeros([2 * B, ...])` -- shape depends on runtime batch size

Fix: Remove `.item()` calls in inference paths. Disable dynamic chunk masking during inference (`use_dynamic_chunk=False`). Use `torch.compile(dynamic=True)` or pad to fixed batch sizes for CFG.

### 3. In-Place Operations (MODERATE)
- `xvector.py`: `torch.nn.ReLU(inplace=True)` in multiple CAMLayer instances
- Various residual connection patterns

Fix: Replace all `inplace=True` activations with `inplace=False` before compilation. Can be done programmatically:
```python
def fix_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, name, torch.nn.ReLU(inplace=False))
        fix_inplace(child)
```

### 4. STFT/ISTFT Operations (LOW)
HiFiGAN uses torch.stft and torch.istft. These are compilable in PyTorch 2.0+ but shape inference can be conservative. Not a blocker.

## Component-by-Component Feasibility

| Component | Feasibility | Compile Mode | Notes |
|-----------|-------------|--------------|-------|
| ConvRNNF0Predictor | HIGH | reduce-overhead | Tiny model, simple sequential convs, no blockers |
| HiFTGenerator.decode() | HIGH | max-autotune | CNN-heavy, benefits from operator fusion. Compile decode() not inference() |
| ConditionalDecoder.forward() | MODERATE | default | Called n_timesteps times per chunk. In-place ReLU fix needed. Large model |
| UpsampleConformerEncoder | MODERATE | reduce-overhead | Dynamic chunk masking must be disabled for inference |
| CausalConditionalCFM.solve_euler() | LOW-MODERATE | reduce-overhead + dynamic | CFG batch doubling, iterative loop, shape-dependent allocations |
| CAMPPlus (speaker encoder) | HIGH | reduce-overhead | One-time call per request, in-place ReLU fix needed |

## Implementation Plan

### Phase 1: Low-risk components
1. Compile ConvRNNF0Predictor.forward() with mode='reduce-overhead'
2. Compile HiFTGenerator.decode() with mode='max-autotune'
3. Where to add: In S3Token2Wav.__init__() or after model loading in tts.py

### Phase 2: Core estimator
1. Fix in-place ReLU in xvector.py and decoder.py
2. Disable dynamic chunk masking for inference in mask.py
3. Compile ConditionalDecoder.forward() with mode='default'
4. Implement shape bucketing for variable-length sequences

### Phase 3: Full pipeline (optional, high risk)
1. Attempt compiling CausalConditionalCFM.solve_euler()
2. Requires refactoring CFG batch-doubling logic
3. May need dynamic=True and careful shape management

## Warmup Strategy
torch.compile() has a cold-start cost (2-5 seconds per unique shape). For production:
1. During server startup, run dummy inputs at common sequence lengths (50, 100, 200, 500 tokens)
2. Cache compiled graphs via torch._dynamo.config.cache_dir
3. Set cache_size_limit to accommodate shape buckets

## Compile Mode Selection
- `reduce-overhead`: Fastest mode activation, minimal recompilation. Best for inference with repeated shapes.
- `max-autotune`: Tests multiple kernel implementations, picks fastest. 1-2 min warmup but best steady-state performance. Best for HiFiGAN.
- `default`: Balanced. Best for complex models like ConditionalDecoder where max-autotune warmup is prohibitive.

## Expected Gains
| Component | Estimated Speedup |
|-----------|------------------|
| ConvRNNF0Predictor | 1.3-1.5x |
| HiFTGenerator.decode() | 1.5-2.0x |
| ConditionalDecoder | 1.5-2.0x |
| Full S3Gen pipeline | 1.5-2.0x overall |

Gains assume fixed shapes after warmup. Variable shapes negate most benefits (recompilation overhead).

## Risks
- Recompilation storms if shape bucketing is too granular
- Numerical differences between eager and compiled execution (rare but possible)
- Debugging compiled code is harder (stack traces are less informative)
- Warmup latency on first request per shape bucket
- Some operations may silently fall back to eager mode (graph breaks)

## Validation
- Compare audio output (waveform-level) between eager and compiled execution to confirm numerical equivalence
- Measure per-component latency before/after with CUDA events
- Monitor graph break count via torch._dynamo.utils.counters
- Test with variable-length inputs to confirm shape bucketing works
- Load test under concurrency to confirm no race conditions in compiled code

## Recommendation
Start with Phase 1 (F0 predictor + HiFiGAN decode). These are the safest targets with clear gains. Phase 2 (ConditionalDecoder) is where the bulk of compute lives and offers the most impact, but requires the prerequisite fixes. Phase 3 is optional and likely not worth the complexity unless latency requirements are extreme.
