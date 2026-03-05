# Strategy 3: Int8 Quantization of S3Gen

## Overview
The S3Gen vocoder contains approximately 600-700M parameters across its components. Currently, flow matching, speaker encoder, and HiFiGAN run in bfloat16 (~1.2-1.4 GB), while the S3 tokenizer stays in float32 (cuFFT requirement). Int8 weight quantization could reduce memory by 40-50% and potentially improve throughput through reduced memory bandwidth. No quantization code currently exists in the codebase.

## Current Dtype Strategy
From tts.py (lines 263-270):
- float32: S3Tokenizer (torch.stft / cuFFT does not support bfloat16)
- bfloat16: flow matching (CausalMaskedDiffWithXvec), speaker encoder (CAMPPlus), HiFiGAN (HiFTGenerator)
- The switch from float16 to bfloat16 was made for numerical stability in iterative diffusion

## Component Breakdown

### Flow Matching (~300-450M params)
- Encoder: UpsampleConformerEncoder -- Conv1d, Linear (attention/feed-forward), 6 blocks, 512 output dim, 8 heads, 2048 linear units
- Decoder: ConditionalDecoder -- U-Net with CausalResnetBlock1D, BasicTransformerBlocks, Conv1d, ConvTranspose1d, GroupNorm, LayerNorm. 12 mid blocks.
- CFM solver: Linear operations for speaker embedding projection and attention

### HiFiGAN (~160-220M params)
- Conv_pre: weight_norm(Conv1d(80->512, kernel=7))
- Upsample path: ConvTranspose1d with rates [8, 5, 3]
- ResBlocks: Dilated Conv1d with Snake activation (learnable alpha)
- Conv_post: weight_norm(Conv1d(ch->n_fft+2, kernel=7))
- ISTFT: magnitude exp() and phase sin/cos reconstruction
- F0 Predictor: 5 Conv1d layers (~10-20M params)

### Speaker Encoder (CAMPPlus, ~2-6M params)
- FCM head: Conv2d + BatchNorm2d + ResBlocks
- TDNN layers: Conv1d with BatchNorm1d
- CAMDenseTDNNBlocks: Dense Conv1d layers with channel attention
- Final: Conv1d(channels*2 -> 192) embedding output

## Key Files
- `src/chatterbox_vllm/models/s3gen/s3gen.py` -- S3Token2Wav, model loading and inference
- `src/chatterbox_vllm/models/s3gen/flow_matching.py` -- CFM decoder, Euler solver
- `src/chatterbox_vllm/models/s3gen/decoder.py` -- ConditionalDecoder (U-Net)
- `src/chatterbox_vllm/models/s3gen/hifigan.py` -- HiFTGenerator vocoder
- `src/chatterbox_vllm/models/s3gen/f0_predictor.py` -- ConvRNNF0Predictor
- `src/chatterbox_vllm/models/s3gen/xvector.py` -- CAMPPlus speaker encoder
- `src/chatterbox_vllm/tts.py` -- Model loading (lines 263-270)

## Quantization Approaches

### A. torch.ao Dynamic Quantization
- API: `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`
- Only quantizes Linear layers (not Conv1d)
- No calibration data needed
- For S3Gen (Conv-heavy): limited benefit (~25-30% reduction)

### B. Int8 Weight-Only Quantization
- Replace Conv1d/Linear weights with int8 + scale factors
- Dequantize on-the-fly during inference
- Covers more layers than dynamic quantization
- Risk: dequantization overhead on GPU may negate latency gains

### C. bitsandbytes
- Optimized mixed-precision inference via custom kernels
- Primary focus is Linear layers (LLM-oriented)
- Less tested for Conv1d-heavy audio models
- External dependency

### D. GPTQ-Style
- Second-order Hessian correction for near-lossless weight quantization
- Requires calibration data (real audio samples)
- Computationally expensive quantization pass
- Overkill for this use case

## Recommended Approach: Selective Weight-Only Quantization

### Safe to Quantize (int8)
- HiFiGAN ResBlock Conv1d layers (standard convolutions, non-critical timing)
- Encoder Conv1d downsampling layers (early processing, less sensitive)
- Decoder down/up block Conv1d layers (with quality testing)
- F0 Predictor Conv1d (5 layers, small, F0 is a smooth signal)
- Upsample ConvTranspose1d layers
- Feed-forward Linear layers in transformer blocks (non-attention)

### Keep in bfloat16/float32 (do NOT quantize)
- Diffusion timestep embeddings (solver stability critical)
- Attention Q, K, V, Output projection layers (core transformer precision)
- All normalization layers (LayerNorm, GroupNorm, BatchNorm) -- affect signal statistics
- ISTFT magnitude/phase computation -- exp() and sin/cos are numerically sensitive
- Snake activation parameters (learnable alpha controls activation curve shape)
- Speaker encoder CAM attention layers (discrimination-critical)
- Speaker encoder final embedding output layer
- S3 Tokenizer (cuFFT requires float32)

## Changes Required

### 1. Implement selective quantization function
Create a function that walks the model graph, identifies quantizable layers by type and position, and replaces them with int8 equivalents while preserving non-quantizable layers.

### 2. Modify model loading in tts.py
Current loading (line 264):
```python
s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
```
After quantization, either:
- Load float32 checkpoint then quantize in-place (simpler, slower startup)
- Save/load pre-quantized checkpoint with int8 weights + scale factors (faster startup, requires custom serialization)

### 3. Remove weight_norm before quantization
HiFiGAN uses weight_norm on Conv layers. Must call `remove_weight_norm()` before quantizing to convert to standard Conv1d with merged weights.

## Audio-Specific Risks

1. Diffusion error accumulation: Quantization noise compounds across 5-10 Euler steps. Each step's int8 rounding error feeds into the next step's input.

2. ISTFT sensitivity: exp(x) for magnitude reconstruction amplifies any quantization error exponentially. Phase reconstruction via sin/cos is periodic and sensitive to small input changes.

3. Snake activation: The learnable alpha parameter controls the periodic activation curve. Int8 rounding changes the curve shape, altering the audio characteristics.

4. Speaker embedding drift: Quantized speaker encoder may produce slightly different embeddings, breaking voice consistency for cached/pre-computed references.

5. Perceptual sensitivity: Human hearing is highly sensitive to artifacts in the 1-4 kHz range. Quantization-induced noise floor elevation is most noticeable in quiet sections.

## Expected Gains
- Memory: 40-50% reduction on quantizable layers. Overall 1.2-1.4 GB bfloat16 -> 0.6-0.9 GB.
- Latency: Likely neutral or slightly negative. Dequantization overhead on GPU can offset memory bandwidth savings. Weight-only quantization does not reduce compute.
- Throughput: May improve if memory savings allow larger vLLM KV cache or more concurrent requests.

## Validation
- Create a reference audio test set (10-20 samples across languages and speakers)
- Measure baseline MOS (Mean Opinion Score) before quantization
- Quantize incrementally: one component at a time, re-measure MOS
- Compare speaker embedding cosine similarity before/after quantization
- Profile memory usage and latency at each stage
- Test edge cases: quiet audio, fast speech, emotional speech, multiple languages

## Recommendation
This strategy is primarily a memory optimization, not a latency optimization. Pursue it if GPU memory is the bottleneck (e.g., wanting to fit larger vLLM batch sizes or run on smaller GPUs). If latency is the primary concern, torch.compile() and diffusion step reduction will give better returns with less risk. If pursued, start with F0 predictor and HiFiGAN ResBlocks as they are the safest targets, then expand to decoder Conv layers with quality testing at each step.
