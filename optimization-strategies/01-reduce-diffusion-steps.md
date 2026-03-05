# Strategy 1: Reduce S3Gen Diffusion Steps

## Overview

The S3Gen vocoder uses Conditional Flow Matching (CFM) with an Euler ODE solver to convert speech tokens into mel-spectrograms. Each diffusion step runs the full ConditionalDecoder estimator network (U-Net with transformers). Reducing the number of steps directly reduces compute proportionally since the estimator is the dominant cost.

## Current Implementation

- Default: 5 steps in API layer (tts.py, server.py), 10 in model code (s3gen.py)
- Solver: Euler method with cosine time schedule
- Classifier-Free Guidance (CFG): rate=0.7, doubles effective batch (conditioned + unconditioned)
- CFG formula: (1.7 * conditioned - 0.7 * unconditioned)
- Temperature-scaled Gaussian noise initialization

## Key Files

- `src/chatterbox_vllm/models/s3gen/flow_matching.py` -- solve_euler() loop (lines 82-137), the core diffusion iteration
- `src/chatterbox_vllm/models/s3gen/s3gen.py` -- inference/batch_inference entry points (defaults to 10)
- `src/chatterbox_vllm/tts.py` -- API-level defaults (5 steps at lines 408, 505, 566, 674)
- `server.py` -- HTTP endpoint defaults (5 steps at lines 60, 102, 140, 172)
- `src/chatterbox_vllm/models/s3gen/configs.py` -- hardcoded solver='euler', t_scheduler='cosine', inference_cfg_rate=0.7

## How the Diffusion Loop Works

1. Initialize z = Gaussian noise * temperature
2. Create time span t_span = linspace(0, 1, n_timesteps+1) with cosine schedule: 1 - cos(t * pi/2)
3. For each step:
   a. Pack conditioned + unconditioned inputs (doubles batch to 2*B)
   b. Run ConditionalDecoder estimator to predict velocity field dphi_dt
   c. Split results, apply CFG: (1+0.7)*cond - 0.7*uncond
   d. Euler integration: x = x + dt * dphi_dt
4. Return final x as mel-spectrogram

## Changes Required

### Minimal (no retraining)

- Adjust default values in tts.py (4 locations) and server.py (4 locations)
- Already fully parameterized -- just changing integer defaults

### Medium effort (no retraining, better quality at low steps)

- Implement DPM++ or Heun's method as alternative solvers in flow_matching.py
- Heun's method: 2nd order, uses two estimator evaluations per step but halves the number of steps needed
- DPM++: proven in image diffusion to achieve equivalent quality at 2-3x fewer steps
- Would require a new solver function alongside solve_euler()

### Optional CFG tuning

- Make inference_cfg_rate adaptive based on step count
- At fewer steps, consider reducing CFG to avoid amplifying discretization errors
- Currently hardcoded at 0.7 in configs.py

## Quality vs Speed Tradeoff

- Euler method error: O(1/n_timesteps) total accumulated error
- 10 steps: baseline quality
- 5 steps: "nearly indistinguishable" per original developers (but later reverted due to user feedback)
- 3 steps: intelligible but noticeably degraded
- 2 steps: significant artifacts, not recommended
- With DPM++ solver: 3-4 steps may match Euler at 8-10 steps

## Risks

- Audio quality degradation (prosody flatness, speaker timbre loss, spectral artifacts)
- CFG amplifies errors at low step counts since it pushes further from natural distribution
- Project history shows the team already tried 5 steps and reverted to 10 once
- Each step costs one full ConditionalDecoder forward pass (the dominant compute)

## Expected Gains

- Linear speedup: halving steps halves S3Gen flow inference time
- No memory savings (peak memory unchanged)
- If flow inference is 60% of S3Gen time, going from 10 to 5 steps saves ~30% of total S3Gen time

## Validation

- A/B test at step counts: 10, 8, 5, 4, 3
- Use MOS (Mean Opinion Score) with native speakers
- Measure: prosody quality, speaker identity consistency, artifact presence
- Test across languages if multilingual support matters
- Profile inference time at each step count to confirm linear scaling

## Recommendation

Start by profiling to confirm what fraction of S3Gen time is spent in flow inference vs HiFiGAN. If flow dominates, reducing steps from the current default of 5 to 3-4 with a DPM++ solver is the highest-value change. If already at 5 and quality is acceptable, this is already partially optimized.
