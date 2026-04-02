import torch
from typing import Optional

ALIGNMENT_LAYER_IDX = 9

# Speech-to-text token ratio heuristics.
# The model generates roughly this many speech tokens per text token.
# These are used to estimate when to suppress/force EOS.
MIN_SPEECH_PER_TEXT = 1.5  # Below this, suppress EOS (too early)
MAX_SPEECH_PER_TEXT = 5    # Above this, force EOS (gibberish)

# Soft EOS boost: progressively increase EOS logit between these ratios.
# This nudges the model toward stopping naturally rather than generating
# junk tokens that get vocoded into audible noise.
SOFT_EOS_START = 3.0       # Start boosting EOS at this ratio
SOFT_EOS_MAX_BOOST = 10.0  # Maximum logit boost added to EOS at MAX_SPEECH_PER_TEXT


class AlignmentState:
    """Per-request alignment tracking state for vLLM.

    Uses token-count heuristics to determine when to force/suppress EOS.
    The text token count is known from prefill, and we estimate expected
    speech token count as a multiple of text tokens.
    """

    def __init__(self, text_token_count: int, eos_idx: int):
        self.text_token_count = text_token_count  # S
        self.eos_idx = eos_idx
        self.step_count = 0

        # Compute expected speech token bounds
        self.min_speech_tokens = text_token_count * MIN_SPEECH_PER_TEXT
        self.max_speech_tokens = text_token_count * MAX_SPEECH_PER_TEXT
        self.soft_eos_start = text_token_count * SOFT_EOS_START

        print(f"[Alignment] Created state: text_tokens={text_token_count}, "
              f"eos_idx={eos_idx}, min_speech={self.min_speech_tokens}, "
              f"soft_eos_start={self.soft_eos_start}, "
              f"max_speech={self.max_speech_tokens}")

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """Modify logits to suppress premature EOS or force EOS after max tokens.

        Args:
            logits: [vocab_size] — logits for this sequence (pre-offset)

        Returns:
            Modified logits tensor
        """
        self.step_count += 1

        if self.step_count < self.min_speech_tokens:
            # Too early — suppress EOS to prevent premature stopping
            logits[self.eos_idx] = -2**15
        elif self.step_count >= self.max_speech_tokens:
            # Too late — force EOS to stop gibberish
            if self.step_count == self.max_speech_tokens:
                print(f"[Alignment] FORCING EOS at step {self.step_count} "
                      f"(max_speech={self.max_speech_tokens})")
            logits = -(2**15) * torch.ones_like(logits)
            logits[self.eos_idx] = 2**15
        elif self.step_count > self.soft_eos_start:
            # Soft boost: linearly increase EOS logit to nudge the model
            # toward stopping before it starts generating junk tokens.
            ramp_range = self.max_speech_tokens - self.soft_eos_start
            if ramp_range > 0:
                progress = (self.step_count - self.soft_eos_start) / ramp_range
                boost = min(progress, 1.0) * SOFT_EOS_MAX_BOOST
                logits[self.eos_idx] += boost

        return logits
