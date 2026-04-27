import torch
from typing import Optional

ALIGNMENT_LAYER_IDX = 9

# Speech-to-text token ratio heuristics.
# The model generates roughly this many speech tokens per text token.
# These are used to estimate when to suppress/force EOS.
MIN_SPEECH_PER_TEXT = 1.5  # Below this, suppress EOS (too early)
MAX_SPEECH_PER_TEXT = 6    # Above this, force EOS (gibberish)

# EOS logit sustained-high detection.
# If the EOS logit stays above this threshold for this many consecutive steps,
# the model wants to stop but keeps losing in the softmax — force EOS.
EOS_LOGIT_THRESHOLD = 3.0
EOS_LOGIT_SUSTAINED_STEPS = 10


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
        self.eos_high_count = 0  # consecutive steps where eos_logit > threshold

        # Compute expected speech token bounds
        self.min_speech_tokens = text_token_count * MIN_SPEECH_PER_TEXT
        self.max_speech_tokens = text_token_count * MAX_SPEECH_PER_TEXT

        print(f"[Alignment] Created state: text_tokens={text_token_count}, "
              f"eos_idx={eos_idx}, min_speech={self.min_speech_tokens}, "
              f"max_speech={self.max_speech_tokens}")

    def step(self, logits: torch.Tensor) -> torch.Tensor:
        """Modify logits to suppress premature EOS or force EOS after max tokens.

        Args:
            logits: [vocab_size] — logits for this sequence (pre-offset)

        Returns:
            Modified logits tensor
        """
        self.step_count += 1

        # Log every step once we're past min_speech_tokens
        if self.step_count > self.min_speech_tokens:
            print(f"[Alignment] step={self.step_count}, min={self.min_speech_tokens}, "
                  f"max={self.max_speech_tokens}, eos_logit={logits[self.eos_idx].item():.2f}")

        if self.step_count < self.min_speech_tokens:
            # Too early — suppress EOS to prevent premature stopping
            if self.step_count <= 3:
                print(f"[Alignment] step={self.step_count} — suppressing EOS (min={self.min_speech_tokens})")
            logits[self.eos_idx] = -2**15
        elif self.step_count >= self.max_speech_tokens:
            # Too late — force EOS to stop gibberish
            if self.step_count == self.max_speech_tokens:
                print(f"[Alignment] FORCING EOS at step {self.step_count} "
                      f"(max_speech={self.max_speech_tokens})")
            logits = -(2**15) * torch.ones_like(logits)
            logits[self.eos_idx] = 2**15
        else:
            # Middle zone — check if EOS logit is sustained-high
            eos_logit = logits[self.eos_idx].item()
            if eos_logit >= EOS_LOGIT_THRESHOLD:
                self.eos_high_count += 1
            else:
                self.eos_high_count = 0

            if self.eos_high_count >= EOS_LOGIT_SUSTAINED_STEPS:
                print(f"[Alignment] SUSTAINED EOS DETECTED: eos_logit >= {EOS_LOGIT_THRESHOLD} "
                      f"for {self.eos_high_count} consecutive steps at step {self.step_count}. "
                      f"Forcing EOS.")
                logits = -(2**15) * torch.ones_like(logits)
                logits[self.eos_idx] = 2**15

        return logits
