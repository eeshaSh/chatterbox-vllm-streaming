import torch
from collections import deque
from typing import Optional

ALIGNMENT_LAYER_IDX = 9

# Speech-to-text token ratio heuristics.
# The model generates roughly this many speech tokens per text token.
# These are used to estimate when to suppress/force EOS.
MIN_SPEECH_PER_TEXT = 1.5  # Below this, suppress EOS (too early)
MAX_SPEECH_PER_TEXT = 6    # Above this, force EOS (gibberish)

# EOS logit drift detection.
# Track running average of eos_logit across all middle-zone steps.
# When a recent window average exceeds the running average by this margin,
# the model has shifted into "wants to stop" mode — force EOS.
EOS_DRIFT_WINDOW = 15       # recent window size to compare against running avg
EOS_DRIFT_MARGIN = 1.5      # how much higher recent avg must be vs running avg
EOS_DRIFT_MIN_RECENT = 1.0  # recent avg must also be at least this high (absolute floor)


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

        # EOS logit drift detection state
        self.eos_logit_sum = 0.0       # sum of all middle-zone eos_logits
        self.eos_logit_count = 0       # number of middle-zone steps
        self.eos_recent = deque(maxlen=EOS_DRIFT_WINDOW)  # recent window

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
        else:
            # Middle zone — track EOS logit drift
            eos_logit = logits[self.eos_idx].item()
            self.eos_logit_sum += eos_logit
            self.eos_logit_count += 1
            self.eos_recent.append(eos_logit)

            if len(self.eos_recent) >= EOS_DRIFT_WINDOW:
                running_avg = self.eos_logit_sum / self.eos_logit_count
                recent_avg = sum(self.eos_recent) / len(self.eos_recent)

                if recent_avg >= EOS_DRIFT_MIN_RECENT and recent_avg >= running_avg + EOS_DRIFT_MARGIN:
                    print(f"[Alignment] EOS DRIFT DETECTED at step {self.step_count}: "
                          f"recent_avg={recent_avg:.2f} vs running_avg={running_avg:.2f} "
                          f"(margin={EOS_DRIFT_MARGIN}). Forcing EOS.")
                    logits = -(2**15) * torch.ones_like(logits)
                    logits[self.eos_idx] = 2**15

        return logits
