import torch
from typing import Optional

ALIGNMENT_LAYER_IDX = 9

# Speech-to-text token ratio heuristics.
# The model generates roughly this many speech tokens per text token.
# These are used to estimate when to suppress/force EOS.
MIN_SPEECH_PER_TEXT = 1.5  # Below this, suppress EOS (too early)
MAX_SPEECH_PER_TEXT = 4    # Above this, force EOS (gibberish)


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

        return logits
