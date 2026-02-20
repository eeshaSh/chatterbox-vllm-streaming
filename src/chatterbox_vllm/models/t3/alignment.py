import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

ALIGNMENT_LAYER_IDX = 9


class AlignmentState:
    """Per-request alignment tracking state for vLLM.

    Mirrors the logic of the original AlignmentStreamAnalyzer from the HuggingFace
    Chatterbox implementation. Instead of capturing attention weights (which vLLM's
    FlashAttention doesn't output), this uses Q and K vectors captured from the
    rotary embedding hook at layer 9 to manually compute alignment scores.
    """

    def __init__(self, text_k: torch.Tensor, text_token_count: int,
                 num_heads: int, head_dim: int, eos_idx: int):
        self.text_k = text_k              # [S, num_heads * head_dim] on GPU
        self.text_token_count = text_token_count  # S
        self.num_heads = num_heads         # 16
        self.head_dim = head_dim           # 64
        self.eos_idx = eos_idx             # stop_speech_token index in speech vocab

        # Precompute reshaped K for efficiency: [num_heads, S, head_dim]
        self._k_reshaped = text_k.view(text_token_count, num_heads, head_dim).permute(1, 0, 2).contiguous()
        self._scale = head_dim ** 0.5

        # Tracking state (mirrors original AlignmentStreamAnalyzer)
        self.alignment = torch.zeros(0, text_token_count)  # [T, S] accumulated on CPU
        self.curr_frame_pos = 0
        self.text_position = 0
        self.started = False
        self.started_at: Optional[int] = None
        self.complete = False
        self.completed_at: Optional[int] = None

    def compute_alignment_scores(self, q: torch.Tensor) -> torch.Tensor:
        """Compute attention scores from decode Q against stored text K.

        Args:
            q: [num_heads * head_dim] — Q for one decode token (on GPU)

        Returns:
            [S] — alignment scores over text tokens (on CPU)
        """
        # Reshape Q to [num_heads, 1, head_dim]
        q = q.view(self.num_heads, 1, self.head_dim)

        # Compute attention: softmax(Q @ K^T / sqrt(d))
        # q: [num_heads, 1, head_dim], k_reshaped: [num_heads, S, head_dim]
        scores = torch.bmm(q, self._k_reshaped.transpose(1, 2))  # [num_heads, 1, S]
        scores = scores / self._scale
        scores = torch.softmax(scores, dim=-1)

        # Average over heads: [S]
        return scores.mean(dim=0).squeeze(0).cpu()

    def step(self, q: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Run one alignment analysis step.

        Mirrors the original AlignmentStreamAnalyzer.step() logic:
        - Tracks text position via argmax of attention over text tokens
        - Detects completion (position >= S-3)
        - Detects long_tail (last tokens accumulate too many frames)
        - Detects repetition (earlier tokens get activation after completion)
        - Forces/suppresses EOS accordingly

        Args:
            q: [num_heads * head_dim] — Q for current decode token
            logits: [vocab_size] — logits for this sequence (pre-offset)

        Returns:
            Modified logits tensor
        """
        alignment_scores = self.compute_alignment_scores(q)  # [S] on CPU
        A_chunk = alignment_scores.unsqueeze(0)  # [1, S]

        S = self.text_token_count

        # Monotonic masking — zero out positions after current frame position
        if self.curr_frame_pos + 1 < S:
            A_chunk[:, self.curr_frame_pos + 1:] = 0

        # Accumulate alignment matrix
        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)
        A = self.alignment
        T = A.shape[0]

        # Update position via argmax
        cur_text_posn = A_chunk[-1].argmax().item()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
        if not discontinuity:
            self.text_position = cur_text_posn

        # False start detection — hallucinations at the beginning show up as
        # activations at the bottom-right of the attention maps
        if T >= 2:
            false_start = (not self.started) and (
                A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5
            )
        else:
            false_start = not self.started
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Completion detection — have we reached the end of text tokens?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # Long tail detection — last 3 text tokens accumulating too many frames
        long_tail = False
        if self.complete and self.completed_at is not None:
            long_tail = A[self.completed_at:, -3:].sum(dim=0).max() >= 10

        # Repetition detection — activations in earlier tokens after completion
        repetition = False
        if self.complete and self.completed_at is not None and S > 5:
            repetition = A[self.completed_at:, :-5].max(dim=1).values.sum() > 5

        # Modify logits to force/suppress EOS
        if long_tail or repetition:
            logger.warning(f"Forcing EOS token: long_tail={long_tail}, repetition={repetition}")
            logits = -(2**15) * torch.ones_like(logits)
            logits[self.eos_idx] = 2**15
        elif cur_text_posn < S - 3:
            # Suppress EOS to prevent early termination
            logits[self.eos_idx] = -2**15

        self.curr_frame_pos += 1
        return logits
