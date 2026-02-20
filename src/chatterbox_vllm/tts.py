from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, AsyncGenerator
import asyncio
import time
import uuid

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from functools import lru_cache

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES

REPO_ID = "ResembleAI/chatterbox"

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation."""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class VocoderBatcher:
    """Collects vocoding requests from concurrent streams and batches them.

    Instead of each streaming request calling S3Gen individually (serialized,
    ~280ms each), requests submit to a shared queue. A background worker
    collects pending requests into a batch and runs a single batched S3Gen
    call, amortizing GPU kernel launch overhead across all items.

    With 10 concurrent requests and batch_size=10, this reduces 10 × 280ms
    = 2.8s of serialized vocoding to a single ~300ms batched call.
    """

    def __init__(self, s3gen: S3Gen, max_batch_size: int = 32, max_wait_ms: float = 50):
        self.s3gen = s3gen
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue = None  # Initialized lazily per event loop
        self._worker_task: asyncio.Task = None

    def _ensure_started(self):
        """Start the batch worker if not already running."""
        loop = asyncio.get_event_loop()
        if self._queue is None or self._worker_task is None or self._worker_task.done():
            self._queue = asyncio.Queue()
            self._worker_task = loop.create_task(self._batch_worker())

    async def vocode(
        self,
        speech_tokens: torch.Tensor,
        ref_dict: dict,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Submit a vocoding request and wait for the batched result."""
        self._ensure_started()
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((speech_tokens, ref_dict, n_timesteps, future))
        return await future

    async def _batch_worker(self):
        """Background worker that collects and batches vocoder requests."""
        while True:
            batch = []
            try:
                # Wait for at least one request
                item = await self._queue.get()
                batch.append(item)

                # Collect more requests up to max_batch_size or timeout
                deadline = time.monotonic() + self.max_wait_ms / 1000
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # All items in the batch must share the same ref_dict and n_timesteps
                # (they do in practice since concurrent requests typically use the same voice)
                tokens_list = [item[0] for item in batch]
                ref_dict = batch[0][1]
                n_timesteps = batch[0][2]

                if len(batch) > 1:
                    print(f"[VocoderBatcher] Batching {len(batch)} requests together")

                # Run batched S3Gen inference with autocast for Tensor Core utilization
                try:
                    with torch.autocast("cuda", dtype=torch.float16):
                        results = self.s3gen.batch_inference(
                            speech_tokens_list=tokens_list,
                            ref_dict=ref_dict,
                            n_timesteps=n_timesteps,
                        )
                    for (_, _, _, future), result in zip(batch, results):
                        if not future.done():
                            future.set_result(result)
                except Exception as e:
                    for _, _, _, future in batch:
                        if not future.done():
                            future.set_exception(e)

            except Exception as e:
                # Don't let the worker die from unexpected errors
                print(f"[VocoderBatcher] Worker error: {e}")
                for _, _, _, future in batch:
                    if not future.done():
                        future.set_exception(e)


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 async_engine: AsyncLLMEngine, t3_config: T3Config, t3_cond_enc: T3CondEnc,
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals,
                 variant: str = "english"):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.async_engine = async_engine
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.variant = variant

        # Batched vocoder for concurrent request throughput
        self.vocoder_batcher = VocoderBatcher(s3gen)

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   variant: str = "english",

                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / ("t3_cfg.safetensors" if variant == "english" else "t3_mtl23ls_v2.safetensors"))

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
        # This rough heuristic gives 1.55GB for the model weights plus 128KB per token.
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        base_vllm_kwargs = {
            "model": "./t3-model" if variant == "english" else "./t3-model-multilingual",
            "task": "generate",
            "tokenizer": "EnTokenizer" if variant == "english" else "MtlTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
            "dtype": "half",
        }

        engine_args = AsyncEngineArgs(**{**base_vllm_kwargs, **kwargs})
        async_engine = AsyncLLMEngine.from_engine_args(engine_args)

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()


        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            async_engine=async_engine, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    def from_pretrained_multilingual(cls,
                                    repo_id: str = REPO_ID,
                                    revision: str = "05e904af2b5c7f8e482687a9d7336c5c824467d9",
                                    *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_mtl23ls_v2.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)
    
    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    async def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.8,
        repetition_penalty=2.0,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return await self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            language_id=language_id,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    async def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 5,

        # From original Chatterbox HF generation args
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(max_tokens, self.max_model_len),
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )

        with torch.inference_mode():
            start_time = time.time()

            # Generate all prompts via async engine
            batch_results = []
            for text in prompts:
                prompt = {
                    "prompt": text,
                    "multi_modal_data": {
                        "conditionals": [cond_emb],
                    },
                }
                request_id = str(uuid.uuid4())
                final_output = None
                async for output in self.async_engine.generate(
                    prompt, sampling_params, request_id
                ):
                    final_output = output
                if final_output is not None:
                    batch_results.append(final_output)

            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    with torch.autocast("cuda", dtype=torch.float16):
                        wav, _ = self.s3gen.inference(
                            speech_tokens=speech_tokens,
                            ref_dict=s3gen_ref,
                            n_timesteps=diffusion_steps,
                        )
                    results.append(wav.cpu())
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results
        
    def _process_token_buffer(
        self,
        new_tokens: torch.Tensor,
        all_tokens_so_far: torch.Tensor,
        context_window: int,
        s3gen_ref: dict,
        start_time: float,
        metrics: StreamingMetrics,
        fade_duration: float = 0.02,
        diffusion_steps: int = 5,
    ) -> Tuple[Optional[torch.Tensor], float, bool]:
        """Convert a chunk of speech tokens to audio using context windowing for smooth boundaries."""
        # Include previous tokens as context so S3Gen can produce coherent audio at boundaries
        if len(all_tokens_so_far) > 0:
            context_tokens = all_tokens_so_far[-context_window:]
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.target_device)
        clean_tokens = clean_tokens[clean_tokens < 6561]
        if len(clean_tokens) == 0:
            return None, 0.0, False

        with torch.autocast("cuda", dtype=torch.float16):
            wav, _ = self.s3gen.inference(
                speech_tokens=clean_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=diffusion_steps,
            )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop out the context portion — only keep audio for the new tokens
        if context_length > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Linear fade-in to smooth chunk boundaries
        fade_samples = min(int(fade_duration * self.sr), len(audio_chunk))
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self.sr
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            print(f"[Streaming] Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    async def _process_token_buffer_batched(
        self,
        new_tokens: torch.Tensor,
        all_tokens_so_far: torch.Tensor,
        context_window: int,
        s3gen_ref: dict,
        start_time: float,
        metrics: StreamingMetrics,
        fade_duration: float = 0.02,
        diffusion_steps: int = 5,
    ) -> Tuple[Optional[torch.Tensor], float, bool]:
        """Async version of _process_token_buffer that routes S3Gen through the VocoderBatcher."""
        # Token prep (same as sync version)
        if len(all_tokens_so_far) > 0:
            context_tokens = all_tokens_so_far[-context_window:]
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.target_device)
        clean_tokens = clean_tokens[clean_tokens < 6561]
        if len(clean_tokens) == 0:
            return None, 0.0, False

        # Submit to batcher — suspends this coroutine, freeing event loop
        # for vLLM token generation and other requests' vocoding submissions
        wav = await self.vocoder_batcher.vocode(
            speech_tokens=clean_tokens,
            ref_dict=s3gen_ref,
            n_timesteps=diffusion_steps,
        )
        wav = wav.detach().cpu().numpy()

        # Post-processing (same as sync version)
        if context_length > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        fade_samples = min(int(fade_duration * self.sr), len(audio_chunk))
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        audio_duration = len(audio_chunk) / self.sr
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            print(f"[Streaming] Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        chunk_size: int = 25,
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 5,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 2.0,
    ) -> AsyncGenerator[Tuple[torch.Tensor, StreamingMetrics], None]:
        """Stream audio chunks as they are generated.

        Uses vLLM's AsyncLLMEngine to get tokens incrementally,
        buffers them into chunks, and runs S3Gen on each chunk with a context
        window for smooth boundaries.

        Yields (audio_chunk, metrics) tuples where audio_chunk is a tensor of
        shape (1, num_samples) and metrics tracks timing information.
        """
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        async for chunk in self.generate_stream_with_conds(
            text=text,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            language_id=language_id,
            exaggeration=exaggeration,
            temperature=temperature,
            chunk_size=chunk_size,
            context_window=context_window,
            fade_duration=fade_duration,
            diffusion_steps=diffusion_steps,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        ):
            yield chunk

    async def generate_stream_with_conds(
        self,
        text: str,
        s3gen_ref: dict,
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        chunk_size: int = 25,
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 5,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 2.0,
    ) -> AsyncGenerator[Tuple[torch.Tensor, StreamingMetrics], None]:
        """Stream audio chunks using pre-computed conditioning tensors.

        Core streaming implementation. Uses AsyncLLMEngine.generate() to get
        tokens incrementally, accumulates speech tokens into a buffer, and
        yields audio chunks as each buffer fills up.
        """
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        text = "[START]" + punc_norm(text) + "[STOP]"
        if self.variant == "multilingual":
            text = f"<{language_id.lower()}>{text}"

        prompt = {
            "prompt": text,
            "multi_modal_data": {
                "conditionals": [cond_emb],
            },
        }

        stop_token_id = self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET
        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[stop_token_id],
            max_tokens=min(max_tokens, self.max_model_len),
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )
        sampling_params.output_kind = RequestOutputKind.DELTA

        request_id = str(uuid.uuid4())

        start_time = time.time()
        metrics = StreamingMetrics()
        total_audio_length = 0.0

        token_buffer: list[int] = []
        all_tokens_processed = torch.tensor([], dtype=torch.long, device=self.target_device)

        with torch.inference_mode():
            async for output in self.async_engine.generate(
                prompt, sampling_params, request_id
            ):
                for completion in output.outputs:
                    token_buffer.extend(completion.token_ids)

                should_process = len(token_buffer) >= chunk_size or output.finished

                if should_process and len(token_buffer) > 0:
                    new_speech_tokens = torch.tensor(
                        [t - SPEECH_TOKEN_OFFSET for t in token_buffer],
                        device=self.target_device,
                    )
                    new_speech_tokens = drop_invalid_tokens(new_speech_tokens)
                    new_speech_tokens = new_speech_tokens[new_speech_tokens < 6561]

                    if len(new_speech_tokens) > 0:
                        audio_tensor, audio_duration, success = await self._process_token_buffer_batched(
                            new_speech_tokens,
                            all_tokens_processed,
                            context_window,
                            s3gen_ref,
                            start_time,
                            metrics,
                            fade_duration,
                            diffusion_steps,
                        )

                        if success:
                            total_audio_length += audio_duration
                            yield audio_tensor, metrics

                        all_tokens_processed = torch.cat([all_tokens_processed, new_speech_tokens])

                    token_buffer = []

            torch.cuda.empty_cache()

        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            print(f"[Streaming] Total generation time: {metrics.total_generation_time:.3f}s")
            print(f"[Streaming] Total audio duration: {metrics.total_audio_duration:.3f}s")
            print(f"[Streaming] RTF: {metrics.rtf:.3f}")
            print(f"[Streaming] Chunks yielded: {metrics.chunk_count}")

    def shutdown(self):
        del self.async_engine
        torch.cuda.empty_cache()
