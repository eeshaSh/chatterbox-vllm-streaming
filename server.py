import io
import struct
import torch
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from chatterbox_vllm.tts import ChatterboxTTS
from typing import Optional
import tempfile
import os

app = FastAPI()

print("Loading model on cuda...")
model = ChatterboxTTS.from_pretrained()
print("Model loaded.")

SAMPLE_RATE = model.sr  # 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def make_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int) -> bytes:
    """Create a WAV header with unknown data size (streamed)."""
    # Use 0xFFFFFFFF for data size to indicate unknown/streaming
    data_size = 0xFFFFFFFF
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size,  # will be wrong but that's fine for streaming
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def audio_stream(
    text: str,
    audio_prompt_path: Optional[str] = None,
    language_id: str = "en",
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    chunk_size: int = 25,
    diffusion_steps: int = 10,
    output_format: str = "pcm",
):
    """Generator that yields raw PCM (or WAV with header) bytes from generate_stream."""
    if output_format == "wav":
        yield make_wav_header(SAMPLE_RATE, NUM_CHANNELS, SAMPLE_WIDTH * 8)

    for audio_chunk, metrics in model.generate_stream(
        text=text,
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        exaggeration=exaggeration,
        temperature=temperature,
        chunk_size=chunk_size,
        diffusion_steps=diffusion_steps,
    ):
        # audio_chunk: torch.Tensor of shape (1, num_samples), float32 in [-1, 1]
        audio_np = audio_chunk.squeeze().cpu().numpy()
        # Convert float32 -> int16 PCM
        pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
        yield pcm_data


@app.get("/tts")
async def tts(
    text: str = Query(..., description="Text to synthesize"),
    language_id: str = Query("en", description="Language code (e.g. 'en', 'zh', 'fr')"),
    exaggeration: float = Query(0.5, description="Emotion exaggeration factor"),
    temperature: float = Query(0.8, description="Sampling temperature"),
    chunk_size: int = Query(25, description="Tokens per chunk"),
    diffusion_steps: int = Query(10, description="Number of diffusion steps for S3Gen"),
    format: str = Query("pcm", description="Output format: 'pcm' (raw int16) or 'wav'"),
):
    """Stream TTS audio as raw PCM or WAV."""
    content_type = "audio/wav" if format == "wav" else "audio/L16;rate=24000;channels=1"

    return StreamingResponse(
        audio_stream(
            text=text,
            language_id=language_id,
            exaggeration=exaggeration,
            temperature=temperature,
            chunk_size=chunk_size,
            diffusion_steps=diffusion_steps,
            output_format=format,
        ),
        media_type=content_type,
    )


@app.post("/tts")
async def tts_post(
    text: str = Form(..., description="Text to synthesize"),
    audio_prompt: Optional[UploadFile] = File(None, description="Reference audio for voice cloning"),
    language_id: str = Form("en"),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    chunk_size: int = Form(25),
    diffusion_steps: int = Form(10),
    format: str = Form("pcm"),
):
    """Stream TTS audio with optional voice cloning via file upload."""
    audio_prompt_path = None
    tmp_file = None

    if audio_prompt is not None:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_file.write(await audio_prompt.read())
        tmp_file.close()
        audio_prompt_path = tmp_file.name

    def stream_and_cleanup():
        try:
            yield from audio_stream(
                text=text,
                audio_prompt_path=audio_prompt_path,
                language_id=language_id,
                exaggeration=exaggeration,
                temperature=temperature,
                chunk_size=chunk_size,
                diffusion_steps=diffusion_steps,
                output_format=format,
            )
        finally:
            if tmp_file is not None:
                os.unlink(tmp_file.name)

    content_type = "audio/wav" if format == "wav" else "audio/L16;rate=24000;channels=1"

    return StreamingResponse(
        stream_and_cleanup(),
        media_type=content_type,
    )


def make_wav_bytes(audio_np: np.ndarray, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a complete WAV file in memory from a numpy array."""
    pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
    data_size = len(pcm_data)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data


@app.post("/tts_batched")
async def tts_batched(
    text: str = Form(..., description="Text to synthesize"),
    audio_prompt: Optional[UploadFile] = File(None, description="Reference audio for voice cloning"),
    language_id: str = Form("en"),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    diffusion_steps: int = Form(10),
    format: str = Form("wav"),
):
    """Non-streaming batched TTS using vLLM's LLM.generate(). Returns complete audio."""
    audio_prompt_path = None
    tmp_file = None

    try:
        if audio_prompt is not None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(await audio_prompt.read())
            tmp_file.close()
            audio_prompt_path = tmp_file.name

        results = model.generate(
            prompts=text,
            audio_prompt_path=audio_prompt_path,
            language_id=language_id,
            exaggeration=exaggeration,
            temperature=temperature,
            diffusion_steps=diffusion_steps,
        )

        wav = results[0]  # First result
        audio_np = wav.squeeze().cpu().numpy()

        if format == "wav":
            wav_bytes = make_wav_bytes(audio_np, SAMPLE_RATE)
            return Response(content=wav_bytes, media_type="audio/wav")
        else:
            pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
            return Response(content=pcm_data, media_type="audio/L16;rate=24000;channels=1")
    finally:
        if tmp_file is not None:
            os.unlink(tmp_file.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
