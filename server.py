import struct
import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from chatterbox_vllm.tts import ChatterboxTTS
from typing import Optional
import tempfile
import os

app = FastAPI(title="Chatterbox vLLM Streaming TTS")

print("Loading model on cuda...")
model = ChatterboxTTS.from_pretrained()
print("Model loaded.")

SAMPLE_RATE = model.sr  # 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def make_wav_header(sample_rate: int, num_channels: int, bits_per_sample: int) -> bytes:
    """Create a WAV header for streaming (unknown data size)."""
    data_size = 0xFFFFFFFF
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


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
    """Generator that yields PCM bytes (or WAV with header) from streaming TTS."""
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
        audio_np = audio_chunk.squeeze().cpu().numpy()
        pcm_data = (audio_np * 32767).astype(np.int16).tobytes()
        yield pcm_data


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tts")
async def tts_get(
    text: str = Query(..., description="Text to synthesize"),
    language_id: str = Query("en", description="Language code"),
    exaggeration: float = Query(0.5, description="Emotion exaggeration factor"),
    temperature: float = Query(0.8, description="Sampling temperature"),
    chunk_size: int = Query(25, description="Tokens per streaming chunk"),
    diffusion_steps: int = Query(10, description="S3Gen diffusion steps"),
    format: str = Query("wav", description="Output format: 'pcm' or 'wav'"),
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
    format: str = Form("wav"),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4123)
