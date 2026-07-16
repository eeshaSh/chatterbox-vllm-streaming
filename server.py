import struct
import time
import numpy as np
from fastapi import FastAPI, Query, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.metrics import (
    ACTIVE_REQUESTS,
    REQUEST_COUNT,
    REQUEST_DURATION,
    TTFB_HISTOGRAM,
    register_batcher_collector,
)
from typing import Optional
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(title="Chatterbox vLLM Streaming TTS")

# Map language codes to voice clone wav files.
# Languages not in this map (e.g. "en") will use the default model voice.
VOICE_CLONE_DIR = Path(__file__).parent / "voice_clone_wavs"
VOICE_CLONE_MAP: dict[str, Path] = {
    # "tr": VOICE_CLONE_DIR / "turkish_voice_clone_male.wav",
    # "no": VOICE_CLONE_DIR / "norwegian_voice_clone_female_2.wav",
    # "tr": VOICE_CLONE_DIR / "real_person_turkish_clone_audio.wav",
    "tr": VOICE_CLONE_DIR / "slower_turkish_audio.wav",
    "no": VOICE_CLONE_DIR / "real_person_norwegian_clone_audio.wav",
    "nl": VOICE_CLONE_DIR / "dutch_voice_clone.wav",
    "da": VOICE_CLONE_DIR / "danish_voice_clone_tesla8.wav",
    "ar": VOICE_CLONE_DIR / "arabic_uae_voice_clone.wav",
    "ar-AE": VOICE_CLONE_DIR / "arabic_uae_voice_clone.wav",
    "ar-SA": VOICE_CLONE_DIR / "arabic_saudi_voice_clone.wav",
    "ar-JO": VOICE_CLONE_DIR / "jordanian_arabic_omar.wav",
    "sv": VOICE_CLONE_DIR / "spiller_swedish_slow.wav",
}
# Snapshot of the defaults so we can tell them apart from runtime additions.
_DEFAULT_CLONE_KEYS: set[str] = set(VOICE_CLONE_MAP)

print("Loading multilingual model on cuda...")
model = ChatterboxTTS.from_pretrained_multilingual()
print("Model loaded.")

# Prometheus /metrics endpoint
register_batcher_collector(lambda: model.vocoder_batcher)

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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


async def audio_stream(
    text: str,
    language_id: str = "en",
    exaggeration: float = 0.5,
    temperature: float = 0.5,
    chunk_size: int = 15,
    diffusion_steps: int = 5,
    output_format: str = "pcm",
):
    """Async generator that yields PCM bytes (or WAV with header) from streaming TTS."""
    request_start = time.time()
    first_audio_sent = False
    chunk_count = 0
    status = "success"

    ACTIVE_REQUESTS.inc()

    # Resolve voice clone file from the full language tag (e.g. "ar-AE"),
    # but only pass the prefix (e.g. "ar") to the model since it only
    # supports the base language codes.
    audio_prompt_path = None
    voice_file = VOICE_CLONE_MAP.get(language_id)
    if voice_file is not None:
        audio_prompt_path = str(voice_file)

    model_language_id = language_id.split("-")[0]

    try:
        if output_format == "wav":
            yield make_wav_header(SAMPLE_RATE, NUM_CHANNELS, SAMPLE_WIDTH * 8)

        async for audio_chunk, metrics in model.generate_stream(
            text=text,
            audio_prompt_path=audio_prompt_path,
            language_id=model_language_id,
            exaggeration=exaggeration,
            temperature=temperature,
            chunk_size=chunk_size,
            diffusion_steps=diffusion_steps,
        ):
            audio_np = audio_chunk.squeeze().cpu().numpy()
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            pcm_data = (audio_np * np.iinfo(np.int16).max).astype("<i2", copy=False).tobytes()
            if not first_audio_sent:
                ttfb = time.time() - request_start
                print(f"[Server] TTFB (request → first audio byte): {ttfb:.3f}s")
                TTFB_HISTOGRAM.observe(ttfb)
                first_audio_sent = True
            chunk_count += 1
            yield pcm_data
    except Exception:
        status = "error"
        raise
    finally:
        total_time = time.time() - request_start
        print(f"[Server] Request complete: {chunk_count} chunks in {total_time:.2f}s")
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(status=status).inc()
        REQUEST_DURATION.observe(total_time)


class SpeechRequest(BaseModel):
    input: str
    language_id: str = "en"
    exaggeration: float = 0.5
    temperature: float = 0.5
    chunk_size: int = 15
    diffusion_steps: int = 5


@app.post("/audio/speech")
async def audio_speech(request: SpeechRequest):
    """Speech endpoint. Streams raw PCM audio. Voice cloning is automatic based on language_id."""
    return StreamingResponse(
        audio_stream(
            text=request.input,
            language_id=request.language_id,
            exaggeration=request.exaggeration,
            temperature=request.temperature,
            chunk_size=request.chunk_size,
            diffusion_steps=request.diffusion_steps,
            output_format="pcm",
        ),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Voice-clone management (runtime) ────────────────────────────────


@app.get("/voice-clones")
async def list_voice_clones():
    """Return the current language_id → wav file mapping."""
    return {
        lang: {
            "file": path.name,
            "default": lang in _DEFAULT_CLONE_KEYS,
        }
        for lang, path in sorted(VOICE_CLONE_MAP.items())
    }


@app.post("/voice-clones")
async def add_voice_clone(
    language_id: str = Form(..., description="Language code to map, e.g. 'ar-JO'"),
    file: UploadFile = File(..., description="WAV file for the voice clone"),
):
    """Upload a WAV file and register it as the voice clone for a language_id.

    Overwrites any existing mapping for the same language_id.
    The mapping lasts until the server is restarted.
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only .wav files are accepted"},
        )

    dest = VOICE_CLONE_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    VOICE_CLONE_MAP[language_id] = dest
    action = "replaced" if language_id in _DEFAULT_CLONE_KEYS else "added"
    print(f"[VoiceClones] {action} clone for '{language_id}' → {dest.name}")

    return {
        "language_id": language_id,
        "file": dest.name,
        "action": action,
        "clones": {k: v.name for k, v in sorted(VOICE_CLONE_MAP.items())},
    }


@app.delete("/voice-clones/{language_id}")
async def remove_voice_clone(language_id: str):
    """Remove a voice-clone mapping. The language will fall back to the default model voice."""
    removed = VOICE_CLONE_MAP.pop(language_id, None)
    if removed is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No voice clone registered for '{language_id}'"},
        )
    print(f"[VoiceClones] removed clone for '{language_id}' (was {removed.name})")
    return {
        "language_id": language_id,
        "removed_file": removed.name,
        "clones": {k: v.name for k, v in sorted(VOICE_CLONE_MAP.items())},
    }


@app.get("/tts")
async def tts_get(
    text: str = Query(..., description="Text to synthesize"),
    language_id: str = Query("en", description="Language code"),
    exaggeration: float = Query(0.5, description="Emotion exaggeration factor"),
    temperature: float = Query(0.5, description="Sampling temperature"),
    chunk_size: int = Query(15, description="Tokens per streaming chunk"),
    diffusion_steps: int = Query(5, description="S3Gen diffusion steps"),
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
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/tts")
async def tts_post(
    text: str = Form(..., description="Text to synthesize"),
    language_id: str = Form("en"),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.5),
    chunk_size: int = Form(15),
    diffusion_steps: int = Form(5),
    format: str = Form("wav"),
):
    """Stream TTS audio. Voice cloning is automatic based on language_id."""
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
        headers={
            "Cache-Control": "no-cache",
            "Transfer-Encoding": "chunked",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4123)
