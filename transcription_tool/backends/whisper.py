"""OpenAI Whisper backend."""
from typing import Any
from returns.result import Result, Success, safe
from toolz import curry
from loguru import logger

from models import AudioData


@safe
def load_model(model_name: str) -> Any:
    """Load whisper model."""
    import whisper
    logger.info(f"[whisper] Loading model: {model_name}")
    return whisper.load_model(model_name)


@curry
def transcribe_segment(
    model: Any,
    audio: AudioData,
    start: float,
    end: float,
) -> Result[str, str]:
    """Transcribe audio segment."""
    try:
        audio_np = audio.waveform.squeeze().numpy()
        result = model.transcribe(audio_np)
        return Success(result["text"])
    except Exception as e:
        return Success("")


class _Backend:
    load_model = staticmethod(load_model)
    transcribe_segment = staticmethod(transcribe_segment)


backend = _Backend()
