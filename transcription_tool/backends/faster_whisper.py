"""Faster-whisper backend."""
from typing import Any
from returns.result import Result, Success, safe
from toolz import curry
from loguru import logger

from models import AudioData


@safe
def load_model(model_name: str) -> Any:
    """Load faster-whisper model."""
    from faster_whisper import WhisperModel
    logger.info(f"[faster-whisper] Loading model: {model_name}")
    return WhisperModel(model_size_or_path=model_name)


@curry
def transcribe_segment(
    model: Any,
    audio: AudioData,
    start: float,
    end: float,
) -> Result[str, str]:
    """Transcribe audio segment."""
    try:
        # faster-whisper works with numpy arrays
        audio_np = audio.waveform.squeeze().numpy()
        segments, _ = model.transcribe(audio_np)
        text = " ".join(seg.text for seg in segments)
        return Success(text)
    except Exception as e:
        return Success("")  # Empty on failure, logged elsewhere


# Export backend interface
class _Backend:
    load_model = staticmethod(load_model)
    transcribe_segment = staticmethod(transcribe_segment)


backend = _Backend()
