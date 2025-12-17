"""MLX-Whisper backend (Apple Silicon optimized)."""
from typing import Any
from returns.result import Result, Success, safe
from toolz import curry
from loguru import logger

from models import AudioData


@safe
def load_model(model_name: str) -> str:
    """Return model path (mlx-whisper loads on demand)."""
    model_path = f"mlx-community/whisper-{model_name}-mlx"
    logger.info(f"[mlx-whisper] Using model: {model_path}")
    return model_path


@curry
def transcribe_segment(
    model_path: str,
    audio: AudioData,
    start: float,
    end: float,
) -> Result[str, str]:
    """Transcribe audio segment."""
    try:
        import mlx_whisper
        audio_np = audio.waveform.squeeze().numpy()
        result = mlx_whisper.transcribe(audio_np, path_or_hf_repo=model_path)
        return Success(result["text"])
    except Exception as e:
        return Success("")


class _Backend:
    load_model = staticmethod(load_model)
    transcribe_segment = staticmethod(transcribe_segment)


backend = _Backend()
