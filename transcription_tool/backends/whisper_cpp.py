"""whisper.cpp backend via pywhispercpp - fast Metal-accelerated transcription."""
from typing import Any
from returns.result import Result, Success, Failure, safe
from toolz import curry
from loguru import logger

from models import AudioData


@safe
def load_model(model_name: str) -> Any:
    """Load whisper.cpp model via pywhispercpp."""
    from pywhispercpp.model import Model

    logger.info(f"[whisper.cpp] Loading model: {model_name}")
    # Model auto-downloads from HuggingFace if not local
    return Model(model_name, print_realtime=False, print_progress=False)


@curry
def transcribe_segment(
    model: Any,
    audio: AudioData,
    start: float,
    end: float,
) -> Result[str, str]:
    """Transcribe audio segment with whisper.cpp."""
    try:
        import tempfile
        import soundfile as sf

        # pywhispercpp needs a file path, write temp WAV
        audio_np = audio.waveform.squeeze().numpy()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, audio.sample_rate)
            segments = model.transcribe(f.name)

        text = " ".join(seg.text for seg in segments)
        return Success(text.strip())

    except Exception as e:
        logger.warning(f"whisper.cpp transcription failed: {e}")
        return Success("")


class _Backend:
    load_model = staticmethod(load_model)
    transcribe_segment = staticmethod(transcribe_segment)


backend = _Backend()
