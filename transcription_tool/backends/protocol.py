"""Backend protocol - structural typing for transcription backends."""
from typing import Protocol, Any
from returns.result import Result

from models import AudioData


class TranscriptionBackend(Protocol):
    """All backends implement this interface."""

    def load_model(self, model_name: str) -> Result[Any, str]:
        """Load and return the model."""
        ...

    def transcribe_segment(
        self,
        model: Any,
        audio: AudioData,
        start: float,
        end: float,
    ) -> Result[str, str]:
        """Transcribe a single audio segment."""
        ...
