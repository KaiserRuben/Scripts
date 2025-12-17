"""Configuration dataclass with defaults."""
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class Config:
    file_path: str
    backend: str = "whisper-cpp"
    model: str | None = None  # None = use backend default
    output: str | None = None
    prompt: str | None = None
    diarize: bool = True
    summarize: bool = True

    # Ollama settings
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "gpt-oss:20b"

    # Processing settings
    sample_rate: int = 16000
    min_segment_duration: float = 0.1
    min_text_length: int = 2
    max_gap_merge: float = 2.0

    # HuggingFace token from env
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    @property
    def effective_model(self) -> str:
        """Return model or backend-specific default."""
        if self.model:
            return self.model
        defaults = {
            "faster-whisper": "Systran/faster-distil-whisper-large-v3",
            "mlx-whisper": "large-v3",
            "whisper": "large-v3",
            "whisper-cpp": "large-v3",
            "glm-asr": "zai-org/GLM-ASR-Nano-2512",
        }
        return defaults.get(self.backend, "large-v3")

    @property
    def output_path(self) -> str:
        """Return output path or generate from input."""
        if self.output:
            return self.output
        base = os.path.splitext(os.path.basename(self.file_path))[0]
        return f"transcriptions/{base}_transcription.md"
