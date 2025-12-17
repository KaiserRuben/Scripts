"""Backend registry - lazy loading to avoid import overhead."""
from typing import TYPE_CHECKING
from returns.result import Result, Failure

if TYPE_CHECKING:
    from .protocol import TranscriptionBackend


def get_backend(name: str) -> Result["TranscriptionBackend", str]:
    """Get backend by name with lazy loading."""
    loaders = {
        "faster-whisper": lambda: __import__("backends.faster_whisper", fromlist=["backend"]).backend,
        "mlx-whisper": lambda: __import__("backends.mlx_whisper", fromlist=["backend"]).backend,
        "whisper": lambda: __import__("backends.whisper", fromlist=["backend"]).backend,
        "glm-asr": lambda: __import__("backends.glm_asr", fromlist=["backend"]).backend,
    }
    if name not in loaders:
        return Failure(f"Unknown backend: {name}. Available: {list(loaders.keys())}")

    from returns.result import Success
    try:
        return Success(loaders[name]())
    except ImportError as e:
        return Failure(f"Failed to load {name}: {e}")
