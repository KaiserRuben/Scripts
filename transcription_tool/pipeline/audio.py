"""Audio loading and segment extraction."""
import subprocess
import numpy as np
import torch
from returns.result import Result, Success, Failure, safe
from toolz import curry

from models import AudioData


@safe
def load_audio(file_path: str, sample_rate: int = 16000) -> AudioData:
    """Load audio file via ffmpeg, return AudioData."""
    process = subprocess.run(
        [
            "ffmpeg", "-i", file_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1", "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return AudioData(waveform=waveform, sample_rate=sample_rate)


@curry
def extract_segment(audio: AudioData, start: float, end: float) -> AudioData:
    """Extract a time segment from audio (pure function)."""
    start_sample = int(start * audio.sample_rate)
    end_sample = int(end * audio.sample_rate)
    segment_waveform = audio.waveform[:, start_sample:end_sample]
    return AudioData(waveform=segment_waveform, sample_rate=audio.sample_rate)
