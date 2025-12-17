"""Speaker diarization using pyannote.audio."""
import warnings
import torch
from returns.result import Result, Success, Failure
from toolz import curry
from loguru import logger

from models import AudioData, SpeakerSegment

# Suppress torchcodec warning - we use preloaded waveforms, not file-based loading
warnings.filterwarnings("ignore", message="torchcodec is not installed")


@curry
def perform_diarization(
    hf_token: str | None,
    audio: AudioData,
) -> Result[tuple[SpeakerSegment, ...], str]:
    """Perform speaker diarization, return speaker segments."""
    if not hf_token:
        return Failure("HF_TOKEN required for diarization")

    try:
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline.to(torch.device(device))

        logger.info(f"Running diarization on {device}...")
        audio_dict = {
            "waveform": audio.waveform.to(device),
            "sample_rate": audio.sample_rate,
        }
        result = pipeline(audio_dict)

        segments = tuple(
            SpeakerSegment(start=seg.start, end=seg.end, speaker=label)
            for seg, _, label in result.speaker_diarization.itertracks(yield_label=True)
        )

        speakers = set(s.speaker for s in segments)
        logger.info(f"Detected {len(speakers)} speakers in {len(segments)} segments")
        return Success(segments)

    except Exception as e:
        return Failure(f"Diarization failed: {e}")


def no_diarization(audio: AudioData) -> Result[tuple[SpeakerSegment, ...], str]:
    """Fallback: single segment covering entire audio."""
    return Success((
        SpeakerSegment(start=0.0, end=audio.duration, speaker="SPEAKER_00"),
    ))
