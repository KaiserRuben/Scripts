"""Transcription orchestration - map over speaker segments."""
from typing import Any
from returns.result import Result, Success, Failure
from returns.pipeline import is_successful
from toolz import curry
from loguru import logger

from models import AudioData, Segment, SpeakerSegment
from pipeline.audio import extract_segment


@curry
def transcribe_segments(
    backend: Any,  # TranscriptionBackend
    model: Any,
    audio: AudioData,
    speaker_segments: tuple[SpeakerSegment, ...],
) -> Result[tuple[Segment, ...], str]:
    """Map transcription over speaker segments."""
    results = []
    total = len(speaker_segments)

    for i, spk_seg in enumerate(speaker_segments):
        if (i + 1) % 50 == 0:
            logger.info(f"Transcribing segment {i + 1}/{total}...")

        # Skip very short segments
        if spk_seg.end - spk_seg.start < 0.5:
            continue

        segment_audio = extract_segment(audio, spk_seg.start, spk_seg.end)
        text_result = backend.transcribe_segment(
            model, segment_audio, spk_seg.start, spk_seg.end
        )

        if is_successful(text_result):
            text = text_result.unwrap()
            if text.strip():
                results.append(Segment(
                    start=spk_seg.start,
                    end=spk_seg.end,
                    text=text,
                    speaker=spk_seg.speaker,
                ))
        else:
            logger.warning(f"Segment {i} failed: {text_result.failure()}")

    logger.info(f"Transcribed {len(results)} segments")
    return Success(tuple(results))


@curry
def transcribe_file_fallback(
    backend: Any,
    model: Any,
    audio: AudioData,
) -> Result[tuple[Segment, ...], str]:
    """Fallback: transcribe entire file without diarization."""
    result = backend.transcribe_segment(model, audio, 0.0, audio.duration)
    if is_successful(result):
        text = result.unwrap()
        return Success((Segment(start=0.0, end=audio.duration, text=text, speaker=None),))
    return Failure(result.failure())
