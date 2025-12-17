"""Immutable data types for transcription pipeline."""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import torch


@dataclass(frozen=True)
class AudioData:
    waveform: torch.Tensor
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.waveform.shape[-1] / self.sample_rate


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


# === Structured Extraction Types ===

@dataclass(frozen=True)
class Commitment:
    who: str
    what: str
    deadline: str | None = None


@dataclass(frozen=True)
class Decision:
    what: str
    by: str | None = None
    context: str | None = None


@dataclass(frozen=True)
class Ask:
    what: str
    from_speaker: str | None = None
    urgency: str | None = None  # low | medium | high


@dataclass(frozen=True)
class Extraction:
    """Structured extraction - all fields optional/empty if not found."""
    intent: str | None = None  # meeting | voicemail | interview | brainstorm | monologue
    asks: tuple[Ask, ...] = ()
    commitments: tuple[Commitment, ...] = ()
    decisions: tuple[Decision, ...] = ()
    open_loops: tuple[str, ...] = ()
    key_topics: tuple[str, ...] = ()
    suggested_reply: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis output."""
    summary: str  # Always present
    extraction: Extraction  # Structured data (may be empty)


@dataclass(frozen=True)
class PipelineResult:
    """Full pipeline output - ready for folder export."""
    source_file: str
    created: datetime
    duration_sec: float
    backend: str
    model: str
    speakers: tuple[str, ...]
    segments: tuple[Segment, ...]
    full_text: str
    analysis: AnalysisResult
    config: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()  # Track degradations
