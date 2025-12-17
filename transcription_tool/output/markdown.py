"""Markdown output writer (standalone)."""
import os
from returns.result import safe
from toolz import curry

from models import PipelineResult


def _format_segment(seg) -> str:
    """Format single segment as markdown."""
    speaker = f"**{seg.speaker}**" if seg.speaker else ""
    return f"{speaker} [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}"


@curry
@safe
def save_markdown(path: str, result: PipelineResult) -> str:
    """Save transcription to markdown file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Transcription\n\n")
        for seg in result.segments:
            f.write(f"{_format_segment(seg)}\n")

        if result.analysis.summary:
            f.write(f"\n# Summary\n\n{result.analysis.summary}")

    return path
