"""CSV output writer (standalone)."""
import csv
import os
from returns.result import safe
from toolz import curry

from models import PipelineResult


@curry
@safe
def save_csv(path: str, result: PipelineResult) -> str:
    """Save transcription segments to CSV."""
    csv_path = os.path.splitext(path)[0] + ".csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for seg in result.segments:
            writer.writerow([seg.speaker or "", f"{seg.start:.2f}", f"{seg.end:.2f}", seg.text])

    return csv_path
