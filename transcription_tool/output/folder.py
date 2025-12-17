"""Folder-based output - one run = one folder with all artifacts."""
import json
import os
import re
from datetime import datetime
from dataclasses import asdict
from returns.result import Result, Success, Failure
from toolz import curry
from loguru import logger

from models import PipelineResult, Segment, Extraction


def _sanitize_filename(name: str) -> str:
    """Sanitize filename for filesystem."""
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s]+', '-', name)
    return name[:50].lower()


def _format_segment_md(seg: Segment) -> str:
    """Format segment for markdown."""
    speaker = f"**{seg.speaker}**" if seg.speaker else ""
    return f"{speaker} [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}"


def _write_meta(path: str, result: PipelineResult) -> None:
    """Write meta.json."""
    meta = {
        "source": result.source_file,
        "created": result.created.isoformat(),
        "duration_sec": round(result.duration_sec, 1),
        "backend": result.backend,
        "model": result.model,
        "speakers": list(result.speakers),
        "segment_count": len(result.segments),
        "intent": result.analysis.extraction.intent,
        "config": result.config,
        "warnings": list(result.warnings) if result.warnings else [],
        "status": "degraded" if result.warnings else "ok",
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _write_transcript(path: str, result: PipelineResult) -> None:
    """Write transcript.md."""
    lines = ["# Transcript\n"]
    for seg in result.segments:
        lines.append(_format_segment_md(seg))
    lines.append(f"\n---\n*{len(result.segments)} segments, {result.duration_sec:.0f}s total*")

    with open(os.path.join(path, "transcript.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_transcript_csv(path: str, result: PipelineResult) -> None:
    """Write transcript.csv."""
    import csv
    with open(os.path.join(path, "transcript.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for seg in result.segments:
            writer.writerow([seg.speaker or "", f"{seg.start:.2f}", f"{seg.end:.2f}", seg.text])


def _write_summary(path: str, result: PipelineResult) -> None:
    """Write summary.md."""
    with open(os.path.join(path, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Summary\n\n")
        f.write(result.analysis.summary)


def _extraction_to_dict(ext: Extraction) -> dict:
    """Convert Extraction to JSON-serializable dict."""
    return {
        "intent": ext.intent,
        "asks": [{"what": a.what, "from_speaker": a.from_speaker, "urgency": a.urgency} for a in ext.asks],
        "commitments": [{"who": c.who, "what": c.what, "deadline": c.deadline} for c in ext.commitments],
        "decisions": [{"what": d.what, "by": d.by, "context": d.context} for d in ext.decisions],
        "open_loops": list(ext.open_loops),
        "key_topics": list(ext.key_topics),
        "suggested_reply": ext.suggested_reply,
    }


def _write_extracted_json(path: str, result: PipelineResult) -> None:
    """Write extracted.json."""
    data = _extraction_to_dict(result.analysis.extraction)
    with open(os.path.join(path, "extracted.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_extracted_md(path: str, result: PipelineResult) -> None:
    """Write extracted.md - human-readable structured data."""
    ext = result.analysis.extraction
    lines = [f"# Extracted Information\n"]
    lines.append(f"**Intent:** {ext.intent or 'Unknown'}\n")

    if ext.key_topics:
        lines.append("## Key Topics")
        for topic in ext.key_topics:
            lines.append(f"- {topic}")
        lines.append("")

    if ext.asks:
        lines.append("## Asks / Requests")
        for ask in ext.asks:
            urgency = f" [{ask.urgency}]" if ask.urgency else ""
            speaker = f" (from {ask.from_speaker})" if ask.from_speaker else ""
            lines.append(f"- {ask.what}{speaker}{urgency}")
        lines.append("")

    if ext.commitments:
        lines.append("## Commitments")
        for c in ext.commitments:
            deadline = f" (by {c.deadline})" if c.deadline else ""
            lines.append(f"- **{c.who}**: {c.what}{deadline}")
        lines.append("")

    if ext.decisions:
        lines.append("## Decisions")
        for d in ext.decisions:
            by = f" (by {d.by})" if d.by else ""
            ctx = f" - {d.context}" if d.context else ""
            lines.append(f"- {d.what}{by}{ctx}")
        lines.append("")

    if ext.open_loops:
        lines.append("## Open Questions / Unresolved")
        for loop in ext.open_loops:
            lines.append(f"- {loop}")
        lines.append("")

    with open(os.path.join(path, "extracted.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_tasks(path: str, result: PipelineResult) -> None:
    """Write tasks.md - actionable checklist."""
    ext = result.analysis.extraction
    lines = ["# Tasks\n"]

    if ext.asks:
        lines.append("## To Respond")
        for ask in ext.asks:
            urgency_mark = "❗" if ask.urgency == "high" else "❓" if ask.urgency == "medium" else ""
            lines.append(f"- [ ] {urgency_mark} {ask.what}")
        lines.append("")

    if ext.commitments:
        lines.append("## Commitments Made")
        for c in ext.commitments:
            deadline = f" *(by {c.deadline})*" if c.deadline else ""
            lines.append(f"- [ ] {c.who}: {c.what}{deadline}")
        lines.append("")

    if ext.open_loops:
        lines.append("## Follow Up")
        for loop in ext.open_loops:
            lines.append(f"- [ ] {loop}")

    with open(os.path.join(path, "tasks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_reply(path: str, result: PipelineResult) -> None:
    """Write reply.md if suggested_reply exists."""
    if result.analysis.extraction.suggested_reply:
        with open(os.path.join(path, "reply.md"), "w", encoding="utf-8") as f:
            f.write("# Suggested Reply\n\n")
            f.write(result.analysis.extraction.suggested_reply)


@curry
def save_folder(base_path: str, result: PipelineResult) -> Result[str, str]:
    """Save all artifacts to a folder."""
    try:
        # Create folder name: date_source-name
        date_str = result.created.strftime("%Y-%m-%d")
        name = _sanitize_filename(result.source_file)
        folder_name = f"{date_str}_{name}"
        folder_path = os.path.join(base_path, folder_name)

        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Writing to: {folder_path}")

        # Write all artifacts
        _write_meta(folder_path, result)
        _write_transcript(folder_path, result)
        _write_transcript_csv(folder_path, result)
        _write_summary(folder_path, result)
        _write_extracted_json(folder_path, result)
        _write_extracted_md(folder_path, result)
        _write_tasks(folder_path, result)
        _write_reply(folder_path, result)

        return Success(folder_path)

    except Exception as e:
        return Failure(f"Failed to save folder: {e}")
