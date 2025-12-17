"""Pure processing functions: merge, filter, normalize."""
from toolz import curry
from models import Segment


@curry
def merge_consecutive(
    segments: tuple[Segment, ...],
    max_gap: float = 2.0,
) -> tuple[Segment, ...]:
    """Merge consecutive segments from same speaker."""
    if not segments or segments[0].speaker is None:
        return segments

    merged = []
    current = None

    for seg in segments:
        if current is None:
            current = seg
        elif seg.speaker == current.speaker and seg.start - current.end <= max_gap:
            # Merge: extend current segment
            current = Segment(
                start=current.start,
                end=seg.end,
                text=f"{current.text.strip()} {seg.text.strip()}",
                speaker=current.speaker,
            )
        else:
            merged.append(current)
            current = seg

    if current:
        merged.append(current)

    return tuple(merged)


@curry
def filter_artifacts(
    segments: tuple[Segment, ...],
    min_duration: float = 0.1,
    min_text_length: int = 2,
    max_repetitions: int = 3,
) -> tuple[Segment, ...]:
    """Filter transcription artifacts."""
    filtered = []
    last_text = None
    rep_count = 0

    for seg in segments:
        text = seg.text.strip()
        duration = seg.end - seg.start

        # Skip empty/short text
        if not text or len(text) < min_text_length:
            continue

        # Skip very short segments
        if duration < min_duration:
            continue

        # Skip UNKNOWN with minimal content
        if seg.speaker == "UNKNOWN" and len(text) < 10:
            continue

        # Skip excessive repetitions
        if text == last_text:
            rep_count += 1
            if rep_count >= max_repetitions:
                continue
        else:
            rep_count = 0
            last_text = text

        filtered.append(seg)

    return tuple(filtered)


def segments_to_text(segments: tuple[Segment, ...]) -> str:
    """Combine segment texts into full text."""
    return " ".join(seg.text.strip() for seg in segments)
