"""Multi-pass LLM analysis via Ollama."""
import json
import requests
from returns.result import Result, Success, Failure
from toolz import curry, pipe
from loguru import logger

from models import Segment, Extraction, AnalysisResult, Ask, Commitment, Decision
from config import Config


def _build_transcript(segments: tuple[Segment, ...]) -> str:
    """Build speaker-attributed transcript."""
    lines = []
    for seg in segments:
        speaker = seg.speaker or "UNKNOWN"
        lines.append(f"[{seg.start:.1f}s] {speaker}: {seg.text}")
    return "\n".join(lines)


def _build_speaker_stats(segments: tuple[Segment, ...]) -> str:
    """Build speaker statistics."""
    stats: dict[str, dict] = {}
    for seg in segments:
        spk = seg.speaker or "UNKNOWN"
        if spk not in stats:
            stats[spk] = {"duration": 0.0, "segments": 0}
        stats[spk]["duration"] += seg.end - seg.start
        stats[spk]["segments"] += 1

    total = sum(s["duration"] for s in stats.values())
    lines = []
    for spk, data in sorted(stats.items()):
        pct = (data["duration"] / total * 100) if total > 0 else 0
        lines.append(f"- {spk}: {data['duration']:.0f}s ({pct:.0f}%)")
    return "\n".join(lines)


def _call_ollama(config: Config, prompt: str) -> Result[str, str]:
    """Make Ollama API call."""
    try:
        response = requests.post(
            config.ollama_url,
            json={"model": config.ollama_model, "prompt": prompt, "stream": False},
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        response.raise_for_status()
        return Success(response.json()["response"])
    except Exception as e:
        return Failure(f"Ollama call failed: {e}")


ANALYSIS_PROMPT = """Du analysierst ein Transkript. Antworte mit ZWEI Teilen:

=== TRANSKRIPT ===
{transcript}

=== SPRECHER-STATISTIK ===
{speaker_stats}

=== NUTZERPROMPT ===
{user_prompt}

=== DEINE AUFGABE ===

**TEIL 1 - ZUSAMMENFASSUNG**
Schreibe eine umfassende Zusammenfassung. Erfasse ALLE wichtigen Punkte.
Ordne Aussagen den Sprechern zu. Sei gründlich aber prägnant.

**TEIL 2 - STRUKTURIERTE EXTRAKTION**
Extrahiere (falls vorhanden) als JSON-Block. Leere Arrays wenn nichts gefunden.

```json
{{
  "intent": "meeting|voicemail|interview|brainstorm|monologue|other",
  "asks": [
    {{"what": "...", "from_speaker": "SPEAKER_XX", "urgency": "low|medium|high"}}
  ],
  "commitments": [
    {{"who": "SPEAKER_XX", "what": "...", "deadline": "..." oder null}}
  ],
  "decisions": [
    {{"what": "...", "by": "SPEAKER_XX", "context": "..."}}
  ],
  "open_loops": ["Ungelöste Fragen/Themen..."],
  "key_topics": ["Hauptthema 1", "Hauptthema 2"],
  "suggested_reply": "Antwortvorschlag falls sinnvoll, sonst null"
}}
```

Beginne mit der Zusammenfassung, dann der JSON-Block am Ende."""


def _parse_extraction(text: str) -> Extraction:
    """Parse JSON extraction from LLM response."""
    try:
        # Find JSON block
        start = text.rfind("```json")
        end = text.rfind("```", start + 7)
        if start == -1 or end == -1:
            # Try without markdown
            start = text.rfind("{")
            end = text.rfind("}") + 1
        else:
            start = text.find("{", start)
            end = text.rfind("}", start, end) + 1

        if start == -1 or end <= start:
            return Extraction()

        data = json.loads(text[start:end])

        return Extraction(
            intent=data.get("intent"),
            asks=tuple(
                Ask(what=a["what"], from_speaker=a.get("from_speaker"), urgency=a.get("urgency"))
                for a in data.get("asks", [])
            ),
            commitments=tuple(
                Commitment(who=c["who"], what=c["what"], deadline=c.get("deadline"))
                for c in data.get("commitments", [])
            ),
            decisions=tuple(
                Decision(what=d["what"], by=d.get("by"), context=d.get("context"))
                for d in data.get("decisions", [])
            ),
            open_loops=tuple(data.get("open_loops", [])),
            key_topics=tuple(data.get("key_topics", [])),
            suggested_reply=data.get("suggested_reply"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse extraction: {e}")
        return Extraction()


def _extract_summary(text: str) -> str:
    """Extract summary part (everything before JSON)."""
    # Find where JSON starts
    markers = ["```json", '{"intent"']
    earliest = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1 and pos < earliest:
            earliest = pos
    return text[:earliest].strip()


@curry
def analyze(
    config: Config,
    segments: tuple[Segment, ...],
) -> Result[AnalysisResult, str]:
    """Multi-pass analysis: summary + structured extraction."""
    if not segments:
        return Success(AnalysisResult(summary="Keine Segmente zum Analysieren.", extraction=Extraction()))

    prompt = ANALYSIS_PROMPT.format(
        transcript=_build_transcript(segments),
        speaker_stats=_build_speaker_stats(segments),
        user_prompt=config.prompt or "Keine spezifischen Anweisungen.",
    )

    logger.info("Running LLM analysis...")
    result = _call_ollama(config, prompt)

    if not result.is_success():
        return Failure(result.failure())

    response = result.unwrap()
    summary = _extract_summary(response)
    extraction = _parse_extraction(response)

    logger.info(f"Analysis complete: intent={extraction.intent}, "
                f"asks={len(extraction.asks)}, commitments={len(extraction.commitments)}")

    return Success(AnalysisResult(summary=summary, extraction=extraction))
