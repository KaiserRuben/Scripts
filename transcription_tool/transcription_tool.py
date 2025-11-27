import argparse
import requests
import os
import csv
from functools import partial
import torch
import numpy as np
import subprocess

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"


# === Backend Registry ===
def create_backend_loader(backend_type, model_name):
    """Pure function: Returns (model/config, transcribe_function) for given backend."""
    loaders = {
        'faster-whisper': lambda: _load_faster_whisper(model_name),
        'mlx-whisper': lambda: _load_mlx_whisper(model_name),
        'whisper': lambda: _load_whisper(model_name)
    }
    return loaders[backend_type]()


def _load_faster_whisper(model_name):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size_or_path=model_name)
    return partial(transcribe_file_faster_whisper, model)


def _load_mlx_whisper(model_name):
    model_path = f"mlx-community/whisper-{model_name}-mlx"
    return partial(transcribe_file_mlx_whisper, model_name=model_path)


def _load_whisper(model_name):
    import whisper
    model = whisper.load_model(model_name)
    return partial(transcribe_file_whisper, model)


# === Segment Normalization ===
def normalize_segment(segment, backend):
    """Pure function: Normalize different backend segment formats to dict."""
    if backend == 'faster-whisper':
        return {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        }
    return segment  # Already dict format for whisper/mlx-whisper


def normalize_segments(segments, backend):
    """Pure function: Normalize all segments."""
    return [normalize_segment(seg, backend) for seg in segments]


# === Speaker Diarization ===
def load_audio_with_ffmpeg(file_path, target_sr=16000):
    """Load audio file using ffmpeg, handling various formats including M4A."""
    # Use ffmpeg to decode audio to raw PCM format
    command = [
        'ffmpeg',
        '-i', file_path,
        '-f', 's16le',  # 16-bit signed little-endian PCM
        '-acodec', 'pcm_s16le',
        '-ar', str(target_sr),  # Resample to target sample rate
        '-ac', '1',  # Convert to mono
        '-'  # Output to stdout
    ]

    try:
        # Run ffmpeg and capture output
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Convert bytes to numpy array
        audio_data = np.frombuffer(process.stdout, dtype=np.int16)

        # Normalize to [-1, 1] float32
        waveform = audio_data.astype(np.float32) / 32768.0

        # Convert to torch tensor and add channel dimension
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

        return waveform_tensor, target_sr

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to decode audio: {e.stderr.decode()}")


def perform_diarization(file_path):
    """Perform speaker diarization using pyannote.audio with MPS acceleration."""
    from pyannote.audio import Pipeline

    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set. Required for speaker diarization.")

    print("\n[Diarization] Loading audio for speaker detection...")
    waveform, sr = load_audio_with_ffmpeg(file_path, target_sr=16000)

    waveform_tensor = waveform.to('mps')
    audio_dict = {'waveform': waveform_tensor, 'sample_rate': sr}

    print("[Diarization] Loading pyannote speaker-diarization-3.1 model...")
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token=hf_token)
    pipeline.to(torch.device('mps'))

    print("[Diarization] Running speaker diarization (MPS accelerated)...")
    diarize_output = pipeline(audio_dict)

    # Extract speaker segments from DiarizeOutput
    annotation = diarize_output.speaker_diarization
    speaker_segments = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        speaker_segments.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': label
        })

    print(f"[Diarization] Detected {len(set(s['speaker'] for s in speaker_segments))} speakers in {len(speaker_segments)} segments")
    return speaker_segments


def assign_speakers_to_segments(transcription_segments, speaker_segments):
    """Pure function: Assign speaker labels to transcription segments based on timestamp overlap."""
    def find_speaker(trans_start, trans_end):
        """Find the speaker with maximum overlap for this transcription segment."""
        max_overlap = 0
        assigned_speaker = None

        for spk_seg in speaker_segments:
            # Calculate overlap between transcription segment and speaker segment
            overlap_start = max(trans_start, spk_seg['start'])
            overlap_end = min(trans_end, spk_seg['end'])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = spk_seg['speaker']

        return assigned_speaker if assigned_speaker else "UNKNOWN"

    # Add speaker field to each segment
    enriched_segments = []
    for seg in transcription_segments:
        enriched_seg = seg.copy()
        enriched_seg['speaker'] = find_speaker(seg['start'], seg['end'])
        enriched_segments.append(enriched_seg)

    return enriched_segments


def transcribe_file_faster_whisper(model, file_path):
    print(f"[faster-whisper] Transcribing file: {file_path}")
    segments, info = model.transcribe(file_path)

    all_text = []
    segment_list = []
    for segment in segments:
        all_text.append(segment.text)
        segment_list.append(segment)

    full_text = " ".join(all_text)
    print(f"[faster-whisper] Transcription complete: {len(segment_list)} segments")
    return full_text, segment_list, info


def transcribe_file_whisper(model, file_path):
    print(f"[whisper] Transcribing file: {file_path}")
    result = model.transcribe(file_path)
    print(f"[whisper] Transcription complete: {len(result['segments'])} segments")
    return result["text"], result["segments"], result


def transcribe_file_mlx_whisper(file_path, model_name):
    print(f"[mlx-whisper] Transcribing file: {file_path}")
    import mlx_whisper

    result = mlx_whisper.transcribe(file_path, path_or_hf_repo=model_name)
    print(f"[mlx-whisper] Transcription complete: {len(result['segments'])} segments")
    return result["text"], result["segments"], result


def generate_speaker_context(segments):
    """Generate speaker context and statistics for enhanced summarization."""
    if not segments or 'speaker' not in segments[0]:
        return None

    # Calculate speaker statistics
    speakers = {}
    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speakers:
            speakers[speaker] = {
                'segments': 0,
                'duration': 0.0,
                'text_parts': []
            }
        speakers[speaker]['segments'] += 1
        speakers[speaker]['duration'] += seg['end'] - seg['start']
        speakers[speaker]['text_parts'].append(seg['text'])

    # Build context string
    context_parts = []
    context_parts.append(f"Anzahl Sprecher: {len(speakers)}")

    total_duration = sum(s['duration'] for s in speakers.values())
    for speaker, data in sorted(speakers.items()):
        percentage = (data['duration'] / total_duration * 100) if total_duration > 0 else 0
        context_parts.append(
            f"{speaker}: {data['segments']} Segmente, "
            f"{data['duration']:.1f}s ({percentage:.1f}% Redezeit)"
        )

    return "\n".join(context_parts), speakers


def build_speaker_aware_transcript(segments):
    """Build a speaker-aware transcript for summarization."""
    if not segments or 'speaker' not in segments[0]:
        return None

    # Group consecutive segments by speaker
    grouped = []
    current_speaker = None
    current_text = []

    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker != current_speaker:
            if current_text:
                grouped.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [seg['text']]
        else:
            current_text.append(seg['text'])

    # Add last group
    if current_text:
        grouped.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(grouped)


def extract_conversation_patterns(segments):
    """Extract structural patterns from conversation for enhanced summarization."""
    if not segments or 'speaker' not in segments[0]:
        return None

    patterns = {
        'questions': [],
        'potential_action_items': [],
        'turn_taking': {},
        'temporal_insights': {}
    }

    # Pattern: Detect questions (segments ending with ?)
    import re
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        if text.endswith('?') or '?' in text:
            patterns['questions'].append({
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'text': text,
                'time': seg['start']
            })

    # Pattern: Detect potential action items (common German action phrases)
    action_patterns = [
        r'\b(werden|werde|wird|sollten|sollte|müssen|muss|können wir|kann ich)\b',
        r'\b(ich werde|wir werden|du solltest|Sie sollten|lass uns|lasst uns)\b',
        r'\b(TODO|Action|Aufgabe|nächste Schritt)\b'
    ]
    combined_pattern = '|'.join(action_patterns)

    for seg in segments:
        text = seg['text']
        if re.search(combined_pattern, text, re.IGNORECASE):
            patterns['potential_action_items'].append({
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'text': text,
                'time': seg['start']
            })

    # Pattern: Turn-taking analysis
    speaker_turns = {}
    last_speaker = None
    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        if speaker not in speaker_turns:
            speaker_turns[speaker] = {'turns': 0, 'interruptions': 0}

        if speaker != last_speaker:
            speaker_turns[speaker]['turns'] += 1
        last_speaker = speaker

    patterns['turn_taking'] = speaker_turns

    # Temporal analysis: Beginning, middle, end
    total_duration = segments[-1]['end'] - segments[0]['start'] if segments else 0
    third = total_duration / 3

    beginning_speakers = set()
    middle_speakers = set()
    end_speakers = set()

    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        relative_time = seg['start'] - segments[0]['start']

        if relative_time < third:
            beginning_speakers.add(speaker)
        elif relative_time < 2 * third:
            middle_speakers.add(speaker)
        else:
            end_speakers.add(speaker)

    patterns['temporal_insights'] = {
        'beginning_speakers': list(beginning_speakers),
        'middle_speakers': list(middle_speakers),
        'end_speakers': list(end_speakers),
        'total_duration': total_duration
    }

    return patterns


def format_conversation_hints(patterns):
    """Format extracted patterns into structured hints for the LLM."""
    if not patterns:
        return ""

    hints = []

    # Questions
    if patterns['questions']:
        hints.append(f"\nERKANNTE FRAGEN ({len(patterns['questions'])} gefunden):")
        for q in patterns['questions'][:10]:  # Limit to first 10
            hints.append(f"  - {q['speaker']} ({q['time']:.0f}s): {q['text'][:100]}")

    # Action items
    if patterns['potential_action_items']:
        hints.append(f"\nPOTENZIELLE ACTION ITEMS ({len(patterns['potential_action_items'])} gefunden):")
        for item in patterns['potential_action_items'][:10]:
            hints.append(f"  - {item['speaker']} ({item['time']:.0f}s): {item['text'][:100]}")

    # Turn-taking
    if patterns['turn_taking']:
        hints.append(f"\nTURN-TAKING ANALYSE:")
        for speaker, data in sorted(patterns['turn_taking'].items(), key=lambda x: x[1]['turns'], reverse=True):
            hints.append(f"  - {speaker}: {data['turns']} Redewechsel")

    # Temporal
    temporal = patterns.get('temporal_insights', {})
    if temporal:
        hints.append(f"\nZEITLICHE STRUKTUR (Gesamtdauer: {temporal.get('total_duration', 0):.0f}s):")
        hints.append(f"  - Anfang: {', '.join(temporal.get('beginning_speakers', []))}")
        hints.append(f"  - Mitte: {', '.join(temporal.get('middle_speakers', []))}")
        hints.append(f"  - Ende: {', '.join(temporal.get('end_speakers', []))}")

    return "\n".join(hints)


def summarize(full_text, segments=None, prompt=None):
    """Generate summary with optional speaker-aware analysis."""
    headers = {"Content-Type": "application/json"}

    # Check if we have speaker information
    has_speakers = segments and len(segments) > 0 and 'speaker' in segments[0]

    if has_speakers:
        # Generate speaker context and structured transcript
        speaker_context, speakers = generate_speaker_context(segments)
        speaker_transcript = build_speaker_aware_transcript(segments)

        # Extract conversation patterns
        patterns = extract_conversation_patterns(segments)
        conversation_hints = format_conversation_hints(patterns) if patterns else ""

        # Enhanced prompt with speaker awareness, patterns, and chain-of-thought reasoning
        base_prompt = f'''Du bist ein Meeting-Analyse-Experte. Erstelle eine professionelle Meeting-Zusammenfassung, die für Teilnehmer und Abwesende gleichermaßen wertvoll ist.

=== EINGABEDATEN ===

SPRECHER-STATISTIK:
{speaker_context}

AUTOMATISCH ERKANNTE MUSTER:
{conversation_hints}

TRANSKRIPT (nach Sprecher gruppiert):
"""
{speaker_transcript}
"""

VOLLTEXT (zur Referenz):
"""
{full_text}
"""

NUTZERPROMPT: {prompt if prompt else "Keine zusätzlichen Anweisungen"}

=== ANALYSE-ANLEITUNG ===

Gehe systematisch vor:

SCHRITT 1 - KONTEXT VERSTEHEN:
- Was ist der Typ dieses Meetings? (Brainstorming, Entscheidungsfindung, Status-Update, Interview, Diskussion)
- Welches übergeordnete Ziel/Thema hat das Gespräch?
- Wie ist die zeitliche Entwicklung? (Anfang → Mitte → Ende)

SCHRITT 2 - SPRECHER ANALYSIEREN:
- Identifiziere die Rolle jedes Sprechers (Moderator, Experte, Fragensteller, Entscheider, etc.)
- Wer treibt das Gespräch voran? Wer reagiert primär?
- Nutze die Turn-Taking-Analyse und erkannten Fragen

SCHRITT 3 - INHALTE STRUKTURIEREN:
- Extrahiere die Hauptthemen in chronologischer oder thematischer Ordnung
- Identifiziere Entscheidungen, offene Fragen und nächste Schritte
- Nutze die erkannten Action Items als Hinweise (aber validiere sie!)

SCHRITT 4 - QUALITÄTSKONTROLLE:
- Sind alle wichtigen Punkte erfasst?
- Sind Action Items eindeutig formuliert und zugeordnet?
- Würde jemand, der nicht dabei war, das Meeting verstehen?

=== AUSGABEFORMAT ===

Erstelle eine strukturierte Zusammenfassung mit diesen Abschnitten:

## 1. Executive Summary
[2-3 Sätze: Was war der Zweck, was wurde erreicht?]

## 2. Meeting-Kontext
- **Typ**: [Brainstorming/Entscheidung/Status/etc.]
- **Teilnehmer**: [Liste mit identifizierten Rollen]
- **Dauer**: [Zeit in Minuten]

## 3. Sprecher-Analyse
Für jeden Sprecher (geordnet nach Relevanz/Redeanteil):

**[SPEAKER_X]** ([Rolle])
- Hauptbeiträge: [Was hat diese Person eingebracht?]
- Kernaussagen: [1-2 wichtigste Statements, evtl. mit Zitat]
- Interaktionsstil: [Wie kommuniziert die Person?]

## 4. Gesprächsverlauf & Hauptthemen
[Chronologisch oder thematisch gruppiert, mit Speaker-Attribution]

**Anfangsphase** ([Zeit]):
- [Thema/Punkt mit Sprecher]

**Hauptdiskussion** ([Zeit]):
- [Thema/Punkt mit Sprecher]

**Abschluss** ([Zeit]):
- [Thema/Punkt mit Sprecher]

## 5. Entscheidungen & Vereinbarungen
[Falls vorhanden: Konkrete Entscheidungen mit Kontext]
- [Entscheidung] (entschieden durch [Sprecher])

## 6. Action Items
[Klare, umsetzbare Aufgaben mit Verantwortung]
- [ ] [Aufgabe] - Zugeordnet: [Sprecher] - [Kontext falls nötig]

## 7. Offene Fragen & Follow-ups
[Fragen die gestellt, aber nicht beantwortet wurden]
- [Frage] (von [Sprecher])

## 8. Wichtige Erkenntnisse
[3-5 Schlüsselerkenntnisse oder Quotes]
- [Erkenntnis mit Kontext]

=== QUALITÄTSSTANDARDS ===

✓ Präzise: Nutze spezifische Details, keine Allgemeinplätze
✓ Zugeordnet: Ordne Aussagen den richtigen Sprechern zu
✓ Actionable: Action Items müssen klar und umsetzbar sein
✓ Vollständig: Decke alle wichtigen Themen ab
✓ Objektiv: Neutrale Sprache, keine Interpretation über das Gesagte hinaus

Beginne mit der Analyse!
'''
    else:
        # Fallback to simple summarization without speaker info
        base_prompt = f'''Fasse das vorliegende Transkript kurz und strukturiert zusammen:
"""
{full_text}
"""

Validiere Vollständigkeit. Wenn Aufforderungen oder Fragen enthalten sind, hebe diese explizit hervor. Füge eine kurze Bewertung/Analyse des Inhalts an.

Nutzerprompt: {prompt if prompt else "Keine"}
'''

    data = {
        "model": OLLAMA_MODEL,
        "prompt": base_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['response']
    except (requests.RequestException, KeyError) as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary"


# === Pure Formatters ===
def format_segment_csv(segment):
    """Pure function: Format a normalized segment for CSV."""
    if 'speaker' in segment:
        return [segment['speaker'], f"{segment['start']:.2f}", f"{segment['end']:.2f}", segment['text']]
    return [f"{segment['start']:.2f}", f"{segment['end']:.2f}", segment['text']]


def format_segment_markdown(segment):
    """Pure function: Format a normalized segment for markdown."""
    if 'speaker' in segment:
        return f"**{segment['speaker']}** [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
    return f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"


def save_to_csv(segments, output_file):
    """Save transcription segments to a CSV file (expects normalized segments)."""
    csv_file = os.path.splitext(output_file)[0] + '.csv'

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Dynamic header based on whether segments have speaker information
            if segments and 'speaker' in segments[0]:
                writer.writerow(['Speaker', 'Start Time (s)', 'End Time (s)', 'Text'])
            else:
                writer.writerow(['Start Time (s)', 'End Time (s)', 'Text'])
            writer.writerows(format_segment_csv(seg) for seg in segments)

        print(f"CSV file saved to: {csv_file}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")


def save_to_markdown(segments, full_text, summary, output_file):
    """Save transcription to a markdown file (expects normalized segments)."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Transcription:\n\n")

            for segment in segments:
                f.write(f"{format_segment_markdown(segment)}\n")

            if summary:
                f.write(f"# Summary:\n\n{summary}")

        print(f"\nTranscription saved to: {output_file}")
    except IOError as e:
        print(f"Error saving markdown file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a file using Whisper")
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument("--prompt", help="Prompt for Summarization", default=None)
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model to use (default: large-v3)")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("-s", "--summarize", action="store_true", default=True,
                        help="Summarize the transcription")
    parser.add_argument("--backend", choices=['faster-whisper', 'whisper', 'mlx-whisper'],
                        default='mlx-whisper', help="Which backend to use")
    parser.add_argument("--diarize", action="store_true", default=True,
                        help="Enable speaker diarization (requires HF_TOKEN env var)")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.file_path):
        print(f"Error: Input file '{args.file_path}' does not exist")
        return

    print(f"Loading model: {args.model}")
    try:
        # Load backend using registry pattern
        transcribe_fn = create_backend_loader(args.backend, args.model)

        # Transcribe once and store results
        full_text, segments, info = transcribe_fn(args.file_path)

        # Normalize segments to common format
        normalized_segments = normalize_segments(segments, args.backend)

        # Perform speaker diarization if requested
        if args.diarize:
            try:
                speaker_segments = perform_diarization(args.file_path)
                normalized_segments = assign_speakers_to_segments(normalized_segments, speaker_segments)
            except Exception as e:
                print(f"\nWarning: Speaker diarization failed: {e}")
                print("Continuing without speaker information...")

        # Generate summary if requested
        summary = ""
        if args.summarize:
            print("\nGenerating summary...")
            summary = summarize(full_text, normalized_segments, args.prompt)
            print(f"\nSummary:\n{summary}")

        # Determine output file name if not provided
        if args.output is None:
            base_name = os.path.splitext(os.path.basename(args.file_path))[0]
            output_file = f"transcriptions/{base_name}_transcription.md"
        else:
            output_file = args.output

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save both markdown and CSV versions (using normalized segments)
        save_to_markdown(normalized_segments, full_text, summary, output_file)
        save_to_csv(normalized_segments, output_file)

    except ImportError as e:
        print(f"Error: Required module not found. {e}")
        print(f"Please install the required packages for the {args.backend} backend.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()