import argparse
import requests
import os
import csv
from functools import partial

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


def transcribe_file_faster_whisper(model, file_path):
    print(f"[faster-whisper] Transcribing file: {file_path}")
    segments, info = model.transcribe(file_path)

    all_text = []
    print("\nTranscript:")
    print("-----------")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        all_text.append(segment.text)

    full_text = " ".join(all_text)
    return full_text, list(segments), info  # Convert generator to list to allow multiple iterations


def transcribe_file_whisper(model, file_path):
    print(f"[whisper] Transcribing file: {file_path}")
    result = model.transcribe(file_path)

    print("\nTranscript:")
    print("-----------")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return result["text"], result["segments"], result


def transcribe_file_mlx_whisper(file_path, model_name):
    print(f"[mlx-whisper] Transcribing file: {file_path}")
    import mlx_whisper

    result = mlx_whisper.transcribe(file_path, path_or_hf_repo=model_name)

    print("\nTranscript:")
    print("-----------")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return result["text"], result["segments"], result


def summarize(text, prompt=None):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL,
        "prompt": f'Fasse das vorliegende Transkript kurz und strukturiert zusammen: \n"""\n{text}\n"""\n\nValidiere vollständigkeit. Wenn Aufforderungen oder Fragen enthalten sind, hebe diese explizit hervor. Füge eine kurze Bewertung/Analyse des Inhalts an.\n\nNutzerprompt: {prompt}',
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
    return [f"{segment['start']:.2f}", f"{segment['end']:.2f}", segment['text']]


def format_segment_markdown(segment):
    """Pure function: Format a normalized segment for markdown."""
    return f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"


def save_to_csv(segments, output_file):
    """Save transcription segments to a CSV file (expects normalized segments)."""
    csv_file = os.path.splitext(output_file)[0] + '.csv'

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
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

            f.write("\n# Full Text:\n\n")
            f.write(f"{full_text}\n\n")
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

        # Generate summary if requested
        summary = ""
        if args.summarize:
            print("\nGenerating summary...")
            summary = summarize(full_text, args.prompt)
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