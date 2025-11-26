import argparse
import requests
import os
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"


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


def save_to_csv(segments, output_file, backend='whisper'):
    """Save transcription segments to a CSV file."""
    csv_file = os.path.splitext(output_file)[0] + '.csv'

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Start Time (s)', 'End Time (s)', 'Text'])

            if backend == 'faster-whisper':
                for segment in segments:
                    writer.writerow([f"{segment.start:.2f}", f"{segment.end:.2f}", segment.text])
            else:
                for segment in segments:
                    writer.writerow([f"{segment['start']:.2f}", f"{segment['end']:.2f}", segment['text']])

        print(f"CSV file saved to: {csv_file}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")


def save_to_markdown(segments, full_text, summary, output_file, backend='whisper'):
    """Save transcription to a markdown file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Transcription:\n\n")

            if backend == 'faster-whisper':
                for segment in segments:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            else:
                for segment in segments:
                    f.write(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n")

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
        if args.backend == 'faster-whisper':
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size_or_path=args.model)
            transcribe_fn = lambda fp: transcribe_file_faster_whisper(model, fp)
        elif args.backend == 'mlx-whisper':
            # mlx-whisper uses HuggingFace repo format
            model_name = f"mlx-community/whisper-{args.model}-mlx"
            transcribe_fn = lambda fp: transcribe_file_mlx_whisper(fp, model_name)
        else:
            import whisper
            model = whisper.load_model(args.model)
            transcribe_fn = lambda fp: transcribe_file_whisper(model, fp)

        # Transcribe once and store results
        full_text, segments, info = transcribe_fn(args.file_path)

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

        # Save both markdown and CSV versions
        save_to_markdown(segments, full_text, summary, output_file, args.backend)
        save_to_csv(segments, output_file, args.backend)

    except ImportError as e:
        print(f"Error: Required module not found. {e}")
        print(f"Please install the required packages for the {args.backend} backend.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()