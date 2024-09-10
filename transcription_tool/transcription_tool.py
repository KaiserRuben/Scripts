import argparse

import requests
import whisper
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

def transcribe_file(file_path, model_name="large"):
    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Transcribe the video
    print(f"Transcribing file: {file_path}")
    result = model.transcribe(file_path)
    return result

def summarize(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": OLLAMA_MODEL,
        "prompt": f'Fasse alle Informationen aus folgendem Text kurz und strukturiert zusammen: \n"""\n{text}\n"""\n\nAchte darauf, dass alle wichtigen Informationen enthalten sind. Wenn Aufforderungen oder Fragen enthalten sind, hebe diese explizit hervor. Füge dann am Ende eine kurze Bewertung/Analyse des Inhalts hinzu.',
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=data, headers=headers)
    return response.json()['response']

def main():
    parser = argparse.ArgumentParser(description="Transcribe a file using Whisper")
    parser.add_argument("file_path", help="Path to the file")
    parser.add_argument("--model", default="large", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: large)")
    parser.add_argument("--output", help="Output file path (optional)")
    parser.add_argument("-s","--summarize", help="Summarize the transcription", default="true")

    args = parser.parse_args()

    t = transcribe_file(args.file_path, args.model)
    print(f"Transcription: {t['text']}", end="\n\n")

    s = ""
    if args.summarize == 'true':
        s = summarize(t['text'])
        print(f"Summary: {s}")

    # Determine output file name if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_file = f"transcriptions/{base_name}_transcription.md"
    else:
        output_file = args.output

    # Save the transcription to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Transcription:\n\n{t['text']}\n\n")
        if args.summarize:
            f.write(f"# Summary:\n\n{s}")

    print(f"Transcription saved to: {output_file}")



if __name__ == "__main__":
    main()
