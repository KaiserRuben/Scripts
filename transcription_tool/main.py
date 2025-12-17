"""Transcription pipeline - functional composition with graceful degradation."""
import argparse
import sys
from datetime import datetime
from toolz import pipe
from returns.result import Success, Failure
from returns.pipeline import is_successful
from loguru import logger

from models import PipelineResult, AnalysisResult, Extraction
from config import Config
from backends import get_backend
from pipeline import (
    load_audio, perform_diarization, no_diarization,
    transcribe_segments, merge_consecutive, filter_artifacts,
    segments_to_text, analyze,
)
from output import save_folder


def parse_args() -> Config:
    """Parse CLI arguments into Config."""
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    parser.add_argument("file_path", help="Path to audio file")
    parser.add_argument("--backend", choices=["faster-whisper", "mlx-whisper", "whisper", "whisper-cpp", "glm-asr"],
                        default="whisper-cpp")
    parser.add_argument("--model", help="Model name (uses backend default if not specified)")
    parser.add_argument("--output", default="transcriptions", help="Output folder base path")
    parser.add_argument("--prompt", help="Custom prompt for analysis")
    parser.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--no-summarize", action="store_true", help="Disable LLM analysis")
    args = parser.parse_args()

    return Config(
        file_path=args.file_path,
        backend=args.backend,
        model=args.model,
        output=args.output,
        prompt=args.prompt,
        diarize=not args.no_diarize,
        summarize=not args.no_summarize,
    )


def run_pipeline(config: Config) -> Failure | Success[PipelineResult]:
    """Execute the transcription pipeline with graceful degradation."""
    warnings: list[str] = []

    # === CRITICAL: Backend + Model (no fallback) ===
    backend_result = get_backend(config.backend)
    if not is_successful(backend_result):
        return Failure(backend_result.failure())
    backend = backend_result.unwrap()

    model_result = backend.load_model(config.effective_model)
    if not is_successful(model_result):
        return Failure(f"Model load failed: {model_result.failure()}")
    model = model_result.unwrap()

    # === CRITICAL: Audio loading (no fallback) ===
    logger.info(f"Loading audio: {config.file_path}")
    audio_result = load_audio(config.file_path, config.sample_rate)
    if not is_successful(audio_result):
        return Failure(f"Audio load failed: {audio_result.failure()}")
    audio = audio_result.unwrap()
    logger.info(f"Audio loaded: {audio.duration:.1f}s")

    # === DEGRADABLE: Diarization → fallback to single speaker ===
    if config.diarize:
        speaker_result = perform_diarization(config.hf_token)(audio)
        if is_successful(speaker_result):
            speaker_segments = speaker_result.unwrap()
        else:
            logger.warning(f"Diarization failed: {speaker_result.failure()}")
            logger.warning("Falling back to single-speaker mode")
            warnings.append(f"Diarization failed ({speaker_result.failure()}), used single-speaker fallback")
            speaker_segments = no_diarization(audio).unwrap()
    else:
        speaker_segments = no_diarization(audio).unwrap()

    # === CRITICAL: Transcription (no fallback) ===
    logger.info(f"Transcribing {len(speaker_segments)} segments...")
    transcribe_result = transcribe_segments(backend, model, audio, speaker_segments)
    if not is_successful(transcribe_result):
        return Failure(f"Transcription failed: {transcribe_result.failure()}")

    # Process: merge + filter
    segments = pipe(
        transcribe_result.unwrap(),
        merge_consecutive(max_gap=config.max_gap_merge),
        filter_artifacts(
            min_duration=config.min_segment_duration,
            min_text_length=config.min_text_length,
        ),
    )

    if not segments:
        return Failure("No segments after processing - audio may be empty or unrecognizable")

    speakers = tuple(sorted(set(s.speaker for s in segments if s.speaker)))

    # === DEGRADABLE: LLM Analysis → fallback to empty extraction ===
    if config.summarize:
        analysis_result = analyze(config, segments)
        if is_successful(analysis_result):
            analysis = analysis_result.unwrap()
        else:
            logger.warning(f"LLM analysis failed: {analysis_result.failure()}")
            warnings.append(f"LLM analysis failed ({analysis_result.failure()})")
            analysis = AnalysisResult(
                summary="[Analysis unavailable - see transcript]",
                extraction=Extraction()
            )
    else:
        analysis = AnalysisResult(summary="[Analysis disabled]", extraction=Extraction())

    # Build result
    return Success(PipelineResult(
        source_file=config.file_path,
        created=datetime.now(),
        duration_sec=audio.duration,
        backend=config.backend,
        model=config.effective_model,
        speakers=speakers,
        segments=segments,
        full_text=segments_to_text(segments),
        analysis=analysis,
        config={
            "diarize": config.diarize,
            "summarize": config.summarize,
            "prompt": config.prompt,
        },
        warnings=tuple(warnings),
    ))


def main():
    """Entry point."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{level}</level> | {message}", level="INFO")

    config = parse_args()
    logger.info(f"Backend: {config.backend}, Model: {config.effective_model}")
    logger.info(f"Diarization: {config.diarize}, Analysis: {config.summarize}")

    result = run_pipeline(config)

    match result:
        case Success(pipeline_result):
            # Show warnings if any
            if pipeline_result.warnings:
                print(f"\n{'='*60}")
                print("WARNINGS")
                print(f"{'='*60}")
                for w in pipeline_result.warnings:
                    print(f"  ⚠ {w}")

            # Save to folder
            save_result = save_folder(config.output, pipeline_result)

            if is_successful(save_result):
                folder = save_result.unwrap()
                logger.info(f"Output saved to: {folder}")

                # Print summary
                print(f"\n{'='*60}")
                print("SUMMARY")
                print(f"{'='*60}\n")
                print(pipeline_result.analysis.summary)

                # Print action items
                ext = pipeline_result.analysis.extraction
                if ext.asks or ext.commitments:
                    print(f"\n{'='*60}")
                    print("ACTION ITEMS")
                    print(f"{'='*60}\n")
                    for ask in ext.asks:
                        print(f"  → {ask.what}")
                    for c in ext.commitments:
                        print(f"  □ {c.who}: {c.what}")
            else:
                logger.error(f"Save failed: {save_result.failure()}")
                sys.exit(1)

        case Failure(error):
            logger.error(f"Pipeline failed: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
