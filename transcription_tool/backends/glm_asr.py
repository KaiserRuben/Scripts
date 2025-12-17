"""GLM-ASR backend."""
from dataclasses import dataclass
from typing import Any
import torch
from returns.result import Result, Success, Failure, safe
from toolz import curry
from loguru import logger

from models import AudioData


@dataclass
class GLMModel:
    """Container for GLM-ASR model components."""
    model: Any
    tokenizer: Any
    feature_extractor: Any
    merge_factor: int
    device: str


def _get_audio_token_length(seconds: float, merge_factor: int) -> int:
    """Calculate audio token length for GLM-ASR."""
    def get_T_after_cnn(L_in: int) -> int:
        for padding, kernel_size, stride in [(1, 3, 1), (1, 3, 2)]:
            L_in = 1 + (L_in + 2 * padding - kernel_size) // stride
        return L_in

    mel_len = int(seconds * 100)
    audio_len = get_T_after_cnn(mel_len)
    tokens = (audio_len - merge_factor) // merge_factor + 1
    return min(tokens, 1500 // merge_factor)


@safe
def load_model(model_name: str) -> GLMModel:
    """Load GLM-ASR model components."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, WhisperFeatureExtractor

    if model_name == "nano-2512" or not model_name:
        model_name = "zai-org/GLM-ASR-Nano-2512"

    logger.info(f"[GLM-ASR] Loading model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.train(False)  # Set to inference mode

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor(
        chunk_length=30, feature_size=128, hop_length=160,
        n_fft=400, n_samples=480000, nb_max_frames=3000,
        padding_side="right", padding_value=0.0,
        return_attention_mask=False, sampling_rate=16000,
    )

    return GLMModel(model, tokenizer, feature_extractor, config.merge_factor, device)


@curry
def transcribe_segment(
    glm: GLMModel,
    audio: AudioData,
    start: float,
    end: float,
) -> Result[str, str]:
    """Transcribe audio segment with GLM-ASR."""
    try:
        duration = audio.duration
        if duration < 0.5:
            return Success("")

        # Extract mel features
        mel = glm.feature_extractor(
            audio.waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]

        num_tokens = _get_audio_token_length(duration, glm.merge_factor)

        # Build prompt
        tokens = []
        tokens += glm.tokenizer.encode("<|user|>\n<|begin_of_audio|>")
        audio_offset = len(tokens)
        tokens += [0] * num_tokens
        tokens += glm.tokenizer.encode("<|end_of_audio|><|user|>\nPlease transcribe this audio into text<|assistant|>\n")

        input_ids = torch.tensor([tokens], dtype=torch.long).to(glm.device)
        mel_tensor = mel.to(glm.device).to(torch.bfloat16)

        with torch.inference_mode():
            generated = glm.model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids),
                audios=mel_tensor,
                audio_offsets=[[audio_offset]],
                audio_length=[[num_tokens]],
                max_new_tokens=256,
                do_sample=False,
            )

        text = glm.tokenizer.decode(
            generated[0, len(tokens):].cpu().tolist(),
            skip_special_tokens=True,
        ).strip()

        return Success(text)
    except Exception as e:
        logger.warning(f"GLM-ASR segment failed: {e}")
        return Success("")


class _Backend:
    load_model = staticmethod(load_model)
    transcribe_segment = staticmethod(transcribe_segment)


backend = _Backend()
