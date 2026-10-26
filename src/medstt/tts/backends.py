"""Generate French audio with Qwen3-TTS VoiceDesign's Python API."""

from __future__ import annotations

import os

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from medstt.tts.base import TTSBackend

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_VOICE_PROMPTS = {
    "male": (
        "Voix d'homme adulte francophone, naturelle, claire et posée, "
        "avec une diction médicale précise."
    ),
    "female": (
        "Voix de femme adulte francophone, naturelle, claire et posée, "
        "avec une diction médicale précise."
    ),
}


class QwenVoiceDesign(TTSBackend):
    """Use fixed male and female prompts with the Qwen VoiceDesign model."""

    def __init__(self, voice_prompts: dict[str, str], hf_token: str | None = None):
        """Validate voice prompts and load the model on GPU when available."""
        if (
            not isinstance(voice_prompts, dict)
            or set(voice_prompts) != {"male", "female"}
            or not all(isinstance(value, str) and value.strip() for value in voice_prompts.values())
        ):
            raise ValueError("tts.voice_prompts must define nonempty 'male' and 'female' prompts.")
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
        self.voice_prompts = voice_prompts

    @property
    def name(self) -> str:
        """Return the fixed Hugging Face model ID."""
        return MODEL_ID

    def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Generate one French recording with the selected voice prompt."""
        wavs, sampling_rate = self.model.generate_voice_design(
            text=text,
            language="French",
            instruct=self.voice_prompts[voice],
        )
        return np.asarray(wavs[0], dtype=np.float32), sampling_rate
