"""Check the direct Qwen VoiceDesign call in the TTS environment."""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

try:
    from medstt.tts.backends import MODEL_ID, QwenVoiceDesign
except ModuleNotFoundError as error:
    if error.name != "qwen_tts":
        raise
    QwenVoiceDesign = None
    MODEL_ID = None


@pytest.mark.skipif(QwenVoiceDesign is None, reason="qwen-tts is only in the TTS environment")
def test_qwen_voice_design_uses_french_and_selected_voice():
    backend = QwenVoiceDesign.__new__(QwenVoiceDesign)
    backend.voice_prompts = {"male": "voix homme", "female": "voix femme"}
    backend.model = Mock()
    backend.model.generate_voice_design.return_value = ([np.zeros(100)], 24000)

    audio, rate = backend.synthesize("Bonjour docteur.", "female")

    backend.model.generate_voice_design.assert_called_once_with(
        text="Bonjour docteur.", language="French", instruct="voix femme"
    )
    assert rate == 24000 and audio.shape == (100,)


@pytest.mark.skipif(QwenVoiceDesign is None, reason="qwen-tts is only in the TTS environment")
def test_qwen_backend_loads_fixed_model_without_gpu(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    prompts = {"male": "voix homme", "female": "voix femme"}
    with (
        patch("medstt.tts.backends.torch.cuda.is_available", return_value=False),
        patch("medstt.tts.backends.Qwen3TTSModel.from_pretrained", return_value=Mock()) as load,
    ):
        backend = QwenVoiceDesign(prompts, hf_token="test-token")

    load.assert_called_once_with(MODEL_ID, device_map="cpu", dtype=torch.float32)
    assert backend.name == MODEL_ID
    assert backend.voice_prompts == prompts
    assert os.environ["HF_TOKEN"] == "test-token"


@pytest.mark.skipif(QwenVoiceDesign is None, reason="qwen-tts is only in the TTS environment")
def test_qwen_requires_both_voice_prompts():
    with pytest.raises(ValueError, match="male.*female"):
        QwenVoiceDesign({"male": "voix homme"})
