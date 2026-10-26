"""Common interface used by speech generators and the dataset writer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSBackend(ABC):
    """Provide one waveform for a sentence and named voice."""

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Return mono or multichannel audio and its native sample rate."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the generator name stored with each audio sample."""
