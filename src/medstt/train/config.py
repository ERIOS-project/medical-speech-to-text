"""Parse and validate the ``train`` section of ``medstt.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medstt.data.loading import DatasetSpec

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


@dataclass
class DataConfig:
    """Training datasets, evaluation data, and transcript length limits."""

    train: list[DatasetSpec]
    eval: list[DatasetSpec] = field(default_factory=list)
    #: Fraction held out when no separate evaluation dataset is listed.
    eval_split: float = 0.0
    sampling_rate: int = 16000
    #: Longer transcripts are dropped; the EOS token counts toward this limit.
    max_label_tokens: int = 256


@dataclass
class TrainConfig:
    """Run name, data sources, random seed, and Transformers training options."""

    run_name: str
    data: DataConfig
    training: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        """Build settings from the workspace's ``train`` mapping."""
        raw = dict(raw)
        if not raw:
            raise ValueError("Missing `train` section in medstt.yaml.")
        if "model" in raw:
            raise ValueError(f"The model is fixed to {MODEL_ID}; remove the `model` section.")
        if "method" in raw:
            raise ValueError("Only full-weight training is supported; remove the `method` section.")
        data_raw = dict(raw.pop("data"))
        data = DataConfig(
            train=[DatasetSpec.parse(v) for v in data_raw.pop("train")],
            eval=[DatasetSpec.parse(v) for v in data_raw.pop("eval", None) or []],
            **data_raw,
        )
        config = cls(data=data, **raw)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject unsupported audio rates, split fractions, and transcript limits."""
        if not 0.0 <= self.data.eval_split < 1.0:
            raise ValueError("data.eval_split must be in [0, 1).")
        if self.data.sampling_rate != 16000:
            raise ValueError("Cohere Transcribe requires data.sampling_rate=16000.")
        if self.data.max_label_tokens < 1:
            raise ValueError("data.max_label_tokens must be positive.")
