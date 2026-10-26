"""CPU checks for Cohere Transcribe's full-weight training path."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoProcessor,
    CohereAsrConfig,
    CohereAsrFeatureExtractor,
    CohereAsrForConditionalGeneration,
)

from medstt.train.cohere import CohereCollator, build_trainer
from medstt.train.config import TrainConfig
from medstt.train.run import train as run_training
from medstt.workspace import Workspace


class ToyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [[10 + ord(char) % 40 for char in item] for item in text]}
        return {"input_ids": [10 + ord(char) % 40 for char in text]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["test"] * len(ids)


class ToyProcessor:
    tokenizer = ToyTokenizer()
    feature_extractor = CohereAsrFeatureExtractor(dither=0)

    def get_decoder_prompt_ids(self, language):
        assert language == "fr"
        return [3, 4, 5]

    def save_pretrained(self, output_dir):
        (Path(output_dir) / "toy_processor.txt").write_text("fr", encoding="utf-8")


def tiny_model():
    config = CohereAsrConfig(
        encoder_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "subsampling_conv_channels": 8,
            "num_mel_bins": 128,
            "max_position_embeddings": 256,
        },
        vocab_size=80,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    return CohereAsrForConditionalGeneration(config)


def test_french_prompt_and_full_backward():
    model = tiny_model()
    batch = CohereCollator(ToyProcessor(), 16000, 1)(
        [
            {
                "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                "text": "bonjour",
            }
        ]
    )

    assert batch["decoder_input_ids"][0, :4].tolist() == [1, 3, 4, 5]
    assert batch["labels"][0, :3].tolist() == [-100, -100, -100]
    assert batch["labels"][0, -1].item() == 2
    loss = model(**batch).loss
    loss.backward()
    assert np.isfinite(loss.detach().item())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "encoder" in name
    )
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "decoder" in name
    )


def test_evaluation_generation_does_not_receive_reference(tmp_path):
    model = tiny_model()
    processor = ToyProcessor()
    dataset = Dataset.from_dict(
        {
            "audio": [
                {"array": np.zeros(16000, dtype=np.float32).tolist(), "sampling_rate": 16000}
            ],
            "text": ["bonjour"],
        }
    )
    config = TrainConfig.from_dict(
        {
            "run_name": "smoke",
            "data": {"train": ["unused"], "max_label_tokens": 16},
            "training": {
                "bf16": False,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
            },
        }
    )
    with (
        patch.object(AutoProcessor, "from_pretrained", return_value=processor),
        patch.object(CohereAsrForConditionalGeneration, "from_pretrained", return_value=model),
    ):
        trainer, _ = build_trainer(config, dataset, dataset, tmp_path)

    batch = trainer.data_collator([dataset[0]])
    model.generate = Mock(return_value=torch.tensor([[3, 4, 5, 12, 2]]))
    loss, generated, labels = trainer.prediction_step(model, batch, False)
    generation_inputs = model.generate.call_args.kwargs

    assert loss.isfinite()
    assert generated.shape == (1, 5)
    assert labels.shape[1] > 3
    assert generation_inputs["decoder_input_ids"].tolist() == [[3, 4, 5]]
    assert "labels" not in generation_inputs


def test_one_cpu_training_step_saves_full_model_and_run_files(tmp_path):
    ws = Workspace.init(tmp_path / "ws")
    dataset_dir = ws.root / "datasets" / "tiny"
    (dataset_dir / "audio").mkdir(parents=True)
    sf.write(dataset_dir / "audio" / "sample.wav", np.zeros(16000), 16000)
    (dataset_dir / "metadata.csv").write_text(
        "file_name,text\naudio/sample.wav,bonjour\n", encoding="utf-8"
    )
    config_path = ws.root / "medstt.yaml"
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["train"] = {
        "run_name": "tiny-cpu",
        "data": {"train": ["tiny"], "eval_split": 0},
        "training": {
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "save_strategy": "no",
            "gradient_checkpointing": False,
            "bf16": False,
            "use_cpu": True,
            "dataloader_num_workers": 0,
            "disable_tqdm": True,
        },
    }
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")

    with (
        patch.object(AutoProcessor, "from_pretrained", return_value=ToyProcessor()),
        patch.object(
            CohereAsrForConditionalGeneration, "from_pretrained", return_value=tiny_model()
        ),
    ):
        output = run_training(Workspace.open(ws.root), resume=False)

    assert output == ws.root / "runs" / "tiny-cpu" / "final"
    assert (output / "config.json").is_file()
    assert list(output.glob("*.safetensors")) or list(output.glob("*.bin"))
    assert (output / "toy_processor.txt").read_text(encoding="utf-8") == "fr"
    assert (ws.root / "runs" / "tiny-cpu" / "train_results.json").is_file()
    assert (ws.root / "runs" / "tiny-cpu" / "train_config.yaml").is_file()
