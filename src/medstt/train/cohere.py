"""Build the French audio batches and trainer for Cohere Transcribe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    CohereAsrForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from medstt.audio import decode_audio
from medstt.metrics import text_metrics
from medstt.train.common import log_trainable_parameters, training_arguments
from medstt.train.config import MODEL_ID, TrainConfig

LANGUAGE = "fr"


def transcript_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize a transcript and ensure it ends with the EOS token."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids or ids[-1] != tokenizer.eos_token_id:
        ids = [*ids, tokenizer.eos_token_id]
    return ids


@dataclass
class CohereCollator:
    """Prepare audio and decoder inputs while masking French prompt tokens."""

    processor: Any
    sampling_rate: int
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate short audio samples and their transcript labels into tensors."""
        arrays = [decode_audio(row["audio"], self.sampling_rate) for row in features]
        audio = self.processor.feature_extractor(
            arrays, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        if len(audio["audio_chunk_index"]) != len(features):
            raise ValueError(
                "Training requires short utterances; split audio longer than 30 seconds."
            )

        prompt = self.processor.get_decoder_prompt_ids(language=LANGUAGE)
        pad_id = self.processor.tokenizer.pad_token_id
        sequences = [
            prompt + transcript_ids(self.processor.tokenizer, row["text"]) for row in features
        ]
        width = max(map(len, sequences))
        decoder_ids = torch.full((len(features), width), pad_id, dtype=torch.long)
        decoder_mask = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            size = len(sequence)
            decoder_ids[row, :size] = torch.tensor(
                [self.decoder_start_token_id, *sequence[:-1]], dtype=torch.long
            )
            decoder_mask[row, :size] = 1
            labels[row, len(prompt) : size] = torch.tensor(sequence[len(prompt) :])

        return {
            "input_features": audio["input_features"],
            "attention_mask": audio["attention_mask"],
            "decoder_input_ids": decoder_ids,
            "decoder_attention_mask": decoder_mask,
            "labels": labels,
        }


def build_trainer(cfg: TrainConfig, train_ds, eval_ds, run_dir: Path):
    """Load Cohere Transcribe and return its trainer and processor."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.use_cache = False

    log_trainable_parameters(model)

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Cohere Transcribe tokenizer must define PAD and EOS tokens.")
    decoder_start = model.config.decoder_start_token_id or tokenizer.bos_token_id
    if decoder_start is None:
        raise ValueError("Cohere Transcribe must define a decoder start or BOS token.")

    prompt_ids = processor.get_decoder_prompt_ids(language=LANGUAGE)
    max_target_tokens = cfg.data.max_label_tokens

    def short_enough(texts: list[str]) -> list[bool]:
        """Keep transcripts that fit within the configured token limit."""
        return [len(transcript_ids(tokenizer, text)) <= max_target_tokens for text in texts]

    train_ds = train_ds.filter(short_enough, input_columns="text", batched=True)
    if not len(train_ds):
        raise ValueError("No training transcripts remain after max_label_tokens filtering.")
    if eval_ds is not None:
        eval_ds = eval_ds.filter(short_enough, input_columns="text", batched=True)
        if not len(eval_ds):
            raise ValueError("No evaluation transcripts remain after max_label_tokens filtering.")

    def compute_metrics(prediction) -> dict[str, float]:
        """Decode predictions and labels before calculating WER and CER."""
        pred_ids = prediction.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(
            prediction.label_ids == -100, tokenizer.pad_token_id, prediction.label_ids
        )
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return text_metrics(predictions, references)

    class CohereTrainer(Seq2SeqTrainer):
        """Generate evaluation text from the French prompt alone."""

        def prediction_step(
            self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs
        ):
            """Calculate loss with labels, then generate without the reference text."""
            if prediction_loss_only or not self.args.predict_with_generate:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = model(**inputs).loss.detach().mean()
                # Decoder inputs contain the reference; generation must use only the prompt.
                prompt = torch.tensor(prompt_ids, device=inputs["input_features"].device)
                prompt = prompt.unsqueeze(0).expand(inputs["input_features"].shape[0], -1)
                generated = self.model.generate(
                    input_features=inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=prompt,
                    max_new_tokens=max_target_tokens,
                )
            return loss, generated, inputs["labels"]

    args = training_arguments(
        Seq2SeqTrainingArguments,
        cfg,
        run_dir,
        has_eval=eval_ds is not None,
        defaults={"learning_rate": 1e-5, "predict_with_generate": True},
    )
    trainer = CohereTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CohereCollator(
            processor=processor,
            sampling_rate=cfg.data.sampling_rate,
            decoder_start_token_id=decoder_start,
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    return trainer, processor
