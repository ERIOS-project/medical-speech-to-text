"""Save two voice recordings per sentence as a Hugging Face AudioFolder.

Each successful sentence/voice pair gets one WAV and a progress record.
Progress files allow interrupted or sharded runs to resume without duplicates;
``finalize_dataset`` merges them into ``metadata.csv``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from medstt.audio import peak_normalize, resample, to_mono
from medstt.io import append_jsonl, iter_jsonl, write_csv
from medstt.tts.base import TTSBackend

log = logging.getLogger(__name__)

PARTS_DIR = "_parts"
AUDIO_DIR = "audio"
METADATA_FILE = "metadata.csv"
VOICES = ("male", "female")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SynthesisSettings:
    """Output audio format, accepted duration, and current shard."""

    sampling_rate: int
    min_duration: float
    max_duration: float
    shard_index: int = 0
    num_shards: int = 1


def synthesize_dataset(
    sentences: list[dict[str, Any]],
    text_column: str,
    output_dir: Path,
    backend: TTSBackend,
    settings: SynthesisSettings,
) -> tuple[int, int, int]:
    """Generate this shard's missing sentence/voice pairs.

    Return counts of saved audio, duration rejections, and generation failures.
    Duration rejections are final; failures remain eligible for a later retry.
    """
    if not 0 <= settings.shard_index < settings.num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    audio_dir = output_dir / AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    part_file = output_dir / PARTS_DIR / f"part-{settings.shard_index:05d}.jsonl"
    rejected_file = output_dir / PARTS_DIR / f"rejected-{settings.shard_index:05d}.jsonl"
    # Read all shards so a rerun can use a different shard count.
    done = {
        record["id"]
        for progress in (output_dir / PARTS_DIR).glob("*.jsonl")
        for record in iter_jsonl(progress)
    }

    items = []
    for row, record in enumerate(sentences):
        text = str(record.get(text_column) or "").strip()
        if not text:
            continue
        sample_id = _UNSAFE_RE.sub("_", str(record.get("id") or f"{row:07d}"))
        for voice in VOICES:
            items.append((f"{sample_id}-{voice}", text, record, voice))
    items = items[settings.shard_index :: settings.num_shards]
    todo = [item for item in items if item[0] not in done]
    log.info(
        "Shard %d/%d: %d audio samples, %d left to synthesize.",
        settings.shard_index,
        settings.num_shards,
        len(items),
        len(todo),
    )

    def work(item):
        """Generate and resample one sentence/voice pair."""
        sample_id, text, record, voice = item
        try:
            audio, sr = backend.synthesize(text, voice)
        except Exception as error:  # noqa: BLE001 - left undone, retried on the next run
            log.warning("TTS failed for %s: %s", sample_id, error)
            return sample_id, text, record, voice, None
        audio = peak_normalize(resample(to_mono(audio), sr, settings.sampling_rate))
        return sample_id, text, record, voice, audio

    written = rejected = failed = 0
    for sample_id, text, record, voice, audio in tqdm(
        map(work, todo), total=len(todo), desc="tts", unit="utt"
    ):
        if audio is None:
            failed += 1
            continue
        duration = len(audio) / settings.sampling_rate
        if not settings.min_duration <= duration <= settings.max_duration:
            append_jsonl(rejected_file, [{"id": sample_id, "text": text, "duration": duration}])
            rejected += 1
            continue
        file_name = f"{AUDIO_DIR}/{sample_id}.wav"
        sf.write(output_dir / file_name, audio, settings.sampling_rate, subtype="PCM_16")
        reserved = {
            "id",
            "file_name",
            "text",
            text_column,
            "duration",
            "speaker",
            "tts",
            "source_id",
        }
        extra = {k: v for k, v in record.items() if k not in reserved}
        append_jsonl(
            part_file,
            [
                {
                    "id": sample_id,
                    "file_name": file_name,
                    "text": text,
                    "duration": round(duration, 3),
                    "speaker": voice,
                    "tts": backend.name,
                    "source_id": sample_id.removesuffix(f"-{voice}"),
                    **extra,
                }
            ],
        )
        written += 1
    return written, rejected, failed


def finalize_dataset(output_dir: Path) -> int:
    """Write ``metadata.csv`` from successful shard records and return its row count."""
    records: dict[str, dict[str, Any]] = {}
    for part in sorted((output_dir / PARTS_DIR).glob("part-*.jsonl")):
        for record in iter_jsonl(part):
            records[record["id"]] = record
    if not records:
        raise ValueError(f"No synthesized samples found in {output_dir / PARTS_DIR}.")
    ordered = [records[key] for key in sorted(records)]
    write_csv(output_dir / METADATA_FILE, ordered, first_columns=("file_name", "text"))
    return len(ordered)
