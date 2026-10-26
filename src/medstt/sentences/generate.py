"""Generate French clinical sentences with an OpenAI-compatible LLM.

Each result is a JSONL row with a term, sentence, and stable ID. Existing IDs
are skipped when the command is rerun.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm import tqdm

from medstt.io import append_jsonl, iter_jsonl
from medstt.text import contains_term, count_words

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Tu rédiges des phrases cliniques réalistes pour entraîner un système de reconnaissance "
    "vocale médicale en français. Réponds uniquement par la phrase, sans guillemets, "
    "numérotation ni explication."
)

USER_PROMPT = (
    "Rédige une phrase naturelle en français, comme lors d'une dictée clinique "
    "en {domain}, contenant le terme « {word} » écrit exactement ainsi. "
    "La phrase doit compter entre {min_words} et {max_words} mots."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"^(sentence|phrase)\s*:\s*", re.IGNORECASE)
_QUOTES = "\"'«»“”‘’` "


@dataclass(frozen=True)
class GenerationSettings:
    """LLM connection, prompt, validation, and retry settings for one run."""

    base_url: str
    api_key: str
    model: str
    domain: str
    min_words: int
    max_words: int
    temperature: float
    max_tokens: int
    concurrency: int
    timeout: float
    max_attempts: int
    per_word: int
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = USER_PROMPT
    extra_body: dict[str, Any] | None = None


def clean_sentence(raw: str) -> str:
    """Extract the first sentence line from an LLM reply and remove wrappers."""
    text = _THINK_RE.sub("", raw or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    sentence = _LABEL_RE.sub("", lines[0]).strip(_QUOTES)
    return re.sub(r"\s+", " ", sentence).strip()


def validate_sentence(sentence: str, word: str, min_words: int, max_words: int) -> str | None:
    """Explain why a sentence fails the term or length check, if it does."""
    if not sentence:
        return "empty reply"
    if not contains_term(sentence, word):
        return "term missing"
    n_words = count_words(sentence)
    if not min_words <= n_words <= max_words:
        return f"{n_words} words (expected {min_words}-{max_words})"
    return None


def generate_sentences(
    words: list[dict[str, Any]],
    word_column: str,
    output_path: Path,
    settings: GenerationSettings,
) -> tuple[int, int]:
    """Generate ``per_word`` sentences per term, skipping IDs already in JSONL.

    Return the number written and the number still missing after retries.
    """
    done = {record["id"] for record in iter_jsonl(output_path)}
    jobs = []
    for row, record in enumerate(words):
        word = str(record.get(word_column) or "").strip()
        if not word:
            continue
        for k in range(settings.per_word):
            job_id = f"{row:06d}-{k}"
            if job_id not in done:
                jobs.append((job_id, word, record))
    log.info("%d sentences to generate (%d already done).", len(jobs), len(done))
    if not jobs:
        return 0, 0
    return asyncio.run(_run(jobs, word_column, output_path, settings))


async def _run(jobs, word_column, output_path, settings) -> tuple[int, int]:
    """Run queued LLM requests concurrently and append successful sentences."""
    client = AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout,
        max_retries=3,
    )
    semaphore = asyncio.Semaphore(settings.concurrency)
    write_lock = asyncio.Lock()
    progress = tqdm(total=len(jobs), desc="sentences", unit="sent")
    counts = {"ok": 0, "failed": 0}

    async def worker(job_id: str, word: str, record: dict[str, Any]) -> None:
        """Generate one sentence and record its result under the write lock."""
        async with semaphore:
            sentence = await _generate_one(client, word, record, settings)
        async with write_lock:
            if sentence is None:
                counts["failed"] += 1
            else:
                extra = {k: v for k, v in record.items() if k not in (word_column, "id", "text")}
                append_jsonl(
                    output_path,
                    [
                        {
                            "id": job_id,
                            "word": word,
                            "text": sentence,
                            "llm": settings.model,
                            **extra,
                        }
                    ],
                )
                counts["ok"] += 1
            progress.update(1)

    try:
        await asyncio.gather(*(worker(*job) for job in jobs))
    finally:
        progress.close()
        await client.close()
    return counts["ok"], counts["failed"]


async def _generate_one(client, word: str, record: dict[str, Any], settings) -> str | None:
    """Retry one prompt until it yields a valid sentence or attempts run out."""
    variables = {
        **{k: v for k, v in record.items() if isinstance(k, str)},
        "word": word,
        "domain": settings.domain,
        "min_words": settings.min_words,
        "max_words": settings.max_words,
    }
    messages = [
        {"role": "system", "content": settings.system_prompt.format_map(variables)},
        {"role": "user", "content": settings.user_prompt.format_map(variables)},
    ]
    reason = "no attempt"
    for attempt in range(settings.max_attempts):
        try:
            response = await client.chat.completions.create(
                model=settings.model,
                messages=messages,
                # Vary sampling on retries so an invalid reply is less likely to repeat.
                temperature=min(settings.temperature + 0.1 * attempt, 1.5),
                max_tokens=settings.max_tokens,
                extra_body=settings.extra_body or None,
            )
        except Exception as error:  # noqa: BLE001 - network errors are retried then reported
            reason = f"request failed: {error}"
            continue
        sentence = clean_sentence(response.choices[0].message.content or "")
        reason = validate_sentence(sentence, word, settings.min_words, settings.max_words)
        if reason is None:
            return sentence
    log.warning("Giving up on %r after %d attempts: %s", word, settings.max_attempts, reason)
    return None
