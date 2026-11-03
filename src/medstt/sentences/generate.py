"""Generate French clinical sentences with an OpenAI-compatible LLM.

Each result is a JSONL row with a word, sentence, and stable ID
(``<source>-<code>-<word>`` for words built from a nomenclature). Existing IDs
are skipped when the command is rerun.
"""

from __future__ import annotations

import asyncio
import logging
import random
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
    "Tu rédiges des phrases de parole médicale réalistes en français, telles qu'on les "
    "entend en {domain} : dictées de comptes rendus, courriers, transmissions, échanges "
    "entre soignants ou avec les patients. Elles servent à entraîner un système de "
    "reconnaissance vocale. Réponds uniquement par la phrase, sans guillemets, "
    "numérotation ni explication."
)

USER_PROMPT = (
    "Rédige une phrase contenant le mot « {word} » écrit exactement ainsi, "
    "majuscules comprises.\n"
    "{context}\n"
    "Emploie « {word} » avec les mots du libellé avec lesquels il forme une expression "
    "habituelle (le nom complet d'une maladie, d'un signe ou d'un acte), sans recopier "
    "le libellé entier s'il n'est pas naturel à l'oral.\n"
    "{constraints}\n"
    "La phrase doit compter environ {target_words} mots (entre {min_words} et {max_words})."
)

#: Default draws, one entry of each list per sentence, chosen from the sentence ID.
DEFAULT_SCENES = (
    "compte rendu d'hospitalisation dicté",
    "courrier au médecin traitant",
    "transmission infirmière",
    "observation aux urgences",
    "consultation de suivi",
    "staff ou réunion de concertation",
    "échange oral entre deux soignants",
    "compte rendu d'examen (imagerie, biologie, anatomopathologie)",
    "entretien avec le patient ou sa famille",
)
DEFAULT_FORMS = (
    "affirmation d'un diagnostic",
    "hypothèse à confirmer",
    "diagnostic écarté ou absent",
    "antécédent ancien",
    "question posée à un confrère",
    "consigne de traitement ou de surveillance",
    "résultat d'examen",
    "évolution sous traitement",
)
DEFAULT_SUBJECTS = (
    "le patient",
    "la patiente",
    "l'enfant",
    "une femme de soixante ans",
    "un homme jeune",
    "je, le soignant qui parle",
    "nous, l'équipe",
    "personne en particulier, formulation impersonnelle",
)
CONSTRAINT_SCENE = "Situation : {scene}."
CONSTRAINT_FORM = "La phrase exprime : {form}."
CONSTRAINT_SUBJECT = "Sujet de la phrase : {subject}."

#: Lines of ``{context}``, built from the ``label`` and ``definition`` columns when present.
CONTEXT_LABEL = "Ce mot vient du libellé « {label} »."
CONTEXT_DEFINITION = "Définition du concept : {definition}"

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
    #: Lists drawn from for every sentence; an empty list adds no constraint.
    scenes: tuple[str, ...] = DEFAULT_SCENES
    forms: tuple[str, ...] = DEFAULT_FORMS
    subjects: tuple[str, ...] = DEFAULT_SUBJECTS


def clean_sentence(raw: str) -> str:
    """Extract the first sentence line from an LLM reply and remove wrappers."""
    text = _THINK_RE.sub("", raw or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    sentence = _LABEL_RE.sub("", lines[0]).strip(_QUOTES)
    return re.sub(r"\s+", " ", sentence).strip()


def sentence_id(record: dict[str, Any], word: str, row: int, k: int) -> str:
    """Return a stable ID: ``<source>-<code>-<word>`` for a nomenclature word, else the row.

    Extra sentences of the same word (``per_word`` above 1) get a ``-2``, ``-3``, ...
    suffix, so the ID of the first sentence does not change when ``per_word`` grows.
    """
    source = str(record.get("source") or "").strip()
    code = str(record.get("code") or "").strip()
    base = f"{source}-{code}-{word}" if source and code else f"{row:06d}"
    return base if k == 0 else f"{base}-{k + 1}"


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
    wanted: set[str] = set()
    for row, record in enumerate(words):
        word = str(record.get(word_column) or "").strip()
        if not word:
            continue
        for k in range(settings.per_word):
            job_id = sentence_id(record, word, row, k)
            wanted.add(job_id)
            if job_id not in done:
                jobs.append((job_id, word, record))
    orphans = len(done - wanted)
    if orphans:
        log.warning(
            "%d sentences in %s have IDs absent from the current word list "
            "(words removed or renumbered); they are kept as they are.",
            orphans,
            output_path,
        )
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
            sentence = await _generate_one(client, job_id, word, record, settings)
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


def draw_constraints(job_id: str, settings: GenerationSettings) -> dict[str, Any]:
    """Draw the scene, form, subject, and target length of one sentence from its ID.

    The draw is deterministic, so a rerun asks for the same sentence, and two
    sentences of the same word (``per_word`` above 1) get different draws. Each
    request is independent for the LLM: varying the constraints is what keeps it
    from writing the same sentence shape every time.
    """
    rng = random.Random(job_id)
    scene = rng.choice(settings.scenes) if settings.scenes else ""
    form = rng.choice(settings.forms) if settings.forms else ""
    subject = rng.choice(settings.subjects) if settings.subjects else ""
    parts = [
        CONSTRAINT_SCENE.format(scene=scene) if scene else "",
        CONSTRAINT_FORM.format(form=form) if form else "",
        CONSTRAINT_SUBJECT.format(subject=subject) if subject else "",
    ]
    return {
        "scene": scene,
        "form": form,
        "subject": subject,
        "target_words": rng.randint(settings.min_words, settings.max_words),
        "constraints": " ".join(part for part in parts if part),
    }


def build_messages(
    word: str, record: dict[str, Any], settings: GenerationSettings, job_id: str = ""
) -> list[dict[str, str]]:
    """Fill the prompt templates for one word.

    Placeholders are ``word``, ``domain``, ``min_words``, ``max_words``, every column of
    the word record, ``context`` (the word's full label and the concept definition when
    the record has them, otherwise empty), and the draws of :func:`draw_constraints`:
    ``scene``, ``form``, ``subject``, ``target_words``, and ``constraints`` (the three
    first ones as a sentence). Blank lines left by empty placeholders are removed.
    """
    label = str(record.get("label") or "").strip()
    definition = str(record.get("definition") or "").strip()
    context = []
    if label and label.casefold() != word.casefold():
        context.append(CONTEXT_LABEL.format(label=label))
    if definition:
        context.append(CONTEXT_DEFINITION.format(definition=definition))
    variables = {
        **{k: v for k, v in record.items() if isinstance(k, str)},
        "word": word,
        "label": label or word,
        "definition": definition,
        "context": " ".join(context),
        **draw_constraints(job_id or word, settings),
        "domain": settings.domain,
        "min_words": settings.min_words,
        "max_words": settings.max_words,
    }
    return [
        {"role": "system", "content": _fill(settings.system_prompt, variables)},
        {"role": "user", "content": _fill(settings.user_prompt, variables)},
    ]


def _fill(template: str, variables: dict[str, Any]) -> str:
    """Format a template and drop the blank lines left by empty placeholders."""
    lines = [line.rstrip() for line in template.format_map(variables).splitlines()]
    return "\n".join(line for line in lines if line).strip()


async def _generate_one(
    client, job_id: str, word: str, record: dict[str, Any], settings
) -> str | None:
    """Retry one prompt until it yields a valid sentence or attempts run out."""
    messages = build_messages(word, record, settings, job_id)
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
