"""Define CLI commands for the workspace, data pipeline, and training.

Model and dataset modules load after workspace selection so ``--workdir`` can
set the Hugging Face cache before those libraries are imported.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from medstt.eval.evaluate import evaluate_model
from medstt.io import read_records
from medstt.sentences.generate import USER_PROMPT, GenerationSettings, generate_sentences
from medstt.words.nomenclature import extract_words
from medstt.workspace import Workspace, WorkspaceError, pick

app = typer.Typer(
    help="Medical speech-to-text: build synthetic datasets and fine-tune ASR models.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
words_app = typer.Typer(help="Phase 0: build word lists.", no_args_is_help=True)
sentences_app = typer.Typer(help="Phase 1: generate sentences from words.", no_args_is_help=True)
tts_app = typer.Typer(help="Phase 2: synthesize audio datasets.", no_args_is_help=True)
dataset_app = typer.Typer(help="Import, finalize and inspect audio datasets.", no_args_is_help=True)
model_app = typer.Typer(help="Download models for offline use.", no_args_is_help=True)
app.add_typer(words_app, name="words")
app.add_typer(sentences_app, name="sentences")
app.add_typer(tts_app, name="tts")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")

log = logging.getLogger("medstt")

WorkdirOption = Annotated[
    Path | None,
    typer.Option(
        "--workdir",
        "-w",
        help="Workspace directory (default: $MEDSTT_WORKDIR, then the current directory).",
    ),
]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Configure application logging for the selected verbosity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The OpenAI client logs every HTTP request at INFO level.
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _open(workdir: Path | None) -> Workspace:
    """Open the selected workspace and activate its Hugging Face cache."""
    try:
        ws = Workspace.open(workdir)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error), param_hint="--workdir") from error
    ws.activate()
    return ws


def _api_key(section: dict, api_key: str | None, api_key_env: str | None) -> str:
    """Choose the LLM key from the CLI, environment, or workspace settings."""
    if api_key:
        return api_key
    env_name = pick(api_key_env, section, "api_key_env")
    from_env = os.environ.get(env_name) if env_name else None
    # Some local endpoints require a nonempty placeholder key.
    return from_env or section.get("api_key") or "EMPTY"


def _output_path(ws: Workspace, value: str, base: str, suffix: str = "") -> Path:
    """Place a relative output name under its workspace directory."""
    path = Path(value)
    if suffix and not path.suffix:
        path = path.with_suffix(suffix)
    return path if path.is_absolute() else ws.subdir(base) / path


# Workspace


@app.command()
def init(
    path: Annotated[Path | None, typer.Argument(help="Directory to initialize.")] = None,
) -> None:
    """Create workspace folders and default settings without replacing existing settings."""
    ws = Workspace.init(path)
    typer.echo(f"Workspace ready: {ws.root}\nEdit {ws.root / 'medstt.yaml'} to configure it.")


# Medical terms


@words_app.command("from-nomenclature")
def words_from_nomenclature(
    nomenclature: Annotated[Path, typer.Argument(help="Nomenclature file (in nomenclature/).")],
    output: Annotated[str, typer.Option("--output", "-o", help="Name of the word list.")],
    workdir: WorkdirOption = None,
) -> None:
    """[TODO] Extract medical terms from a nomenclature file."""
    ws = _open(workdir)
    try:
        extract_words(
            ws.resolve(nomenclature, "nomenclature"), _output_path(ws, output, "words", ".csv")
        )
    except NotImplementedError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error


# Sentences


@sentences_app.command("generate")
def sentences_generate(
    words: Annotated[
        Path, typer.Argument(help="Word list: txt/csv/tsv/jsonl/parquet (in words/).")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Sentence set name (in sentences/).")
    ],
    column: Annotated[
        str, typer.Option(help="Word column (txt files: one word per line).")
    ] = "word",
    per_word: Annotated[
        int | None, typer.Option(help="Sentences per term (overrides sentences.per_word).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only process the first N words.")] = None,
    base_url: Annotated[str | None, typer.Option(help="OpenAI-compatible endpoint.")] = None,
    model: Annotated[str | None, typer.Option(help="LLM model name.")] = None,
    api_key: Annotated[str | None, typer.Option(help="LLM API key.")] = None,
    api_key_env: Annotated[str | None, typer.Option(help="Env var holding the API key.")] = None,
    concurrency: Annotated[int | None, typer.Option(help="Parallel requests.")] = None,
    min_words: Annotated[int | None, typer.Option()] = None,
    max_words: Annotated[int | None, typer.Option()] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(
            help="User prompt template; placeholders: {word}, {domain}, "
            "{min_words}, {max_words} and any column of the word list."
        ),
    ] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Generate the configured number of French sentences per term; resume by ID."""
    ws = _open(workdir)
    llm, conf = ws.section("llm"), ws.section("sentences")
    words_path = ws.resolve(words, "words")
    records = read_records(words_path, txt_column=column)
    if limit is not None:
        records = records[:limit]
    if records and column not in records[0]:
        raise typer.BadParameter(
            f"Column {column!r} not found in {words_path}.", param_hint="--column"
        )

    sentences_per_word = pick(per_word, conf, "per_word", 1)
    if (
        not isinstance(sentences_per_word, int)
        or isinstance(sentences_per_word, bool)
        or sentences_per_word < 1
    ):
        raise typer.BadParameter(
            "sentences.per_word must be a positive integer.", param_hint="--per-word"
        )

    settings = GenerationSettings(
        base_url=pick(base_url, llm, "base_url"),
        api_key=_api_key(llm, api_key, api_key_env),
        model=pick(model, llm, "model"),
        domain=conf.get("domain", "pratique clinique hospitalière"),
        min_words=pick(min_words, conf, "min_words", 8),
        max_words=pick(max_words, conf, "max_words", 20),
        temperature=llm.get("temperature", 0.8),
        max_tokens=llm.get("max_tokens", 200),
        concurrency=pick(concurrency, llm, "concurrency", 16),
        timeout=llm.get("timeout", 120),
        max_attempts=conf.get("max_attempts", 4),
        per_word=sentences_per_word,
        user_prompt=prompt_file.read_text(encoding="utf-8") if prompt_file else USER_PROMPT,
        extra_body=llm.get("extra_body") or None,
    )
    if not settings.base_url or not settings.model:
        raise typer.BadParameter("Set llm.base_url and llm.model in medstt.yaml or via options.")
    output_path = _output_path(ws, output, "sentences", ".jsonl")
    ok, failed = generate_sentences(records, column, output_path, settings)
    typer.echo(f"{ok} sentences written to {output_path} ({failed} failed, rerun to retry).")
    if failed:
        raise typer.Exit(code=1)


# Synthetic audio


@tts_app.command("synthesize")
def tts_synthesize(
    sentences: Annotated[Path, typer.Argument(help="Sentence set: jsonl/csv/txt (in sentences/).")],
    output: Annotated[str, typer.Option("--output", "-o", help="Dataset name (in datasets/).")],
    text_column: Annotated[str, typer.Option(help="Sentence column.")] = "text",
    min_duration: Annotated[float | None, typer.Option()] = None,
    max_duration: Annotated[float | None, typer.Option()] = None,
    shard_index: Annotated[
        int | None, typer.Option(help="Shard to process (default: $SLURM_ARRAY_TASK_ID or 0).")
    ] = None,
    num_shards: Annotated[
        int | None, typer.Option(help="Shard count (default: $SLURM_ARRAY_TASK_COUNT or 1).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only process the first N sentences.")] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Write one male and one female Qwen recording per sentence."""
    ws = _open(workdir)
    backends = import_module("medstt.tts.backends")
    synthesis = import_module("medstt.tts.synthesize")
    conf = ws.section("tts")
    records = read_records(ws.resolve(sentences, "sentences"), txt_column=text_column)
    if limit is not None:
        records = records[:limit]
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if num_shards is None:
        num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    tts = backends.QwenVoiceDesign(
        voice_prompts=conf.get("voice_prompts", backends.DEFAULT_VOICE_PROMPTS),
        hf_token=os.environ.get("HF_TOKEN") or conf.get("hf_token"),
    )
    settings = synthesis.SynthesisSettings(
        sampling_rate=conf.get("sampling_rate", 16000),
        min_duration=pick(min_duration, conf, "min_duration", 1.0),
        max_duration=pick(max_duration, conf, "max_duration", 20.0),
        shard_index=shard_index,
        num_shards=num_shards,
    )
    output_dir = _output_path(ws, output, "datasets")
    written, rejected, failed = synthesis.synthesize_dataset(
        records, text_column, output_dir, tts, settings
    )
    typer.echo(
        f"Shard {shard_index}: {written} written, {rejected} rejected (duration), "
        f"{failed} failed (rerun to retry)."
    )
    if failed:
        raise typer.Exit(code=1)
    if num_shards == 1:
        total = synthesis.finalize_dataset(output_dir)
        typer.echo(f"Dataset {output_dir} finalized with {total} samples.")
    else:
        typer.echo(f"When all shards are done: medstt dataset finalize {output}")


# Datasets


@dataset_app.command("finalize")
def dataset_finalize(
    name: Annotated[str, typer.Argument(help="Dataset name (in datasets/).")],
    workdir: WorkdirOption = None,
) -> None:
    """Merge successful audio records from all shards into ``metadata.csv``."""
    ws = _open(workdir)
    synthesis = import_module("medstt.tts.synthesize")
    path = ws.resolve(name, "datasets")
    typer.echo(f"{path}: {synthesis.finalize_dataset(path)} samples.")


@dataset_app.command("import")
def dataset_import(
    source: Annotated[str, typer.Argument(help="Hugging Face hub id or local dataset path.")],
    output: Annotated[str, typer.Option("--output", "-o", help="Dataset name (in datasets/).")],
    config: Annotated[str | None, typer.Option(help="Dataset configuration (e.g. 'fr').")] = None,
    split: Annotated[str, typer.Option()] = "train",
    audio_column: Annotated[str, typer.Option()] = "audio",
    text_column: Annotated[str, typer.Option(help="e.g. 'sentence', 'transcription'.")] = "text",
    max_samples: Annotated[int | None, typer.Option()] = None,
    streaming: Annotated[
        bool, typer.Option(help="Stream instead of downloading everything.")
    ] = True,
    workdir: WorkdirOption = None,
) -> None:
    """Copy a local or Hub ASR dataset into the workspace as an AudioFolder."""
    ws = _open(workdir)
    hub = import_module("medstt.data.hub")
    output_dir = _output_path(ws, output, "datasets")
    count = hub.import_dataset(
        source,
        output_dir,
        config=config,
        split=split,
        audio_column=audio_column,
        text_column=text_column,
        max_samples=max_samples,
        streaming=streaming,
    )
    typer.echo(f"{count} samples imported into {output_dir}")


@dataset_app.command("info")
def dataset_info(
    names: Annotated[list[str], typer.Argument(help="Dataset names or paths.")],
    workdir: WorkdirOption = None,
) -> None:
    """Show each dataset's sample count and up to three transcripts."""
    ws = _open(workdir)
    loading = import_module("medstt.data.loading")
    for name in names:
        ds = loading.load_dataset_spec(ws, loading.DatasetSpec(path=name))
        typer.echo(f"== {name}: {len(ds)} samples")
        for row in ds.select(range(min(3, len(ds)))):
            typer.echo(f"   {row['text']}")


# Model downloads


@model_app.command("download")
def model_download(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face model id.")],
    revision: Annotated[str | None, typer.Option()] = None,
    local_name: Annotated[
        str | None, typer.Option(help="Save under models/<name> instead of the HF cache.")
    ] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Download model files into the HF cache or ``models/<local_name>``."""
    ws = _open(workdir)
    hub = import_module("medstt.data.hub")
    local_dir = ws.subdir("models") / local_name if local_name else None
    typer.echo(f"Downloaded to {hub.download_model(repo_id, revision, local_dir)}")


# Training and evaluation


@app.command()
def train(
    resume: Annotated[
        bool, typer.Option(help="Resume from the last checkpoint of the run.")
    ] = False,
    workdir: WorkdirOption = None,
) -> None:
    """Fine-tune all Cohere Transcribe weights using ``medstt.yaml``."""
    ws = _open(workdir)
    training = import_module("medstt.train.run")
    output = training.train(ws, resume)
    typer.echo(f"Trained model saved to {output}")


@app.command()
def evaluate(
    model: Annotated[str, typer.Option("--model", "-m", help="Run dir, local path or hub id.")],
    datasets: Annotated[list[str], typer.Option("--dataset", "-d", help="Repeatable.")],
    workdir: WorkdirOption = None,
) -> None:
    """[TODO] Evaluate a trained model outside the training run."""
    ws = _open(workdir)
    try:
        evaluate_model(ws, model, datasets)
    except NotImplementedError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
