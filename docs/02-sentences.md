# 02 — Generate sentences

Input: a term list under `<workdir>/words/`. The simplest format is a `.txt` file with one term per line; CSV, TSV, JSONL, JSON, and Parquet are also supported ([details](01-words.md)). An LLM served through an OpenAI-compatible API writes a clinical sentence for each term.

```bash
medstt sentences generate cardio.txt -o cardio
```

The command uses `llm.base_url` and `llm.model` from [`medstt.yaml`](00-init.md); configure them for your LLM server before running it. The system and user prompts generate French sentences. The `sentences` section controls context, length, retry count, and `per_word`. Command-line options such as `--base-url`, `--model`, and `--per-word N` override the settings when needed. Use `--limit N` for the first N input rows and `--concurrency` for parallel requests.

`sentences.per_word` is the number of sentences requested **for each term**. Its default is `1`: a list of 100 terms requests 100 sentences. With `per_word: 3`, the command makes three separate requests per term and aims to write 300 sentences. The first term gets IDs `000000-0`, `000000-1`, and `000000-2`. The code does not check whether the generated sentences differ. If you increase `per_word` and rerun the command, existing IDs are skipped and only the additional sentences are generated; decreasing it does not remove existing output.

Each reply is cleaned and checked: the term must occur regardless of case and accents, and the sentence must have between `min_words` and `max_words` words. Generation is retried up to `max_attempts` times. Failed items are not written and will be tried again on the next run.

Output: `<workdir>/sentences/cardio.jsonl`, **not a `.txt` file**. Each JSONL line is a JSON object with `id`, `word`, `text` (the generated sentence), `llm`, and the other input columns. Existing records are skipped when the command is rerun. For example:

```json
{"id":"000000-0","word":"péricardite","text":"Le patient présente une péricardite aiguë depuis ce matin.","llm":"<model-name>"}
```

Use `--prompt-file` to supply a custom user prompt. It can reference `{word}`, `{domain}`, `{min_words}`, `{max_words}`, and columns from the input file. You can also write sentences manually: a `.txt` file with one sentence per line, or a CSV/JSONL file with a `text` column (changeable with `--text-column` during synthesis).

See [Slurm instructions](slurm.md#sentences). Next: [synthesize audio](03-tts.md).
