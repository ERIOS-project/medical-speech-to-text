# 02 — Generate sentences

Input: a word list under `<workdir>/words/`, normally the JSONL written by [`medstt words build`](01-words.md) (one object per word with its `code`, `label`, and `definition`). A `.txt` file with one term per line, or CSV, TSV, JSON, and Parquet with a `word` column, also work. An LLM served through an OpenAI-compatible API writes a clinical sentence for each term.

```bash
medstt sentences generate medical.jsonl -o medical
```

The command uses `llm.base_url` and `llm.model` from [`medstt.yaml`](00-init.md); configure them for your LLM server before running it. The `sentences` section controls context, length, retry count, and `per_word`. Command-line options such as `--base-url`, `--model`, and `--per-word N` override the settings when needed. Use `--limit N` for the first N input rows and `--concurrency` for parallel requests.

`sentences.per_word` is the number of sentences requested **for each term**. Its default is `1`: a list of 100 terms requests 100 sentences. With `per_word: 3`, the command makes three separate requests per term and aims to write 300 sentences. The first term gets IDs `000000-0`, `000000-1`, and `000000-2`. The code does not check whether the generated sentences differ. If you increase `per_word` and rerun the command, existing IDs are skipped and only the additional sentences are generated; decreasing it does not remove existing output.

Each reply is cleaned and checked: the term must occur regardless of case and accents, and the sentence must have between `min_words` and `max_words` words. Generation is retried up to `max_attempts` times. Failed items are not written and will be tried again on the next run.

Output: `<workdir>/sentences/medical.jsonl`, **not a `.txt` file**. Each JSONL line is a JSON object with `id`, `word`, `text` (the generated sentence), `llm`, and the other input columns. Existing records are skipped when the command is rerun. For example:

```json
{"id":"000000-0","word":"péricardite","text":"Le patient présente une péricardite aiguë depuis ce matin.","llm":"<model-name>"}
```

## Prompt

The default user prompt asks for a clinical dictation sentence in `sentences.domain` that contains the word written exactly as given, tells the LLM which label the word comes from and, when the word list has one, the concept definition, and asks it to use the word together with the neighbouring words of the label it belongs with (« acné varioliforme » rather than « varioliforme » alone) without copying a long label verbatim. For a row of `words build` it reads, for example:

```text
Rédige une phrase naturelle en français, comme lors d'une dictée clinique en pratique clinique hospitalière, contenant le mot « varioliforme » écrit exactement ainsi.
Ce mot vient du libellé « Acné varioliforme ».
Emploie « varioliforme » avec les mots du libellé avec lesquels il forme une expression habituelle (le nom complet d'une maladie, d'un signe ou d'un acte), sans recopier le libellé entier s'il n'est pas naturel à l'oral. La phrase doit compter entre 8 et 20 mots.
```

With a bare word list (no `label` or `definition` column) the context line is simply omitted. Only the word itself is checked in the reply, so the LLM stays free to shorten or rephrase the label.

Use `--prompt-file` to supply a custom user prompt. It can reference `{word}`, `{label}` (the word itself when the list has no label), `{definition}` (empty when missing), `{context}` (the label and definition lines above, or empty), `{domain}`, `{min_words}`, `{max_words}`, and any column from the input file; blank lines left by empty placeholders are dropped. You can also write sentences manually: a `.txt` file with one sentence per line, or a CSV/JSONL file with a `text` column (changeable with `--text-column` during synthesis).

See [Slurm instructions](slurm.md#sentences). Next: [synthesize audio](03-tts.md).
