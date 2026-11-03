# 02 — Generate sentences

Input: a word list under `<workdir>/words/`, normally the JSONL written by [`medstt words build`](01-words.md) (one object per word with its `code`, `label`, and `definition`). A `.txt` file with one term per line, or CSV, TSV, JSON, and Parquet with a `word` column, also work. An LLM served through an OpenAI-compatible API writes a clinical sentence for each term.

```bash
medstt sentences generate                       # words/<words.output>.jsonl -> sentences/<sentences.output>.jsonl
medstt sentences generate sample.jsonl -o test  # another list, another output name
medstt sentences generate --limit 50            # only the first 50 words
```

The command uses `llm.base_url` and `llm.model` from [`medstt.yaml`](00-init.md); configure them for your LLM server before running it. The `sentences` section controls the output name (`output`, `sentences` by default), context, length, retry count, and `per_word`. Command-line options such as `--base-url`, `--model`, and `--per-word N` override the settings when needed. Use `--limit N` for the first N input rows and `--concurrency` for parallel requests.

`sentences.per_word` is the number of sentences requested **for each word**. Its default is `1`: a list of 100 words requests 100 sentences. With `per_word: 3`, the command makes three separate requests per word and aims to write 300 sentences. The code does not check whether the generated sentences differ. If you increase `per_word` and rerun the command, existing IDs are skipped and only the additional sentences are generated; decreasing it does not remove existing output.

Each sentence has a stable `id` built from the word's `source`, `code`, and `word` columns, for example `cim10-A00.0-vibrio`; the extra sentences of a word when `per_word` is above 1 are `cim10-A00.0-vibrio-2`, `-3`, and so on. A list without `source` and `code` columns (a `.txt` file) uses the row number instead (`000000`, `000000-2`). The ID therefore survives a rebuild of the word list: a word that is still there keeps its sentences, and sentences whose ID is no longer in the list are left in the file with a warning.

Each reply is cleaned and checked: the term must occur regardless of case and accents, and the sentence must have between `min_words` and `max_words` words. Generation is retried up to `max_attempts` times. Failed items are not written and will be tried again on the next run.

Output: `<workdir>/sentences/sentences.jsonl`, **not a `.txt` file**. Each JSONL line is a JSON object with `id`, `word`, `text` (the generated sentence), `llm`, and the other input columns. Existing records are skipped when the command is rerun. For example:

```json
{"id":"cim10-I30.9-péricardite","word":"péricardite","text":"Le patient présente une péricardite aiguë depuis ce matin.","llm":"<model-name>","source":"cim10","code":"I30.9","label":"Péricardite aiguë, sans précision", ...}
```

## Prompt

The system prompt asks for realistic French medical speech in `sentences.domain` (dictated reports, letters, handovers, exchanges between carers or with patients). The user prompt asks for one sentence that contains the word written exactly as given, capitals included, tells the LLM which label the word comes from and, when the word list has one, the concept definition, and asks it to use the word together with the neighbouring words of the label it belongs with (« acné varioliforme » rather than « varioliforme » alone) without copying a long label verbatim.

Each request is independent for the LLM, which therefore tends to write the same sentence shape every time (« Le patient présente une … confirmée par coproculture »). To spread the sentences, every one draws, deterministically from its ID, one entry of each list in the `sentences` section, and a target length between `min_words` and `max_words`:

```yaml
sentences:
  scenes: [compte rendu d'hospitalisation dicté, courrier au médecin traitant, transmission infirmière, ...]
  forms: [affirmation d'un diagnostic, hypothèse à confirmer, diagnostic écarté ou absent, ...]
  subjects: [le patient, la patiente, l'enfant, je, le soignant qui parle, ...]
```

The draw is stable across reruns, and the extra sentences of a word (`per_word` above 1) get different draws. Edit the lists to fit the specialty; an empty list removes that constraint. For a row of `words build`, the user prompt reads, for example:

```text
Rédige une phrase contenant le mot « varioliforme » écrit exactement ainsi, majuscules comprises.
Ce mot vient du libellé « Acné varioliforme ».
Emploie « varioliforme » avec les mots du libellé avec lesquels il forme une expression habituelle (le nom complet d'une maladie, d'un signe ou d'un acte), sans recopier le libellé entier s'il n'est pas naturel à l'oral.
Situation : courrier au médecin traitant. La phrase exprime : diagnostic écarté ou absent. Sujet de la phrase : la patiente.
La phrase doit compter environ 14 mots (entre 8 et 20).
```

With a bare word list (no `label` or `definition` column) the context line is simply omitted. Only the word itself is checked in the reply, so the LLM stays free to shorten or rephrase the label.

Use `--prompt-file` to supply a custom user prompt. It can reference `{word}`, `{label}` (the word itself when the list has no label), `{definition}` (empty when missing), `{context}` (the label and definition lines above, or empty), `{scene}`, `{form}`, `{subject}`, `{target_words}`, `{constraints}` (the three draws as one sentence), `{domain}`, `{min_words}`, `{max_words}`, and any column from the input file; blank lines left by empty placeholders are dropped. You can also write sentences manually: a `.txt` file with one sentence per line, or a CSV/JSONL file with a `text` column (changeable with `--text-column` during synthesis).

See [Slurm instructions](slurm.md#sentences). Next: [synthesize audio](03-tts.md).
