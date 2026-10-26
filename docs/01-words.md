# 01 — Medical terms

**TODO: automatic extraction from a nomenclature is not implemented.** The `medstt words from-nomenclature` command exists but exits with an error. Selection and normalization rules still need to be defined in [TODO.md](../TODO.md).

For now, place a manually prepared list under `<workdir>/words/`:

```text
péricardite
échocardiographie
tachycardie sinusale
```

A `.txt` file contains one non-empty term per line. CSV, TSV, JSONL, JSON (a list of objects), and Parquet are also accepted, with a `word` column by default. Use `--column` in the next step to select a different column. Additional columns such as `code`, `category`, or `definition` are carried into the sentence and audio metadata and can be used in a custom prompt.

Next: [generate sentences](02-sentences.md).
