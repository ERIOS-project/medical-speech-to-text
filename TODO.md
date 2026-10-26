# TODO

## Phase 0 — words from a nomenclature

`medstt words from-nomenclature` is a stub (`src/medstt/words/nomenclature.py`).
To define before implementing:

- [ ] Source nomenclatures and formats (e.g. CCAM, CIM-10, SNOMED CT, LOINC, drug lists).
- [ ] Selection rules: which entries become words (preferred labels, synonyms, abbreviations).
- [ ] Granularity: single terms vs. multi-word expressions; splitting of long labels.
- [ ] Filtering: drop terms already well recognized by the baseline model (measure the
      baseline term recall first), deduplicate, normalize case and accents.
- [ ] Pronunciation issues: acronyms (spelled vs. read), numbers, units, dosages.
- [ ] Columns kept for the prompt (code, category, definition) to get realistic contexts.

## Evaluation

`medstt evaluate` is a stub (`src/medstt/eval/evaluate.py`). To define before implementing:

- [ ] Test sets: synthetic held-out split (by word, to measure unseen terms) and
      public general French speech to detect forgetting.
- [ ] Metrics: WER/CER, text normalization rules, medical term recall / term error rate.
- [ ] Baseline: the pretrained model before fine-tuning, on the same test sets.
- [ ] Report format (per dataset, per term) and Slurm script.
