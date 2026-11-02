# Roadmap and design decisions

This page records the decisions taken for MedSTT and the phases that remain. It is the reference when a step is reopened; the step guides describe what is implemented.

## Goal and approach

Adapt Cohere Transcribe to the vocabulary of medical nomenclatures (CIM-10, Orphanet) and of the CHU de Montpellier, without losing accuracy on general French speech. No real medical audio is available, so the training data is synthetic: rare words → LLM sentences → TTS audio → full fine-tuning.

| Risk | Mitigation |
| --- | --- |
| The TTS mispronounces a rare word and the ASR learns a wrong sound/word association | Round-trip filter with the base model (phase 6) and targeted listening |
| The ASR learns the TTS voices rather than the words | Pool of voices (phase 4) and on-the-fly audio augmentation (phase 8) |
| The ASR forgets general speech | Real French speech mixed into training and a general test set (phases 7 and 9) |
| Metrics measure memorization, not generalization | Word-level train/held-out split and standalone evaluation (phases 3 and 9) |
| Silent losses between steps | A check at each step: direct reading of the JSONL files, `words build --stats`, sentence and audio checks, evaluation report |

## Decisions

- **Nomenclatures**: CIM-10 FR (ATIH, ClaML) and Orphanet (nomenclature pack FR) now, declared in `medstt.yaml` with a format and a URL and always downloaded from the source. Later: the CHU de Montpellier directory export (`Struct_Med` XML: 86 services and 198 units with clean accents under `Dep`/`Eme`, 28 pôles, about 850 physicians with a Dr/Pr title under `User/@Title`, hospital sites through `RoomNumber` prefixes such as ADV, LAP, SEL, GDC, COL, BEL; uppercase without accents elsewhere) plus a manual CSV for spoken acronyms, places, neighbouring institutions, and pronunciations. French MeSH (Loterre) and BDPM drug names are candidates once their licences are checked. SNOMED CT (ANS licence, RF2, partial translation) and HPO are not used.
- **Words**: the unit is (concept, rare word). A word is rare when its `wordfreq` Zipf frequency in French is at most `words.max_zipf`. Each word is kept once with the first concept that mentions it (code, label, definition). Hyphenated chains are split. No `term` or `spoken` column: the sentence prompt asks the LLM to use the word together with the neighbouring words of the label it belongs with (« acné varioliforme »). CHU terms will bypass the rarity filter.
- **Split** (phase 3, `words split`): by word, stratified by source, 10% held out. Two synthetic evaluation sets: *seen words, new sentences* (an extra sentence for 10% of the trained words, with another voice draw) and *unseen words*. The final model is retrained on 100% of the words once the settings are fixed.
- **Sentences**: the prompt receives `word`, `label`, and `definition`, and asks for the word in its natural expression. A warning is raised when an existing ID carries another word than the current list.
- **TTS**: a pool of 6 voices (3 male, 3 female) with different ages, regional accents, speeds, and tones, drawn deterministically per sentence; two recordings per sentence. Synthesis stays sequential within a shard; parallelism comes from the Slurm job array.
- **Pronunciation filter** (train environment, after synthesis): transcribe each synthetic WAV with the base model; `cer_context` (CER on the sentence without the word, threshold 0.3) and `term_distance` (edit distance on the word, threshold 0.6, optional); rejections recorded in `_parts/rejected-asr.jsonl`, audio kept, `--dry-run` for calibration. Why the TTS pronounces better than the ASR recognizes: French spelling is regular from letters to sounds, so a TTS generalizes by rule, while an ASR recalls by memory with a prior towards known words. TTS failures are predictable (acronyms, proper names, numbers): listen to those first.
- **General speech**: MLS French mixed **during** training (about 25–30% at first), tuned by measuring the WER on FLEURS fr (more real speech or a lower learning rate when it degrades by more than 10% relative).
- **Training**: short runs (smoke 10 min, calibration 1 h on about 2 000 words, full 2–3 h) with `max_steps` and frequent checkpoints; `eval_split` grouped by word then by source sentence; on-the-fly augmentation in the collator (noise, babble, reverberation, speed, telephone band, gain), training only, off by default.
- **Evaluation**: seen / unseen / FLEURS fr (and Common Voice fr); no SIMSAMU or PxSLU; base model vs. run vs. Whisper large-v3; WER/CER plus exact and approximate term recall (edit distance ≤ 0.2), per source; outputs under `evals/<name>/` (predictions, `metrics.json`, `per_term.csv`, `report.md`).
- **Before the hackathon**: prepare the workspace (import, words, split, sentences, TTS if a GPU is available) and copy it to the cluster. First checks on the cluster: `decoder_start_token_id` (training prepends BOS to the prompt while generation starts from the bare prompt), prompt absent from `generate` outputs, base WER on FLEURS, listening, rejection rate, smoke run.

## Phases

| Phase | Content | Status |
| --- | --- | --- |
| 0–2 | Workspace layout, `nomenclature import`, `words build` | Done |
| 3 | `words split`, richer sentence prompt (`label`, `definition`, natural expression), ID warning, `sentences check`, *seen words, new sentences* set | Next |
| 4 | `tts.voices` (6 descriptions), deterministic draw, `dataset check`, `dataset sample` | |
| 5 | Shared inference (`CohereTranscriber`, batching, chunk joining for audio over 30 s), `medstt transcribe`, `inference:` section | |
| 6 | `medstt dataset filter` (round-trip ASR), rejections excluded by `finalize_dataset`, `slurm/filter_dataset.sbatch` | |
| 7 | `source_id` kept when loading datasets, grouped `eval_split`, MLS mixing recipe and run profiles | |
| 8 | `train/augment.py`, `CohereCollator(augmenter)`, evaluation collator without augmentation | |
| 9 | `evaluate:` section, model specs, Whisper transcriber, reports, `slurm/evaluate.sbatch` | |
| CHU | Directory importer and manual CSV, case/accent restoration, spoken forms, « Pr NOM Prénom » / « Dr Prénom Nom » variants | |

## Open points

Source fields worth passing to the prompt; content of the CHU CSV; thresholds (Zipf, CER, edit distance); external comparison model; the BOS mismatch if confirmed on the cluster.
