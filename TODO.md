# TODO

The design decisions and the planned phases are in [docs/roadmap.md](docs/roadmap.md). Open points:

## Words

- [ ] Calibrate `words.max_zipf` on the `--stats` output and on a sample of `words/*.jsonl` (ordinary words slipping through vs. medical words dropped).
- [ ] Decide which source-specific fields (`extra`) are worth passing to the sentence prompt.
- [ ] Add the CHU de Montpellier vocabulary (directory export + manual CSV); decide how acronyms and proper names are spelled out for the TTS.
- [ ] Consider further sources once the licence is checked: French MeSH (Loterre), BDPM drug names.

## Evaluation

`medstt evaluate` is a stub (`src/medstt/eval/evaluate.py`); see the roadmap for the planned protocol (held-out words, FLEURS fr, term recall, comparison with the base model).

- [ ] Verify on the cluster that training (BOS + prompt) and generation (bare prompt) agree on `decoder_start_token_id` before relying on evaluation numbers.
