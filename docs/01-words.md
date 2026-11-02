# 01b — Select medical words

The speech model must learn the words it does not know. Nomenclature labels mix those words with ordinary French (« maladie », « aiguë », « sans précision »), so this step keeps, for each concept of the [normalized nomenclatures](01-nomenclatures.md), only the words that are **rare in general French**. Each kept word carries its concept (code, label, definition) so that the [sentence step](02-sentences.md) can give the LLM a realistic context.

## Configure

```yaml
words:
  sources: [cim10, orphanet]   # normalized nomenclatures to read (default: all configured)
  output: medical              # writes words/medical.jsonl
  max_zipf: 3.0                # a word is rare when its French Zipf frequency is at most this
  min_length: 4                # minimum letters per word
```

Frequencies come from the [`wordfreq`](https://pypi.org/project/wordfreq/) lexicon, bundled with the package and usable offline. Its Zipf scale is the base-10 logarithm of a word's occurrences per billion words of French: about 7 for « de », 4 for « maladie », 3 for « arthrite », 2 for « choléra », and 0 for a word absent from the lexicon such as « varioliforme ».

## Build

```bash
medstt words build                    # sources and output from the YAML
medstt words build cim10 -o cim10     # one source, another output name
medstt words build --limit 500 --stats
```

The rules, applied to the label and then to each synonym of every concept:

1. Split the text into words. Elided articles are dropped (« l'index » → « index ») and hyphenated chains are split (« multiple-macrocéphalie-dysmorphie » gives three words), because nomenclatures use hyphens to join whole phrases.
2. Drop words shorter than `min_length`, numbers, Roman numerals, and common function words.
3. Keep the words whose Zipf frequency is at most `max_zipf`.
4. Keep each word once (compared without case or accents): the first concept that mentions it, in the order of `sources`, becomes its context. A concept without any rare word is skipped.

`--stats` prints, per source, the concepts read, those without a rare word, and the words kept; a histogram of candidate words per Zipf band; and the 30 words kept and 30 words dropped closest to the threshold. Use it to tune `max_zipf`: lower it when ordinary words slip through, raise it when medical words are dropped. With the default settings, CIM-10 and Orphanet give about 16 000 words.

## Output

`<workdir>/words/<output>.jsonl`, one JSON object per word:

```json
{"word": "varioliforme", "zipf": 0.0, "source": "cim10", "code": "L70.2",
 "label": "Acné varioliforme", "definition": null, "category": "L", "synonyms": [],
 "extra": {"chapter": "L", "block": "L60-L75"}}
```

When the word comes from a synonym, `label` is that synonym and `extra.label` holds the concept's main label. The file can be read directly to check what was kept, and edited or filtered by hand before the next step.

A manually prepared list still works: a `.txt` file with one word per line, or CSV/TSV/JSONL/JSON/Parquet with a `word` column (`--column` selects another one in the next step).

Next: [generate sentences](02-sentences.md).
