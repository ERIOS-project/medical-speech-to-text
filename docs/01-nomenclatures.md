# 01a — Import nomenclatures

The vocabulary to learn comes from medical nomenclatures. Each one is declared in `<workdir>/medstt.yaml`, downloaded from its source, and rewritten in one common JSONL format so that the next step ([select words](01-words.md)) does not depend on the source.

## Configure

`medstt init` writes this section. Each entry needs a `format` and a `url`; `file` (a glob) is only required when the archive contains several data files.

```yaml
nomenclatures:
  cim10:
    format: cim10-claml
    url: https://www.atih.sante.fr/sites/default/files/content/documents/cim10fr2024syst_claml_20231215.zip
  orphanet:
    format: orphanet-xml
    url: https://www.orphacode.org/data/packs/Orphanet_Nomenclature_Pack_FR.zip
    file: ORPHAnomenclature_fr_*.xml
```

| Format | Source | What is kept |
| --- | --- | --- |
| `cim10-claml` | [CIM-10-FR à usage PMSI](https://www.atih.sante.fr/cim-10-fr-2024-usage-pmsi) (ATIH, ClaML XML in a zip). See the ATIH terms of use. | The 10 040 leaf diagnostic categories: preferred label, inclusion terms as synonyms, definition when present, chapter and block. Chapters, blocks, and categories with subcategories are skipped. Cross-reference codes nested in labels are removed. |
| `orphanet-xml` | [Orphanet nomenclature pack, French](https://www.orphacode.org/pack-nomenclature/) (zip with several XML files, CC-BY 4.0). | All 11 645 clinical entities: name, synonyms, definition (available for about 60%), disorder type, classification level. |

To add a source, add an importer under `src/medstt/nomenclatures/importers/`, register its format in `IMPORTERS`, and declare it in the YAML.

## Import

```bash
medstt nomenclature import            # every configured nomenclature
medstt nomenclature import orphanet   # one of them
medstt nomenclature import --force    # download and normalize again
```

The command needs network access. Only the data file of each download is kept, as `<workdir>/nomenclature/raw/<name>.xml` (the archive and its other files are discarded), and reused on later runs; the normalized file is `<workdir>/nomenclature/normalized/<name>.jsonl`. Prepare the workspace on a connected machine before copying it to an offline cluster.

```text
<workdir>/nomenclature/
  raw/cim10.xml        the ClaML file from the ATIH zip
  raw/orphanet.xml     ORPHAnomenclature_fr_<year>.xml from the Orphanet pack
  normalized/cim10.jsonl
  normalized/orphanet.jsonl
```

## Normalized format

One JSON object per concept:

```json
{"code": "A00.9", "source": "cim10", "label": "Choléra, sans précision", "synonyms": [],
 "definition": null, "category": "A", "parent": "A00",
 "extra": {"chapter": "A", "block": "A00-A09"}}
```

| Field | Meaning |
| --- | --- |
| `code` | Identifier in the source (CIM-10 code, ORPHAcode) |
| `source` | Nomenclature name from the YAML |
| `label` | Preferred label |
| `synonyms` | Other labels, deduplicated |
| `definition` | Definition text, or `null` |
| `category` | Chapter letter (CIM-10) or disorder type (Orphanet) |
| `parent` | Parent code when the source has a hierarchy, else `null` |
| `extra` | Source-specific fields (block, classification level, status, expert link) |

Next: [select the rare words](01-words.md).
