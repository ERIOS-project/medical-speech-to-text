"""Nomenclature download, parsing, and normalization without network access."""

import io
import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from medstt.cli import app
from medstt.nomenclatures import fetch
from medstt.nomenclatures.config import NomenclatureSpec, parse_nomenclatures
from medstt.nomenclatures.importers.cim10_claml import iter_cim10_concepts
from medstt.nomenclatures.importers.orphanet_xml import iter_orphanet_concepts
from medstt.nomenclatures.normalize import import_nomenclature
from medstt.workspace import Workspace

CLAML = """<?xml version="1.0" encoding="UTF-8"?>
<ClaML version="2.0.0">
  <Class code="I" kind="chapter"><SubClass code="A00-A09"/>
    <Rubric kind="preferred"><Label>Certaines maladies infectieuses</Label></Rubric></Class>
  <Class code="A00-A09" kind="block"><SuperClass code="I"/><SubClass code="A00"/>
    <Rubric kind="preferred"><Label>Maladies intestinales infectieuses</Label></Rubric></Class>
  <Class code="A00" kind="category"><SuperClass code="A00-A09"/>
    <SubClass code="A00.0"/><SubClass code="A00.9"/>
    <Rubric kind="preferred"><Label>Choléra</Label></Rubric></Class>
  <Class code="A00.0" kind="category"><SuperClass code="A00"/>
    <Rubric kind="preferred"><Label xml:space="default">À Vibrio   cholerae 01</Label></Rubric>
    <Rubric kind="inclusion"><Label>Choléra classique</Label></Rubric>
    <Rubric kind="inclusion"><Label>choléra CLASSIQUE</Label></Rubric>
    <Rubric kind="inclusion"><Label>
      <Fragment type="list">Hépatite (à) :</Fragment>
      <Fragment type="list">cytomégalovirus<Reference>B25.1</Reference></Fragment>
    </Label></Rubric>
    <Rubric kind="definition"><Label>Une définition.</Label></Rubric>
  </Class>
  <Class code="A00.9" kind="category"><SuperClass code="A00"/>
    <Rubric kind="preferred">
      <Label>Abcès amibien du foie<Reference>K77.0</Reference> </Label></Rubric>
    <Rubric kind="exclusion"><Label>Ignoré</Label></Rubric>
  </Class>
</ClaML>
"""

ORPHANET = """<?xml version="1.0" encoding="UTF-8"?>
<JDBOR>
  <DisorderList count="2">
    <Disorder id="1">
      <OrphaCode>166024</OrphaCode>
      <ExpertLink lang="fr">http://example.org/166024</ExpertLink>
      <Name lang="fr">Syndrome de dysplasie épiphysaire multiple</Name>
      <Totalstatus lang="fr">Active</Totalstatus>
      <SynonymList count="2">
        <Synonym lang="fr">Dysplasie épiphysaire type Al-Gazali</Synonym>
        <Synonym lang="fr">Syndrome de dysplasie épiphysaire multiple</Synonym>
      </SynonymList>
      <DisorderType><Name lang="fr">Maladie</Name></DisorderType>
      <ClassificationLevel><Name lang="fr">Pathologie</Name></ClassificationLevel>
      <SummaryInformationList><SummaryInformation><TextSectionList>
        <TextSection><TextSectionType><Name lang="fr">Définition</Name></TextSectionType>
          <Contents>Dysplasie osseuse  rare.</Contents></TextSection>
      </TextSectionList></SummaryInformation></SummaryInformationList>
    </Disorder>
    <Disorder id="2">
      <OrphaCode>166032</OrphaCode>
      <Name lang="fr">Fibromatose hyaline</Name>
      <SynonymList count="0"></SynonymList>
      <DisorderType><Name lang="fr">Maladie</Name></DisorderType>
    </Disorder>
  </DisorderList>
</JDBOR>
"""


def test_cim10_keeps_leaf_categories_without_reference_codes(tmp_path):
    path = tmp_path / "cim10.xml"
    path.write_text(CLAML, encoding="utf-8")

    concepts = {c.code: c for c in iter_cim10_concepts(path)}

    assert set(concepts) == {"A00.0", "A00.9"}
    first = concepts["A00.0"]
    assert first.label == "À Vibrio cholerae 01"
    assert first.synonyms == ["Choléra classique", "Hépatite (à) : cytomégalovirus"]
    assert first.definition == "Une définition."
    assert first.parent == "A00"
    assert first.category == "A"
    assert first.extra == {"chapter": "A", "block": "A00-A09"}
    assert concepts["A00.9"].label == "Abcès amibien du foie"
    assert concepts["A00.9"].definition is None


def test_orphanet_reads_name_synonyms_and_definition(tmp_path):
    path = tmp_path / "orpha.xml"
    path.write_text(ORPHANET, encoding="utf-8")

    concepts = list(iter_orphanet_concepts(path))

    assert [c.code for c in concepts] == ["166024", "166032"]
    first = concepts[0]
    assert first.synonyms == ["Dysplasie épiphysaire type Al-Gazali"]
    assert first.definition == "Dysplasie osseuse rare."
    assert first.category == "Maladie"
    assert first.extra["classification_level"] == "Pathologie"
    assert concepts[1].definition is None and concepts[1].synonyms == []


def test_spec_parse_rejects_bad_settings():
    with pytest.raises(ValueError, match="unknown keys"):
        NomenclatureSpec.parse("x", {"format": "cim10-claml", "url": "u", "path": "p"})
    with pytest.raises(ValueError, match="requires"):
        NomenclatureSpec.parse("x", {"format": "cim10-claml"})
    with pytest.raises(ValueError, match="unknown format"):
        NomenclatureSpec.parse("x", {"format": "snomed", "url": "u"})
    specs = parse_nomenclatures({"a": {"format": "orphanet-xml", "url": "u", "file": "f.xml"}})
    assert specs[0].file == "f.xml"


def _fake_urlopen(payload: bytes):
    def urlopen(url):
        return io.BytesIO(payload)

    return urlopen


def _zip_bytes(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buffer.getvalue()


def test_fetch_downloads_once_and_extracts_the_single_xml(tmp_path, monkeypatch):
    calls = []
    payload = _zip_bytes({"pack/readme.txt": "x", "pack/data.xml": ORPHANET})

    def urlopen(url):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(fetch, "urlopen", urlopen)
    spec = NomenclatureSpec("orphanet", "orphanet-xml", "https://example.org/pack.zip")

    data_file = fetch.fetch_source(tmp_path, spec)
    again = fetch.fetch_source(tmp_path, spec)

    assert data_file == tmp_path / "orphanet.xml" == again
    assert data_file.read_text(encoding="utf-8") == ORPHANET
    assert calls == ["https://example.org/pack.zip"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["orphanet.xml"]


def test_fetch_needs_file_when_several_xml(tmp_path, monkeypatch):
    payload = _zip_bytes({"a.xml": ORPHANET, "b.xml": ORPHANET})
    monkeypatch.setattr(fetch, "urlopen", _fake_urlopen(payload))
    spec = NomenclatureSpec("o", "orphanet-xml", "https://example.org/p.zip")

    with pytest.raises(ValueError, match="set `file:`"):
        fetch.fetch_source(tmp_path, spec)

    chosen = fetch.fetch_source(tmp_path, NomenclatureSpec("o", "orphanet-xml", spec.url, "b*.xml"))
    assert chosen == tmp_path / "o.xml" and chosen.read_text(encoding="utf-8") == ORPHANET
    assert sorted(path.name for path in tmp_path.iterdir()) == ["o.xml"]
    with pytest.raises(FileNotFoundError, match="exactly one"):
        fetch.fetch_source(tmp_path, NomenclatureSpec("o", "orphanet-xml", spec.url, "*.xml"), True)
    assert chosen.read_text(encoding="utf-8") == ORPHANET  # the kept file survives a failed --force


def test_import_writes_normalized_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "urlopen", _fake_urlopen(CLAML.encode("utf-8")))
    spec = NomenclatureSpec("cim10", "cim10-claml", "https://example.org/cim10.xml")

    output, count = import_nomenclature(tmp_path / "raw", tmp_path / "norm", spec)

    assert (tmp_path / "raw" / "cim10.xml").is_file()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert count == 2 and len(rows) == 2
    assert rows[0] == {
        "code": "A00.0",
        "source": "cim10",
        "label": "À Vibrio cholerae 01",
        "synonyms": ["Choléra classique", "Hépatite (à) : cytomégalovirus"],
        "definition": "Une définition.",
        "category": "A",
        "parent": "A00",
        "extra": {"chapter": "A", "block": "A00-A09"},
    }


def test_cli_imports_every_configured_nomenclature(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "medstt.yaml").write_text(
        "nomenclatures:\n"
        "  cim10: {format: cim10-claml, url: https://example.org/c.xml}\n"
        "  orphanet: {format: orphanet-xml, url: https://example.org/o.xml}\n",
        encoding="utf-8",
    )
    payloads = {"https://example.org/c.xml": CLAML, "https://example.org/o.xml": ORPHANET}
    monkeypatch.setattr(fetch, "urlopen", lambda url: io.BytesIO(payloads[url].encode("utf-8")))

    result = CliRunner().invoke(app, ["nomenclature", "import", "--workdir", str(ws.root)])

    assert result.exit_code == 0, result.output
    assert "cim10: 2 concepts" in result.output and "orphanet: 2 concepts" in result.output
    assert (ws.root / "nomenclature" / "normalized" / "orphanet.jsonl").is_file()

    unknown = CliRunner().invoke(app, ["nomenclature", "import", "hpo", "--workdir", str(ws.root)])
    assert unknown.exit_code != 0 and "Unknown nomenclature" in unknown.output


def test_cli_reports_download_errors(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")

    def failing(url):
        raise OSError("no network")

    monkeypatch.setattr(fetch, "urlopen", failing)
    result = CliRunner().invoke(app, ["nomenclature", "import", "cim10", "--workdir", str(ws.root)])

    assert result.exit_code == 1
    assert "no network" in result.output
    assert not list(Path(ws.root / "nomenclature" / "normalized").glob("*.jsonl"))
