"""Unit tests for `SeroRefRecord`, `SeroRefRecords`, `LociCacheBundle`, and multi-antigen parsing."""

from pathlib import Path

from seroverse.core.collections import BatchedContainer, RaggedArrayContainer, ScalarView

from seroref.db import (
    LociCacheBundle,
    LociCacheEntry,
    SeroRefRecord,
    SeroRefRecords,
    extract_multi_antigen_tokens,
)
from seroref.loci.models import Loci, Locus


def test_seroref_scalar_and_container_hierarchy() -> None:
    assert ScalarView in SeroRefRecord.__mro__
    assert ScalarView in LociCacheEntry.__mro__
    assert BatchedContainer in SeroRefRecords.__mro__
    assert RaggedArrayContainer in LociCacheBundle.__mro__


def test_extract_multi_antigen_tokens() -> None:
    ecoli = extract_multi_antigen_tokens("Escherichia coli", "O1:K1(L1):H7")
    assert ecoli == [(b"O-antigen", b"O1"), (b"CPS", b"K1")]

    kleb = extract_multi_antigen_tokens("Klebsiella pneumoniae", "Capsular Serotype 3")
    assert kleb == [(b"CPS", b"K3")]

    paer = extract_multi_antigen_tokens("Pseudomonas aeruginosa", "Serotype 11")
    assert paer == [(b"O-antigen", b"O11")]


def test_seroref_records_roundtrip(tmp_path: Path) -> None:
    rec1 = SeroRefRecord(
        organism=b"Escherichia coli",
        antigen_type=b"O-antigen",
        serotype=b"O1",
        strain=b"NCTC 9001",
        strain_synonyms=(b"NCTC 9001", b"ATCC 11775"),
        assembly_id=b"GCA_900706755.1",
        locus=Locus(nodes=(), edges=()),
        ncbi_taxid=562,
        straininfo_id=1001,
        bacdive_id=2001,
        pmids=(15247885, 19000001),
    )
    rec2 = SeroRefRecord(
        organism=b"Escherichia coli",
        antigen_type=b"CPS",
        serotype=b"K1",
        strain=b"NCTC 9001",
        strain_synonyms=(b"NCTC 9001",),
        assembly_id=b"GCA_900706755.1",
        locus=Locus(nodes=(), edges=()),
        ncbi_taxid=562,
        pmids=(),
    )
    batch = SeroRefRecords.from_list([rec1, rec2])
    assert len(batch) == 2
    assert batch[0].serotype == b"O1"
    assert batch[0].strain_synonyms == (b"NCTC 9001", b"ATCC 11775")
    assert batch[0].pmids == (15247885, 19000001)
    assert len(batch.where_antigen_type(b"CPS")) == 1

    out_path = tmp_path / "seroref.npz"
    batch.save(out_path)
    loaded = SeroRefRecords.load(out_path)
    assert len(loaded) == 2
    assert loaded[0].strain_synonyms == (b"NCTC 9001", b"ATCC 11775")
    assert loaded[1].serotype == b"K1"


def test_loci_cache_bundle_roundtrip(tmp_path: Path) -> None:
    bundle = LociCacheBundle.from_dicts(
        {b"GCA_900706755.1": (b"GCA_900706755.1", Loci.empty())},
        {(b"Escherichia coli", b"NCTC 9001"): b"GCA_900706755.1"},
        {15247885: b"10.1016/j.carres.2004.05.001"},
    )
    assert len(bundle) == 1
    assert bundle[0].cache_key == b"GCA_900706755.1"
    p = tmp_path / "loci_cache.npz"
    bundle.save(p)
    loaded = LociCacheBundle.load(p)
    assert len(loaded) == 1
    assert loaded.strain_to_assembly[(b"Escherichia coli", b"NCTC 9001")] == b"GCA_900706755.1"
    assert loaded.pmid_to_doi[15247885] == b"10.1016/j.carres.2004.05.001"
