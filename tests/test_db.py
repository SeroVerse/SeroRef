"""Unit tests for `seroref.db` species ontology, reference strains, and `SeroRefRecords`."""

from pathlib import Path

from seroverse.core.collections import BatchedContainer, RaggedArrayContainer, ScalarView

from seroref.db.models import ReferenceStrain, SeroRefRecord
from seroref.db.records import ReferenceStrains, SeroRefRecords
from seroref.db.species import BacterialSpecies, GramStain, SpeciesSpec, SpeciesSpecs
from seroref.loci.archetypes import ArchetypeCategory, LocusArchetype
from seroref.loci.models import Locus


def test_species_ontology_hierarchy() -> None:
    """`SpeciesSpec` and `SpeciesSpecs` must follow the 3-tier `ScalarView` / `RaggedArrayContainer` architecture."""
    assert ScalarView in SpeciesSpec.__mro__
    assert RaggedArrayContainer in SpeciesSpecs.__mro__
    assert ScalarView in ReferenceStrain.__mro__
    assert RaggedArrayContainer in ReferenceStrains.__mro__
    assert ScalarView in SeroRefRecord.__mro__
    assert BatchedContainer in SeroRefRecords.__mro__

    assert GramStain.resolve("negative") == GramStain.NEGATIVE
    assert GramStain.resolve(b"positive") == GramStain.POSITIVE
    assert GramStain.NEGATIVE.label == b"Negative"

    specs = BacterialSpecies.specs()
    assert isinstance(specs, SpeciesSpecs)
    assert len(specs) == len(BacterialSpecies)
    assert isinstance(specs[0], SpeciesSpec)
    assert specs[0].name == b"Acinetobacter baumannii"
    assert specs[0].scientific_name == b"Acinetobacter baumannii"
    assert len(SpeciesSpecs.concat([specs[:4], specs[4:8]])) == 8
    assert len(SpeciesSpecs.empty()) == 0

    assert BacterialSpecies.resolve("Klebsiella") == BacterialSpecies.KLEBSIELLA_PNEUMONIAE
    assert BacterialSpecies.resolve(b"Escherichia coli") == BacterialSpecies.ESCHERICHIA_COLI
    assert BacterialSpecies.resolve("Salmonella enterica") == BacterialSpecies.SALMONELLA_ENTERICA


def test_species_serotype_parsing_and_archetypes() -> None:
    """`BacterialSpecies.parse_serotype` must extract antigen categories and normalized serotype tokens."""
    ecoli = BacterialSpecies.ESCHERICHIA_COLI
    assert ecoli.parse_serotype("O1:K1(L1):H7") == (
        (ArchetypeCategory.O_ANTIGEN, b"O1"),
        (ArchetypeCategory.CAPSULE, b"K1"),
    )

    kleb = BacterialSpecies.KLEBSIELLA_PNEUMONIAE
    assert kleb.parse_serotype("Capsular Serotype 3") == ((ArchetypeCategory.CAPSULE, b"K3"),)
    assert kleb.spec.default_category == ArchetypeCategory.CAPSULE

    paer = BacterialSpecies.PSEUDOMONAS_AERUGINOSA
    assert paer.parse_serotype("Serotype 11") == ((ArchetypeCategory.O_ANTIGEN, b"O11"),)


def test_reference_strains_and_seroref_records_roundtrip(tmp_path: Path) -> None:
    """`ReferenceStrains` and `SeroRefRecords` must round-trip cleanly through compressed `.npz` archives."""
    rec1 = SeroRefRecord.from_fields(
        b"Escherichia coli",
        serotype=b"O1",
        strain=b"NCTC 9001",
        strain_synonyms=(b"NCTC 9001", b"ATCC 11775"),
        assembly_id=b"GCA_900706755.1",
        locus=Locus.from_genes([b"wzx", b"wzy"], archetype=LocusArchetype.LPS_WZY, name=b"O1"),
        ncbi_taxid=562,
        straininfo_id=1001,
        bacdive_id=2001,
        pmids=(15247885, 19000001),
    )
    rec2 = SeroRefRecord.from_fields(
        b"Escherichia coli",
        serotype=b"K1",
        strain=b"NCTC 9001",
        strain_synonyms=(b"NCTC 9001",),
        assembly_id=b"GCA_900706755.1",
        locus=Locus.from_genes([b"kpsM", b"kpsT"], archetype=LocusArchetype.CPS_ABC, name=b"K1"),
        ncbi_taxid=562,
        pmids=(),
    )
    batch = SeroRefRecords.from_list([rec1, rec2])
    assert len(batch) == 2
    assert len(batch.strains) == 1
    assert batch[0].serotype == b"O1"
    assert batch[0].strain_synonyms == (b"NCTC 9001", b"ATCC 11775")
    assert batch[0].pmids == (15247885, 19000001)
    assert len(batch.where_antigen_type(b"CPS")) == 1

    out_path = tmp_path / "seroref_test.npz"
    batch.save(out_path)
    loaded = SeroRefRecords.load(out_path)
    assert len(loaded) == 2
    assert loaded[0].strain_synonyms == (b"NCTC 9001", b"ATCC 11775")
    assert loaded[1].serotype == b"K1"

    db = SeroRefRecords.load()
    assert len(db) > 100
    assert len(db.where_species(BacterialSpecies.STREPTOCOCCUS_PNEUMONIAE)) > 50
    assert len(db.where_antigen_type(b"CPS")) > 50
