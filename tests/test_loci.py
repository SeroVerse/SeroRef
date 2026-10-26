"""Tests for `seroref.loci` archetype ontology, `LocusPredictor`, `Loci`, and `LocusMatcher`."""

from pathlib import Path

import numpy as np
from seroverse.core.collections import RaggedArrayContainer, ScalarView
from seroverse.core.genome import GenomeAssembly

from seroref.db.records import SeroRefRecords, loci_from_npz_dict, loci_to_npz_dict
from seroref.loci.archetypes import ArchetypeCategory, ArchetypeSpec, ArchetypeSpecs, LocusArchetype
from seroref.loci.matcher import LocusMatcher, Match, Matches
from seroref.loci.models import Loci, Locus
from seroref.loci.predictor import LocusPredictor


def test_archetype_ontology_hierarchy() -> None:
    """`ArchetypeCategory`, `LocusArchetype`, `ArchetypeSpec`, and `ArchetypeSpecs` must follow the 3-tier design."""
    assert ScalarView in ArchetypeSpec.__mro__
    assert RaggedArrayContainer in ArchetypeSpecs.__mro__

    assert ArchetypeCategory.resolve("Capsule") == ArchetypeCategory.CAPSULE
    assert ArchetypeCategory.resolve(b"O-Antigen") == ArchetypeCategory.O_ANTIGEN
    assert ArchetypeCategory.CAPSULE.label == b"Capsule"

    specs = LocusArchetype.specs()
    assert isinstance(specs, ArchetypeSpecs)
    assert len(specs) == len(LocusArchetype) == 11
    assert isinstance(specs[0], ArchetypeSpec)
    assert specs[0].name == b"CPS_WZY"
    assert specs[0].category == ArchetypeCategory.CAPSULE
    assert len(ArchetypeSpecs.concat([specs[:3], specs[3:6]])) == 6
    assert len(ArchetypeSpecs.empty()) == 0

    assert LocusArchetype.resolve("LPS_WZY") == LocusArchetype.LPS_WZY
    assert LocusArchetype.resolve(b"CPS_ABC") == LocusArchetype.CPS_ABC


def test_locus_predictor_and_serialization(
    sgh10_assembly: GenomeAssembly,
    tmp_path: Path,
) -> None:
    """`LocusPredictor` must extract antigen loci from *K. pneumoniae* SGH10 and round-trip via `.npz`."""
    predictor = LocusPredictor()
    loci = predictor.predict(sgh10_assembly)

    assert isinstance(loci, Loci)
    assert len(loci) >= 2
    assert isinstance(loci[0], Locus)

    categories = {loc.archetype.category for loc in loci}
    assert ArchetypeCategory.CAPSULE in categories or ArchetypeCategory.O_ANTIGEN in categories

    for loc in loci:
        assert len(loc.nodes) >= 1
        assert loc.start < loc.end

    out_path = tmp_path / "sgh10_loci.npz"
    np.savez_compressed(out_path, **loci_to_npz_dict(loci))
    with np.load(out_path, allow_pickle=False) as data:
        loaded = loci_from_npz_dict(data)
    assert len(loaded) == len(loci)
    assert loaded[0].archetype == loci[0].archetype
    assert loaded[0].archetype.category == loci[0].archetype.category


def test_locus_matcher_against_reference_db(sgh10_assembly: GenomeAssembly) -> None:
    """`LocusMatcher` must match predicted loci against `SeroRefRecords`."""
    predictor = LocusPredictor()
    loci = predictor.predict(sgh10_assembly)
    db = SeroRefRecords.load()
    kleb_db = db.where_species("Klebsiella pneumoniae")
    matcher = LocusMatcher(kleb_db.loci)
    matches = matcher.match(loci, top_k=3)

    assert isinstance(matches, Matches)
    assert len(matches) > 0
    assert isinstance(matches[0], Match)
    assert 0.0 <= matches[0].jaccard <= 1.0
