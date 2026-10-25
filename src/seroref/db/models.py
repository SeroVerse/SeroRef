"""Data models for curated reference strains, genome assemblies, and SeroRef antigen loci."""

from __future__ import annotations

from dataclasses import dataclass

from seroverse.core.collections import ScalarView

from seroref.loci.archetypes import LocusArchetype
from seroref.loci.models import Locus


@dataclass(frozen=True, slots=True)
class SeroRefRecord(ScalarView):
    """Immutable scalar view of a reference bacterial antigen locus record in `SeroRefRecords`.

    Attributes:
        organism: Canonical bacterial species name (e.g. `b"Escherichia coli"`).
        antigen_type: Surface polysaccharide antigen family (`b"O-antigen"`, `b"CPS"`, `b"WTA"`, `b"EPS"`).
        serotype: Canonical serotype designation (e.g. `b"O1"`, `b"K1"`, `b"KL3"`).
        strain: Primary reference strain identifier (e.g. `b"NCTC 9001"`, `b"ATCC 25238"`, `b"PrK 1/57"`).
        strain_synonyms: All equivalent culture-collection and lab designations from StrainInfo v2.
        assembly_id: INSDC genome assembly or nucleotide accession (e.g. `b"GCA_900706755.1"`, `b"KY710685.1"`).
        locus: Predicted [`Locus`][seroref.loci.models.Locus] biosynthetic gene cluster.
        ncbi_taxid: NCBI Taxonomy ID integer, or `0` if unknown.
        straininfo_id: DSMZ StrainInfo `SI-ID` integer, or `0` if unknown.
        bacdive_id: DSMZ BacDive ID integer, or `0` if unknown.
        pmids: Tuple of PubMed IDs associated with this reference strain or locus.
    """

    organism: bytes
    antigen_type: bytes
    serotype: bytes
    strain: bytes = b""
    strain_synonyms: tuple[bytes, ...] = ()
    assembly_id: bytes = b""
    locus: Locus | None = None
    ncbi_taxid: int = 0
    straininfo_id: int = 0
    bacdive_id: int = 0
    pmids: tuple[int, ...] = ()

    @property
    def archetype(self) -> LocusArchetype:
        """Return the `LocusArchetype` of the reference `locus`, or `LocusArchetype.UNKNOWN`."""
        return self.locus.archetype if self.locus is not None else LocusArchetype.UNKNOWN

    @property
    def has_locus(self) -> bool:
        """True if this reference record holds a non-empty `Locus`."""
        return self.locus is not None and len(self.locus.nodes) > 0

    @property
    def compound_key(self) -> bytes:
        """Unique composite reference key `b"<organism>|<antigen_type>|<serotype>|<assembly_id>"`."""
        return b"%s|%s|%s|%s" % (self.organism, self.antigen_type, self.serotype, self.assembly_id)
