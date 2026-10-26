"""OOP scalar data models for `ReferenceStrain` and `SeroRefRecord`."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

from seroverse.core.collections import ScalarView

from seroref.db.species import BacterialSpecies
from seroref.loci.archetypes import ArchetypeCategory, LocusArchetype
from seroref.loci.models import Loci, Locus

INVARIANT_ARCHETYPE_GLYCAN_NAMES: dict[LocusArchetype, bytes] = {
    LocusArchetype.EPS_PEL: b"Pel",
    LocusArchetype.EPS_PGA: b"PNAG",
    LocusArchetype.EPS_ALGINATE: b"Alginate",
    LocusArchetype.EPS_BCS: b"Cellulose",
}


def archetype_to_antigen_type(archetype: LocusArchetype) -> bytes:
    """Map a `LocusArchetype` to its canonical surface polysaccharide antigen class (`CPS`, `O-antigen`, `EPS`, `WTA`)."""
    cat = archetype.category
    if cat == ArchetypeCategory.CAPSULE:
        return b"CPS"
    if cat == ArchetypeCategory.EXOPOLYSACCHARIDE:
        return b"EPS"
    if archetype == LocusArchetype.TEICHOIC_ACID:
        return b"WTA"
    return b"O-antigen"


@dataclass(frozen=True, slots=True)
class ReferenceStrain(ScalarView):
    """Immutable scalar view of a reference bacterial strain isolate and genome assembly.

    Attributes:
        species: Canonical [`BacterialSpecies`][seroref.db.species.BacterialSpecies] enum member.
        name: Normalized primary reference strain designation (e.g. `b"NCTC 9001"`, `b"ATCC 25238"`).
        synonyms: Culture-collection and laboratory strain equivalence clique from DSMZ StrainInfo v2.
        assembly_id: INSDC genome assembly or nucleotide accession (e.g. `b"GCA_900706755.1"`).
        ncbi_taxid: Strain-specific or species-level NCBI Taxonomy ID integer.
        straininfo_id: DSMZ StrainInfo `SI-ID` integer, or `0` if unknown.
        bacdive_id: DSMZ BacDive ID integer, or `0` if unknown.
        pmids: Tuple of PubMed IDs associated with this reference strain or assembly.
    """

    species: BacterialSpecies
    name: bytes = b""
    synonyms: tuple[bytes, ...] = ()
    assembly_id: bytes = b""
    ncbi_taxid: int = 0
    straininfo_id: int = 0
    bacdive_id: int = 0
    pmids: tuple[int, ...] = ()

    @property
    def organism(self) -> bytes:
        """Canonical scientific name byte string from `self.species`."""
        return self.species.scientific_name

    @property
    def effective_taxid(self) -> int:
        """Resolved NCBI Taxonomy ID, falling back to `self.species.ncbi_taxid` when `self.ncbi_taxid == 0`."""
        return self.ncbi_taxid if self.ncbi_taxid > 0 else self.species.ncbi_taxid

    @property
    def compound_key(self) -> bytes:
        """Unique composite strain-assembly key `b"<organism>|<name>|<assembly_id>"`."""
        return b"%s|%s|%s" % (self.organism, self.name, self.assembly_id)

    @classmethod
    def from_raw(
        cls,
        organism: str | bytes | int | BacterialSpecies,
        strain: str | bytes = b"",
        *,
        raw_serotype: str | bytes = b"",
        synonyms: Sequence[bytes | str] = (),
        assembly_id: bytes | str = b"",
        ncbi_taxid: int = 0,
        straininfo_id: int = 0,
        bacdive_id: int = 0,
        pmids: Sequence[int] = (),
    ) -> Self:
        """Construct a normalized `ReferenceStrain` using `BacterialSpecies` rules."""
        sp = BacterialSpecies.resolve(organism)
        if sp == BacterialSpecies.UNKNOWN and ncbi_taxid > 0:
            sp = BacterialSpecies.from_taxid(ncbi_taxid)
        norm_name = sp.normalize_strain(strain, raw_serotype)
        asm_b = assembly_id.encode("utf-8") if isinstance(assembly_id, str) else bytes(assembly_id)
        syn_list: list[bytes] = []
        if norm_name:
            syn_list.append(norm_name)
        for s in synonyms:
            sb = s.encode("utf-8").strip() if isinstance(s, str) else bytes(s).strip()
            if sb and sb not in syn_list:
                syn_list.append(sb)
        pm_tuple = tuple(dict.fromkeys(int(p) for p in pmids if int(p) > 0))
        return cls(
            species=sp,
            name=norm_name,
            synonyms=tuple(syn_list),
            assembly_id=asm_b,
            ncbi_taxid=int(ncbi_taxid) if int(ncbi_taxid) > 0 else sp.ncbi_taxid,
            straininfo_id=max(0, int(straininfo_id)),
            bacdive_id=max(0, int(bacdive_id)),
            pmids=pm_tuple,
        )

    def with_passport(self, passport: Any, *, raw_serotype: str | bytes = b"") -> Self:
        """Enrich this `ReferenceStrain` with culture-collection synonyms and IDs from a `StrainPassport`."""
        if passport is None:
            return self
        des_bytes = [d.encode("utf-8") for d in getattr(passport, "designations", ()) if d]
        norm_name = self.name
        if not norm_name and des_bytes:
            norm_name = self.species.normalize_strain(des_bytes[0], raw_serotype)
        syns = tuple(dict.fromkeys((*((norm_name,) if norm_name else ()), *self.synonyms, *des_bytes)))
        p_pmids = [int(p) for p in getattr(passport, "pmids", ()) if int(p) > 0]
        merged_pmids = tuple(dict.fromkeys((*self.pmids, *p_pmids)))
        p_tax = int(getattr(passport, "ncbi_taxid", 0) or 0)
        p_si = int(getattr(passport, "si_id", 0) or 0)
        p_bd = int(getattr(passport, "bacdive_id", 0) or 0)
        return type(self)(
            species=self.species,
            name=norm_name,
            synonyms=syns,
            assembly_id=self.assembly_id,
            ncbi_taxid=self.ncbi_taxid or p_tax or self.species.ncbi_taxid,
            straininfo_id=self.straininfo_id or p_si,
            bacdive_id=self.bacdive_id or p_bd,
            pmids=merged_pmids,
        )

    def merge(self, other: ReferenceStrain) -> Self:
        """Merge metadata from another `ReferenceStrain` describing the same isolate or assembly."""
        sp = self.species if self.species != BacterialSpecies.UNKNOWN else other.species
        name = self.name or other.name
        syns = tuple(dict.fromkeys((*self.synonyms, *other.synonyms)))
        pmids = tuple(dict.fromkeys((*self.pmids, *other.pmids)))
        return type(self)(
            species=sp,
            name=name,
            synonyms=syns,
            assembly_id=self.assembly_id or other.assembly_id,
            ncbi_taxid=self.ncbi_taxid or other.ncbi_taxid or sp.ncbi_taxid,
            straininfo_id=self.straininfo_id or other.straininfo_id,
            bacdive_id=self.bacdive_id or other.bacdive_id,
            pmids=pmids,
        )

    def bind_loci(
        self,
        loci: Loci,
        raw_serotype: str | bytes = b"",
    ) -> list[SeroRefRecord]:
        """Bind a batch of predicted `Loci` on this reference assembly into `SeroRefRecord` rows.

        Splits multi-antigen formulas (e.g. `'O1:K1(L1):H7'`) across predicted `ArchetypeCategory`
        groups (`O-Antigen`, `Capsule`, `Exopolysaccharide`, `Other`), assigns invariant glycan names
        (`Pel`, `PNAG`, `Alginate`, `Cellulose`) to exopolysaccharide archetypes, and synchronizes
        `locus.name` with the resolved `serotype`.
        """
        if len(loci) == 0:
            return []

        strain_rec = (
            self
            if self.name or not raw_serotype
            else self.from_raw(
                self.species,
                self.name,
                raw_serotype=raw_serotype,
                synonyms=self.synonyms,
                assembly_id=self.assembly_id,
                ncbi_taxid=self.ncbi_taxid,
                straininfo_id=self.straininfo_id,
                bacdive_id=self.bacdive_id,
                pmids=self.pmids,
            )
        )

        # Keep highest-scoring locus per (ArchetypeCategory, LocusArchetype, start, end)
        best_by_cat: dict[ArchetypeCategory, list[Locus]] = defaultdict(list)
        seen_intervals: set[tuple[LocusArchetype, int, int]] = set()
        for i in range(len(loci)):
            loc = loci[i].canonicalize_strand()
            if loc.archetype == LocusArchetype.UNKNOWN or len(loc.nodes) == 0:
                continue
            if not strain_rec.species.spec.supports_archetype(loc.archetype):
                continue
            key = (loc.archetype, int(loc.start), int(loc.end))
            if key in seen_intervals:
                continue
            seen_intervals.add(key)
            best_by_cat[loc.archetype.category].append(loc)

        for cat_loci in best_by_cat.values():
            cat_loci.sort(key=lambda loc: (float(loc.completeness), float(loc.score), len(loc.nodes)), reverse=True)

        # Parse multi-antigen or single-antigen serotype tokens using species grammar
        primary_arch = next(
            (locs[0].archetype for locs in best_by_cat.values() if locs),
            LocusArchetype.UNKNOWN,
        )
        token_by_cat: dict[ArchetypeCategory, bytes] = {}
        for cat, tok_b in strain_rec.species.parse_serotype(raw_serotype, archetype=primary_arch):
            token_by_cat.setdefault(cat, tok_b)

        out: list[SeroRefRecord] = []
        for cat, cat_loci in best_by_cat.items():
            matched_sero = token_by_cat.get(cat, b"")
            for rank, loc in enumerate(cat_loci):
                sero_for_loc = matched_sero if rank == 0 else b""
                if not sero_for_loc and loc.archetype in INVARIANT_ARCHETYPE_GLYCAN_NAMES:
                    sero_for_loc = INVARIANT_ARCHETYPE_GLYCAN_NAMES[loc.archetype]
                if sero_for_loc:
                    parsed = strain_rec.species.parse_serotype(sero_for_loc, archetype=loc.archetype)
                    if parsed:
                        sero_for_loc = parsed[0][1]
                loc_named = loc.with_name(sero_for_loc)
                out.append(
                    SeroRefRecord(
                        strain_record=strain_rec,
                        serotype=sero_for_loc,
                        locus=loc_named,
                    )
                )
        return out


@dataclass(frozen=True, slots=True)
class SeroRefRecord(ScalarView):
    """Immutable scalar view of a reference bacterial antigen locus record in `SeroRefRecords`.

    Attributes:
        strain_record: Parent [`ReferenceStrain`][seroref.db.models.ReferenceStrain] isolate and assembly record.
        serotype: Canonical serotype or glycan name (e.g. `b"O1"`, `b"K1"`, `b"35B"`, `b"Pel"`, `b"PNAG"`).
        locus: Predicted [`Locus`][seroref.loci.models.Locus] biosynthetic gene cluster.
    """

    strain_record: ReferenceStrain
    serotype: bytes = b""
    locus: Locus | None = None

    @classmethod
    def from_fields(
        cls,
        organism: str | bytes | int | BacterialSpecies,
        *,
        serotype: str | bytes = b"",
        strain: str | bytes = b"",
        strain_synonyms: Sequence[bytes | str] = (),
        assembly_id: bytes | str = b"",
        locus: Locus | None = None,
        ncbi_taxid: int = 0,
        straininfo_id: int = 0,
        bacdive_id: int = 0,
        pmids: Sequence[int] = (),
    ) -> Self:
        """Construct a `SeroRefRecord` from flat organism, strain, serotype, and locus fields."""
        strain_rec = ReferenceStrain.from_raw(
            organism,
            strain,
            raw_serotype=serotype,
            synonyms=strain_synonyms,
            assembly_id=assembly_id,
            ncbi_taxid=ncbi_taxid,
            straininfo_id=straininfo_id,
            bacdive_id=bacdive_id,
            pmids=pmids,
        )
        arch = locus.archetype if locus is not None else None
        parsed = strain_rec.species.parse_serotype(serotype, archetype=arch)
        sero_b = parsed[0][1] if parsed else b""
        if not sero_b and arch is not None and arch in INVARIANT_ARCHETYPE_GLYCAN_NAMES:
            sero_b = INVARIANT_ARCHETYPE_GLYCAN_NAMES[arch]
        loc_named = locus.with_name(sero_b) if locus is not None else None
        return cls(
            strain_record=strain_rec,
            serotype=sero_b,
            locus=loc_named,
        )

    @property
    def species(self) -> BacterialSpecies:
        """Canonical `BacterialSpecies` enum member of this record."""
        return self.strain_record.species

    @property
    def organism(self) -> bytes:
        """Canonical bacterial species name (e.g. `b"Escherichia coli"`)."""
        return self.strain_record.organism

    @property
    def strain(self) -> bytes:
        """Primary reference strain designation (e.g. `b"NCTC 9001"`)."""
        return self.strain_record.name

    @property
    def strain_synonyms(self) -> tuple[bytes, ...]:
        """All equivalent culture-collection and lab designations from DSMZ StrainInfo v2."""
        return self.strain_record.synonyms

    @property
    def assembly_id(self) -> bytes:
        """INSDC genome assembly or nucleotide accession (e.g. `b"GCA_900706755.1"`)."""
        return self.strain_record.assembly_id

    @property
    def ncbi_taxid(self) -> int:
        """Resolved NCBI Taxonomy ID integer."""
        return self.strain_record.effective_taxid

    @property
    def straininfo_id(self) -> int:
        """DSMZ StrainInfo `SI-ID` integer, or `0` if unknown."""
        return self.strain_record.straininfo_id

    @property
    def bacdive_id(self) -> int:
        """DSMZ BacDive ID integer, or `0` if unknown."""
        return self.strain_record.bacdive_id

    @property
    def pmids(self) -> tuple[int, ...]:
        """Tuple of PubMed IDs associated with this reference strain or locus."""
        return self.strain_record.pmids

    @property
    def archetype(self) -> LocusArchetype:
        """Return the `LocusArchetype` of the reference `locus`, or `LocusArchetype.UNKNOWN`."""
        return self.locus.archetype if self.locus is not None else LocusArchetype.UNKNOWN

    @property
    def category(self) -> ArchetypeCategory:
        """Return the high-level `ArchetypeCategory` of the reference `locus`."""
        return self.archetype.category

    @property
    def antigen_type(self) -> bytes:
        """Surface polysaccharide antigen class (`b"CPS"`, `b"O-antigen"`, `b"EPS"`, `b"WTA"`) derived from `archetype`."""
        return archetype_to_antigen_type(self.archetype)

    @property
    def has_locus(self) -> bool:
        """True if this reference record holds a non-empty `Locus`."""
        return self.locus is not None and len(self.locus.nodes) > 0

    @property
    def compound_key(self) -> bytes:
        """Unique composite reference key `b"<organism>|<antigen_type>|<serotype>|<assembly_id>"`."""
        return b"%s|%s|%s|%s" % (self.organism, self.antigen_type, self.serotype, self.assembly_id)

    def merge(self, other: SeroRefRecord) -> Self:
        """Merge metadata from another `SeroRefRecord` describing the same physical locus."""
        merged_strain = self.strain_record.merge(other.strain_record)
        sero = self.serotype or other.serotype
        if merged_strain.name and sero and merged_strain.name.lower() == sero.lower():
            merged_strain = ReferenceStrain(
                species=merged_strain.species,
                name=b"",
                synonyms=merged_strain.synonyms,
                assembly_id=merged_strain.assembly_id,
                ncbi_taxid=merged_strain.ncbi_taxid,
                straininfo_id=merged_strain.straininfo_id,
                bacdive_id=merged_strain.bacdive_id,
                pmids=merged_strain.pmids,
            )
        base_loc = self.locus if self.locus is not None else other.locus
        loc = base_loc.with_name(sero) if base_loc is not None else None
        return type(self)(
            strain_record=merged_strain,
            serotype=sero,
            locus=loc,
        )
