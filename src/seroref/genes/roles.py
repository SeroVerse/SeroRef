"""Canonical biosynthetic gene roles and Structure-of-Arrays (SoA) metadata specifications."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Self, overload

import numpy as np
from seroverse.core.collections import RaggedArrayContainer, ScalarView, lengths_to_offsets


class RoleCategory(IntEnum):
    """Functional category of a biosynthetic gene role."""

    TRANSPORT = 0
    POLYMER_ASSEMBLY = 1
    MODIFIER = 2
    PRECURSOR_SYNTHESIS = 3
    MOBILE_ELEMENT = 4
    BACKGROUND = 5

    @property
    def label(self) -> bytes:
        """Canonical ASCII byte-string label for this category."""
        return _ROLE_CATEGORY_LABELS[int(self)]

    @classmethod
    def from_name(cls, key: str | bytes | int | RoleCategory) -> RoleCategory:
        """Resolve a `RoleCategory` by integer ID, enum name, or label."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        b_clean = b.strip().lower().replace(b"-", b"").replace(b"_", b"").replace(b" ", b"")
        for cat in cls:
            if b_clean in (
                cat.name.encode("ascii").lower().replace(b"_", b""),
                cat.label.lower().replace(b"_", b""),
            ):
                return cat
        raise KeyError(f"Unknown RoleCategory identifier: {key!r}")

    @classmethod
    def resolve(cls, key: str | bytes | int | RoleCategory, default: RoleCategory | None = None) -> RoleCategory:
        """Resolve a `RoleCategory` identifier, falling back to `default` (or `RoleCategory.BACKGROUND`)."""
        fallback = cls.BACKGROUND if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback


_ROLE_CATEGORY_LABELS: tuple[bytes, ...] = (
    b"Transport",
    b"PolymerAssembly",
    b"Modifier",
    b"PrecursorSynthesis",
    b"MobileElement",
    b"Background",
)


@dataclass(frozen=True, slots=True)
class RoleSpec(ScalarView):
    """Immutable scalar view of a canonical `GeneRole` metadata specification."""

    name: bytes
    abbreviation: bytes
    category: RoleCategory
    description: bytes
    aliases: tuple[bytes, ...] = ()
    is_machinery: bool = False
    null_prior: float = 0.005

    @property
    def is_mobile_element(self) -> bool:
        """True if this role is a mobile genetic element (`TRANSPOSASE`, `INTEGRASE`)."""
        return self.category == RoleCategory.MOBILE_ELEMENT

    @property
    def is_biosynthetic(self) -> bool:
        """True if this role represents active locus biosynthesis, modification, or transport."""
        return self.category not in (RoleCategory.MOBILE_ELEMENT, RoleCategory.BACKGROUND)


@dataclass(frozen=True, slots=True)
class RoleSpecs(RaggedArrayContainer[RoleSpec, "RoleSpecs"]):
    """Pure CSR-packed Ragged Structure-of-Arrays (SoA) batch container over `RoleSpec` definitions."""

    names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    abbreviations: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    categories: np.ndarray[tuple[int], np.dtype[np.int8]]
    descriptions: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    is_mobile_element: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_biosynthetic: np.ndarray[tuple[int], np.dtype[np.bool_]]
    is_machinery: np.ndarray[tuple[int], np.dtype[np.bool_]]
    null_priors: np.ndarray[tuple[int], np.dtype[np.float64]]
    feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    alias_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[RoleSpec]:
        for i in range(len(self)):
            yield self[i]

    @property
    def biosynthetic_mask(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean mask (`(N,)`) marking active biosynthetic/transport/modifier gene roles."""
        return self.is_biosynthetic

    @property
    def machinery_roles(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array of role indices marked as core export/polymerization machinery."""
        return np.flatnonzero(self.is_machinery).astype(np.int32)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> RoleSpec: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> RoleSpec | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> RoleSpec | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"RoleSpecs index {item} out of range for length {n}.")
            off, ln = int(self.offsets[idx]), int(self.lengths[idx])
            aliases = tuple(bytes(self.alias_values[k]) for k in range(off, off + ln))
            return RoleSpec(
                name=bytes(self.names[idx]),
                abbreviation=bytes(self.abbreviations[idx]),
                category=RoleCategory(int(self.categories[idx])),
                description=bytes(self.descriptions[idx]),
                aliases=aliases,
                is_machinery=bool(self.is_machinery[idx]),
                null_prior=float(self.null_priors[idx]),
            )

        idx_pool = np.arange(len(self.alias_values), dtype=np.int32)
        indices, new_offsets, new_lengths, (gathered_idx,) = self._slice_ragged(item, idx_pool)
        if len(indices) == 0:
            return self.empty()
        alias_v = self.alias_values[gathered_idx] if len(gathered_idx) > 0 else np.empty(0, dtype="S1")
        return type(self)(
            names=self.names[indices],
            abbreviations=self.abbreviations[indices],
            categories=self.categories[indices],
            descriptions=self.descriptions[indices],
            is_mobile_element=self.is_mobile_element[indices],
            is_biosynthetic=self.is_biosynthetic[indices],
            is_machinery=self.is_machinery[indices],
            null_priors=self.null_priors[indices],
            feature_matrix=self.feature_matrix[indices],
            alias_values=alias_v,
            offsets=new_offsets,
            lengths=new_lengths,
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `RoleSpecs` container."""
        n_feat = len(RoleCategory) + 1
        return cls(
            names=np.empty(0, dtype="S1"),
            abbreviations=np.empty(0, dtype="S1"),
            categories=np.empty(0, dtype=np.int8),
            descriptions=np.empty(0, dtype="S1"),
            is_mobile_element=np.empty(0, dtype=np.bool_),
            is_biosynthetic=np.empty(0, dtype=np.bool_),
            is_machinery=np.empty(0, dtype=np.bool_),
            null_priors=np.empty(0, dtype=np.float64),
            feature_matrix=np.empty((0, n_feat), dtype=np.float32),
            alias_values=np.empty(0, dtype="S1"),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_list(cls, specs: Sequence[RoleSpec]) -> Self:
        """Compile a sequence of `RoleSpec` scalar views into a frozen `RoleSpecs` SoA container."""
        n = len(specs)
        if n == 0:
            return cls.empty()

        names = np.array([s.name for s in specs], dtype="S24")
        abbreviations = np.array([s.abbreviation for s in specs], dtype="S8")
        categories = np.array([int(s.category) for s in specs], dtype=np.int8)
        descriptions = np.array([s.description for s in specs], dtype="S")
        is_mobile = np.array([s.is_mobile_element for s in specs], dtype=np.bool_)
        is_biosyn = np.array([s.is_biosynthetic for s in specs], dtype=np.bool_)
        is_mach = np.array([s.is_machinery for s in specs], dtype=np.bool_)

        null_priors = np.array([s.null_prior for s in specs], dtype=np.float64)
        prior_sum = float(null_priors.sum())
        if prior_sum > 0.0:
            null_priors /= prior_sum

        n_cats = len(RoleCategory)
        feature_matrix = np.zeros((n, n_cats + 1), dtype=np.float32)
        feature_matrix[np.arange(n), np.clip(categories, 0, n_cats - 1)] = 1.0
        for i, s in enumerate(specs):
            if s.name in (b"POLYMERASE", b"TRANSMEMBRANE_SYNTHASE", b"GLYCOSYLTRANSFERASE"):
                feature_matrix[i, -1] = 1.0

        alias_flat: list[bytes] = []
        alias_lens = np.empty(n, dtype=np.int32)
        for i, s in enumerate(specs):
            alias_lens[i] = len(s.aliases)
            alias_flat.extend(s.aliases)
        alias_values = np.array(alias_flat, dtype="S") if alias_flat else np.empty(0, dtype="S1")
        offsets = lengths_to_offsets(alias_lens)

        for arr in (
            names,
            abbreviations,
            categories,
            descriptions,
            is_mobile,
            is_biosyn,
            is_mach,
            null_priors,
            feature_matrix,
            alias_values,
            offsets,
            alias_lens,
        ):
            arr.setflags(write=False)

        return cls(
            names=names,
            abbreviations=abbreviations,
            categories=categories,
            descriptions=descriptions,
            is_mobile_element=is_mobile,
            is_biosynthetic=is_biosyn,
            is_machinery=is_mach,
            null_priors=null_priors,
            feature_matrix=feature_matrix,
            alias_values=alias_values,
            offsets=offsets,
            lengths=alias_lens,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `RoleSpecs` containers."""
        items: list[RoleSpec] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)


class GeneRole(IntEnum):
    """Canonical biosynthetic gene roles (0..29) for surface polysaccharide loci.

    Subclasses `IntEnum` for zero-overhead integer semantics in bitfields (`IntNode`),
    32-bit presence masks (`1 << role`), Numba kernels, and `CSR` matrices.
    """

    FLIPPASE = 0
    POLYMERASE = 1
    COPOLYMERASE = 2
    KINASE = 3
    PHOSPHATASE = 4
    ABC_PERMEASE = 5
    ABC_ATPASE = 6
    LIGASE = 7
    TRANSLOCON = 8
    CAPSULE_ANCHOR = 9
    PYRUVYLTRANSFERASE = 10
    METHYLTRANSFERASE = 11
    ACETYLTRANSFERASE = 12
    PHOSPHOTRANSFERASE = 13
    FORMYLTRANSFERASE = 14
    ALANINE_TRANSFERASE = 15
    AMINOTRANSFERASE = 16
    ALDOLASE = 17
    DEACETYLASE = 18
    EPIMERASE = 19
    OXIDOREDUCTASE = 20
    TRANSMEMBRANE_SYNTHASE = 21
    NUCLEOTIDYLTRANSFERASE = 22
    NUCLEOTIDASE = 23
    INITIATOR = 24
    GLYCOSYLHYDROLASE = 25
    GLYCOSYLTRANSFERASE = 26
    TRANSPOSASE = 27
    INTEGRASE = 28
    UNKNOWN = 29

    @property
    def spec(self) -> RoleSpec:
        """Return the immutable `RoleSpec` scalar view for this role."""
        return self.specs()[int(self)]

    @property
    def abbreviation(self) -> bytes:
        """Short display abbreviation (e.g. `b"Wzx"`, `b"GT"`)."""
        return bytes(self.specs().abbreviations[int(self)])

    @property
    def category(self) -> RoleCategory:
        """Functional `RoleCategory` of this gene role."""
        return RoleCategory(int(self.specs().categories[int(self)]))

    @property
    def description(self) -> bytes:
        """Human-readable biochemical description."""
        return bytes(self.specs().descriptions[int(self)])

    @property
    def aliases(self) -> tuple[bytes, ...]:
        """Known gene symbol aliases for this role."""
        return self.spec.aliases

    @property
    def bitmask(self) -> int:
        """32-bit integer bitmask (`1 << int(self)`) for archetype presence masks."""
        return 1 << int(self)

    @property
    def is_mobile_element(self) -> bool:
        """True if this role is a mobile genetic element (`TRANSPOSASE`, `INTEGRASE`)."""
        return bool(self.specs().is_mobile_element[int(self)])

    @property
    def is_biosynthetic(self) -> bool:
        """True if this role represents active locus biosynthesis, modification, or transport."""
        return bool(self.specs().is_biosynthetic[int(self)])

    @property
    def is_machinery(self) -> bool:
        """True if this role is core export, translocation, or polymerization machinery."""
        return bool(self.specs().is_machinery[int(self)])

    @classmethod
    def specs(cls) -> RoleSpecs:
        """Return the cached singleton `RoleSpecs` SoA container over all `GeneRole` members."""
        return _get_role_specs()

    @classmethod
    def from_name(cls, key: str | bytes | int | GeneRole) -> GeneRole:
        """Resolve a role from an integer index, canonical name, abbreviation, or alias."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        b_clean = b.strip()
        lookup = _get_role_lookup()
        if b_clean in lookup:
            return lookup[b_clean]
        b_lower = b_clean.lower()
        if b_lower in lookup:
            return lookup[b_lower]
        raise KeyError(f"Unknown GeneRole identifier: {key!r}")

    @classmethod
    def resolve(cls, key: str | bytes | int | GeneRole, default: GeneRole | None = None) -> GeneRole:
        """Resolve a role identifier, falling back to `default` (or `GeneRole.UNKNOWN`) if not found."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback

    @classmethod
    def names_array(cls) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """Contiguous 1D NumPy byte-string array of canonical role names."""
        return cls.specs().names

    @classmethod
    def abbreviations_array(cls) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """Contiguous 1D NumPy byte-string array of role abbreviations."""
        return cls.specs().abbreviations

    @classmethod
    def categories_array(cls) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        """Contiguous 1D `int8` NumPy array of `RoleCategory` IDs."""
        return cls.specs().categories

    @classmethod
    def feature_matrix(cls) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Return precomputed `(30, D)` float32 one-hot category feature matrix for embeddings."""
        return cls.specs().feature_matrix


@lru_cache(maxsize=1)
def _get_role_specs() -> RoleSpecs:
    """Build and cache the singleton `RoleSpecs` SoA container over all `GeneRole` members (0..29)."""
    return RoleSpecs.from_list(
        (
            # 0: FLIPPASE
            RoleSpec(
                name=b"FLIPPASE",
                abbreviation=b"Wzx",
                category=RoleCategory.TRANSPORT,
                description=b"Wzx flippase; translocates completed lipid-linked repeat units across inner membrane.",
                aliases=(b"wzx", b"gmla"),
                is_machinery=True,
            ),
            # 1: POLYMERASE
            RoleSpec(
                name=b"POLYMERASE",
                abbreviation=b"Wzy",
                category=RoleCategory.POLYMER_ASSEMBLY,
                description=b"Wzy polymerase; forms the polymer repeat unit boundary glycosidic linkage.",
                aliases=(b"wzy",),
                is_machinery=True,
            ),
            # 2: COPOLYMERASE
            RoleSpec(
                name=b"COPOLYMERASE",
                abbreviation=b"Wzz",
                category=RoleCategory.TRANSPORT,
                description=b"Wzz polysaccharide co-polymerase; modal chain length regulator.",
                aliases=(b"wzz", b"cld"),
                is_machinery=True,
            ),
            # 3: KINASE
            RoleSpec(
                name=b"KINASE",
                abbreviation=b"Kin",
                category=RoleCategory.TRANSPORT,
                description=b"Wzc / BY-kinase; tyrosine autokinase coordinating polymer export.",
                aliases=(b"wzc", b"ptk"),
                is_machinery=True,
            ),
            # 4: PHOSPHATASE
            RoleSpec(
                name=b"PHOSPHATASE",
                abbreviation=b"Wzb",
                category=RoleCategory.TRANSPORT,
                description=b"Wzb / PTP / CpsACP; low-molecular-weight phosphotyrosine and acid phosphatases.",
                aliases=(b"wzb", b"ptp", b"cpsacp", b"cpsb"),
                is_machinery=True,
            ),
            # 5: ABC_PERMEASE
            RoleSpec(
                name=b"ABC_PERMEASE",
                abbreviation=b"Wzm",
                category=RoleCategory.TRANSPORT,
                description=b"Wzm / KpsM ABC transporter permease; transmembrane translocation channel.",
                aliases=(b"wzm", b"kpsm"),
                is_machinery=True,
            ),
            # 6: ABC_ATPASE
            RoleSpec(
                name=b"ABC_ATPASE",
                abbreviation=b"Wzt",
                category=RoleCategory.TRANSPORT,
                description=b"Wzt / KpsT ABC transporter ATPase; nucleotide-binding domain powering export.",
                aliases=(b"wzt", b"kpst"),
                is_machinery=True,
            ),
            # 7: LIGASE
            RoleSpec(
                name=b"LIGASE",
                abbreviation=b"WaaL",
                category=RoleCategory.POLYMER_ASSEMBLY,
                description=b"WaaL ligase; attaches periplasmic polysaccharide to Lipid A core.",
                aliases=(b"waal", b"rfal"),
                is_machinery=True,
            ),
            # 8: TRANSLOCON
            RoleSpec(
                name=b"TRANSLOCON",
                abbreviation=b"Wza",
                category=RoleCategory.TRANSPORT,
                description=b"Wza / KpsD / KpsE outer membrane and periplasmic translocon complex for capsular polysaccharide surface export.",
                aliases=(b"wza", b"gfcd", b"gfcb", b"gfcc", b"kpsd", b"kpse"),
                is_machinery=True,
            ),
            # 9: CAPSULE_ANCHOR
            RoleSpec(
                name=b"CAPSULE_ANCHOR",
                abbreviation=b"Wzi",
                category=RoleCategory.TRANSPORT,
                description=b"Wzi outer membrane lectin / KpsC lipid anchor / Wzg (CpsA) LytR-CpsA-Psr peptidoglycan capsule anchor transferase.",
                aliases=(b"wzi", b"anchor", b"capsule_anchor", b"kpsc", b"wzg", b"cpsa", b"cps2a", b"capa"),
                is_machinery=True,
            ),
            # 10: PYRUVYLTRANSFERASE
            RoleSpec(
                name=b"PYRUVYLTRANSFERASE",
                abbreviation=b"PT",
                category=RoleCategory.MODIFIER,
                description=b"Pyruvyltransferase; adds pyruvate acetal (Pyr) modifications.",
                aliases=(b"wcak", b"ptr", b"wbmx"),
            ),
            # 11: METHYLTRANSFERASE
            RoleSpec(
                name=b"METHYLTRANSFERASE",
                abbreviation=b"MetT",
                category=RoleCategory.MODIFIER,
                description=b"Methyltransferase; adds O-methyl (OMe) groups.",
            ),
            # 12: ACETYLTRANSFERASE
            RoleSpec(
                name=b"ACETYLTRANSFERASE",
                abbreviation=b"AT",
                category=RoleCategory.MODIFIER,
                description=b"Acetyltransferase / Acyltransferase; adds O-acetyl (OAc) or hydroxyacyl (3HOBut, 3HOMyr) groups.",
                aliases=(b"oata", b"wecd", b"atr", b"qdte", b"wcio"),
            ),
            # 13: PHOSPHOTRANSFERASE
            RoleSpec(
                name=b"PHOSPHOTRANSFERASE",
                abbreviation=b"PhoT",
                category=RoleCategory.MODIFIER,
                description=b"Phosphotransferase; adds phosphate (P), phosphoethanolamine (PEtn), or phosphocholine (ChoP) groups.",
                aliases=(b"licd", b"pgt"),
            ),
            # 14: FORMYLTRANSFERASE
            RoleSpec(
                name=b"FORMYLTRANSFERASE",
                abbreviation=b"ForT",
                category=RoleCategory.MODIFIER,
                description=b"Formyltransferase; adds formyl (Fo) modifications.",
                aliases=(b"qdtf", b"fdtd", b"vioa"),
            ),
            # 15: ALANINE_TRANSFERASE
            RoleSpec(
                name=b"ALANINE_TRANSFERASE",
                abbreviation=b"Alt",
                category=RoleCategory.MODIFIER,
                description=b"Alt1 alanine transferase; transfers D-alanine to the carboxyl group of GalA/GalNAcA forming an amide.",
                aliases=(b"alt", b"alt1", b"ala_transferase"),
            ),
            # 16: AMINOTRANSFERASE
            RoleSpec(
                name=b"AMINOTRANSFERASE",
                abbreviation=b"ATn",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"Aminotransferase / Transaminase (e.g. WecE, RhnB, RhnC); converts keto-intermediates to amino sugars.",
                aliases=(b"wece", b"rhnb", b"rhnc"),
            ),
            # 17: ALDOLASE
            RoleSpec(
                name=b"ALDOLASE",
                abbreviation=b"Ald",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"PEP-condensing aldolase/synthase (e.g. NeuB, PseI, LegI, KdsA); synthesizes nonulosonic and ulosonic acids.",
                aliases=(b"neub", b"psei", b"legi", b"kdsa"),
            ),
            # 18: DEACETYLASE
            RoleSpec(
                name=b"DEACETYLASE",
                abbreviation=b"DAc",
                category=RoleCategory.MODIFIER,
                description=b"Deacetylase; generates free amino sugar derivatives.",
                aliases=(b"pgab", b"pda1", b"pda2", b"pda2a", b"ahy"),
            ),
            # 19: EPIMERASE
            RoleSpec(
                name=b"EPIMERASE",
                abbreviation=b"Epi",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"Epimerase / Isomerase / Mutase; interconverts UDP-sugar stereoisomers and hexose-phosphate intermediates.",
                aliases=(b"gale", b"rmlc", b"neuc", b"gne", b"rfbc", b"kpsf", b"manb", b"pgm", b"gpi", b"glf"),
            ),
            # 20: OXIDOREDUCTASE
            RoleSpec(
                name=b"OXIDOREDUCTASE",
                abbreviation=b"OxR",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"Reductase / Dehydratase / Dehydrogenase (e.g. RmlB, RmlD, Ugd, Gnd, Gmd, Fcl); synthesizes deoxyhexoses, dideoxyhexoses, and uronic acids.",
                aliases=(b"rmlb", b"rmld", b"rhna", b"epab", b"epad", b"rfbb", b"rfbd", b"ugd", b"gnd", b"gmd", b"fcl"),
            ),
            # 21: TRANSMEMBRANE_SYNTHASE
            RoleSpec(
                name=b"TRANSMEMBRANE_SYNTHASE",
                abbreviation=b"TMS",
                category=RoleCategory.POLYMER_ASSEMBLY,
                description=b"Processive transmembrane glycosyltransferase (e.g. HasA, PgaC, BcsA); polymerizes and extrudes glycan chains.",
                aliases=(b"hasa", b"pgac", b"bcsa"),
                is_machinery=True,
            ),
            # 22: NUCLEOTIDYLTRANSFERASE
            RoleSpec(
                name=b"NUCLEOTIDYLTRANSFERASE",
                abbreviation=b"NT",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"Nucleotide-sugar precursor synthase (e.g. GalF, GalU, RmlA, ManC, NeuA, KdsB); acts on base monosaccharide pathways.",
                aliases=(b"rmla", b"galu", b"galf", b"manc", b"neua", b"kdsb", b"kpsu", b"epaa", b"epac"),
            ),
            # 23: NUCLEOTIDASE
            RoleSpec(
                name=b"NUCLEOTIDASE",
                abbreviation=b"Nuc",
                category=RoleCategory.PRECURSOR_SYNTHESIS,
                description=b"Sugar-nucleotide nucleotidase / hydrolase (e.g. PsaD, PsaG, PsaI, LgaF, Gmm, WcaH); cleaves sugar-nucleotide intermediates.",
                aliases=(b"psad", b"psag", b"psai", b"lgaf", b"gmm", b"wcah", b"sugar_phosphatase"),
            ),
            # 24: INITIATOR
            RoleSpec(
                name=b"INITIATOR",
                abbreviation=b"Init",
                category=RoleCategory.POLYMER_ASSEMBLY,
                description=b"Initial sugar transferase (e.g. WbaP, WcaJ, WchA, ItrA, WecA); attaches first non-modifier sugar onto undecaprenyl phosphate.",
                aliases=(b"wbap", b"wcaj", b"wcha", b"itra", b"itrb", b"weca", b"initiating_gt"),
                is_machinery=True,
            ),
            # 25: GLYCOSYLHYDROLASE
            RoleSpec(
                name=b"GLYCOSYLHYDROLASE",
                abbreviation=b"GH",
                category=RoleCategory.MODIFIER,
                description=b"Glycosylhydrolase / lyase; processes, depolymerizes, or cleaves glycosidic linkages.",
                aliases=(b"gnlb", b"ghl", b"orf1(ghy)", b"ghy"),
            ),
            # 26: GLYCOSYLTRANSFERASE
            RoleSpec(
                name=b"GLYCOSYLTRANSFERASE",
                abbreviation=b"GT",
                category=RoleCategory.POLYMER_ASSEMBLY,
                description=b"Glycosyltransferase; synthesizes glycosidic linkages and non-modifier monosaccharide additions.",
                aliases=(
                    b"kpss",
                    b"gtr",
                    b"glt",
                    b"htrl",
                    b"wbmv",
                    b"wbmw",
                    b"gmlc",
                    b"spsa",
                    b"wcaa",
                    b"rhamnan",
                    b"rgpf",
                ),
            ),
            # 27: TRANSPOSASE
            RoleSpec(
                name=b"TRANSPOSASE",
                abbreviation=b"Tnp",
                category=RoleCategory.MOBILE_ELEMENT,
                description=b"Transposase / Insertion sequence (IS) element; mobile genetic element non-biosynthetic infiltrator.",
                aliases=(b"tnp", b"transposase", b"is_element", b"insertion_sequence"),
                null_prior=0.015,
            ),
            # 28: INTEGRASE
            RoleSpec(
                name=b"INTEGRASE",
                abbreviation=b"Int",
                category=RoleCategory.MOBILE_ELEMENT,
                description=b"Integrase / Recombinase / Excisionase; mediates DNA integration/excision, prophage insertions, or mobile cassette movement.",
                aliases=(b"int", b"integrase", b"recombinase", b"excisionase"),
                null_prior=0.015,
            ),
            # 29: UNKNOWN
            RoleSpec(
                name=b"UNKNOWN",
                abbreviation=b"?",
                category=RoleCategory.BACKGROUND,
                description=b"Uncharacterized gene; background or non-biosynthetic role.",
                null_prior=0.825,
            ),
        )
    )


@lru_cache(maxsize=1)
def _get_role_lookup() -> dict[bytes, GeneRole]:
    """Build and cache the case-insensitive name/abbreviation/alias lookup dictionary for `GeneRole`."""
    specs = _get_role_specs()
    lookup: dict[bytes, GeneRole] = {}
    for role in GeneRole:
        spec = specs[int(role)]
        lookup[spec.name] = role
        lookup[spec.name.lower()] = role
        lookup[spec.abbreviation] = role
        lookup[spec.abbreviation.lower()] = role
        lookup[spec.abbreviation.upper()] = role
        for al in spec.aliases:
            lookup[al.strip().lower()] = role
    return lookup
