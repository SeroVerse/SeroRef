"""Biosynthetic locus archetypes, ontology categories, and Structure-of-Arrays (SoA) specifications."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Self, overload

import numpy as np
from seroverse.core.collections import RaggedArrayContainer, ScalarView, lengths_to_offsets

from seroref.genes.roles import GeneRole


def roles_to_mask(roles: Iterable[int | GeneRole] | np.ndarray) -> int:
    """Pack a collection of `GeneRole` values into a 32-bit integer presence bitmask."""
    mask = 0
    if isinstance(roles, np.ndarray):
        if roles.size == 0:
            return 0
        valid = roles[(roles >= 0) & (roles < 30)]
        if valid.size == 0:
            return 0
        return int(np.bitwise_or.reduce(1 << valid.astype(np.int64)))

    for r in roles:
        val = int(r)
        if 0 <= val < 30:
            mask |= 1 << val
    return mask


def mask_to_roles(mask: int) -> tuple[GeneRole, ...]:
    """Unpack a 32-bit integer presence bitmask into a sorted tuple of `GeneRole` enums."""
    return tuple(role for role in GeneRole if (mask & (1 << int(role))) != 0)


class ArchetypeCategory(IntEnum):
    """High-level structural category of a polysaccharide biosynthetic locus archetype."""

    CAPSULE = 0
    O_ANTIGEN = 1
    EXOPOLYSACCHARIDE = 2
    OTHER = 3

    @property
    def label(self) -> bytes:
        """Canonical ASCII byte-string label for this category (e.g. `b"Capsule"`, `b"O-Antigen"`)."""
        return _ARCHETYPE_CATEGORY_LABELS[int(self)]

    @classmethod
    def from_name(cls, key: str | bytes | int | ArchetypeCategory) -> ArchetypeCategory:
        """Resolve an `ArchetypeCategory` by integer ID, enum name, or label."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        b_clean = b.strip().lower().replace(b"-", b"_").replace(b" ", b"_")
        for cat in cls:
            if b_clean in (
                cat.name.encode("ascii").lower(),
                cat.label.lower().replace(b"-", b"_"),
            ):
                return cat
        raise KeyError(f"Unknown ArchetypeCategory identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | ArchetypeCategory,
        default: ArchetypeCategory | None = None,
    ) -> ArchetypeCategory:
        """Resolve an `ArchetypeCategory` identifier, falling back to `default` (or `ArchetypeCategory.OTHER`)."""
        fallback = cls.OTHER if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback


_ARCHETYPE_CATEGORY_LABELS: tuple[bytes, ...] = (
    b"Capsule",
    b"O-Antigen",
    b"Exopolysaccharide",
    b"Other",
)


@dataclass(frozen=True, slots=True)
class ArchetypeSpec(ScalarView):
    """Immutable biological specification for a canonical `LocusArchetype`."""

    name: bytes
    category: ArchetypeCategory
    forbidden_mask: int
    cargo_mask: int
    description: bytes
    module_masks: tuple[int, ...] = ()
    aliases: tuple[bytes, ...] = ()

    @property
    def required_mask(self) -> int:
        """Combined 32-bit union mask of all roles participating in any required module."""
        mask = 0
        for m in self.module_masks:
            mask |= m
        return mask

    @property
    def all_relevant_mask(self) -> int:
        """Combined 32-bit bitmask of all required and optional cargo roles."""
        return self.required_mask | self.cargo_mask

    @property
    def num_modules(self) -> int:
        """Number of required functional modules (`M_A`) for this archetype."""
        return len(self.module_masks)


@dataclass(frozen=True, slots=True)
class ArchetypeSpecs(RaggedArrayContainer[ArchetypeSpec, "ArchetypeSpecs"]):
    """Pure CSR-packed Ragged Structure-of-Arrays (SoA) batch container over `ArchetypeSpec` definitions.

    Encapsulates all archetype metadata, 32-bit presence bitmasks, and role-indicator tensors required
    by [`ArchetypeClassifier`][seroref.loci.classifier.ArchetypeClassifier] and [`LocusSegmenter`][seroref.loci.segmenter.LocusSegmenter].
    """

    names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    categories: np.ndarray[tuple[int], np.dtype[np.int8]]
    descriptions: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    forbidden_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    cargo_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    required_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    all_relevant_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    module_masks: np.ndarray[tuple[int, int], np.dtype[np.uint32]]
    num_modules: np.ndarray[tuple[int], np.dtype[np.int32]]
    active_module_mask: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    module_indicators: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    forbidden_indicators: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    relevant_indicators: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    alias_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[ArchetypeSpec]:
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> ArchetypeSpec: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> ArchetypeSpec | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> ArchetypeSpec | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"ArchetypeSpecs index {item} out of range for length {n}.")
            n_mods = int(self.num_modules[idx])
            mod_tuple = tuple(int(self.module_masks[idx, m]) for m in range(n_mods))
            off, ln = int(self.offsets[idx]), int(self.lengths[idx])
            aliases = tuple(bytes(self.alias_values[k]) for k in range(off, off + ln))
            return ArchetypeSpec(
                name=bytes(self.names[idx]),
                category=ArchetypeCategory(int(self.categories[idx])),
                forbidden_mask=int(self.forbidden_masks[idx]),
                cargo_mask=int(self.cargo_masks[idx]),
                description=bytes(self.descriptions[idx]),
                module_masks=mod_tuple,
                aliases=aliases,
            )

        idx_pool = np.arange(len(self.alias_values), dtype=np.int32)
        indices, new_offsets, new_lengths, (gathered_idx,) = self._slice_ragged(item, idx_pool)
        if len(indices) == 0:
            return self.empty()
        alias_v = self.alias_values[gathered_idx] if len(gathered_idx) > 0 else np.empty(0, dtype="S1")
        return type(self)(
            names=self.names[indices],
            categories=self.categories[indices],
            descriptions=self.descriptions[indices],
            forbidden_masks=self.forbidden_masks[indices],
            cargo_masks=self.cargo_masks[indices],
            required_masks=self.required_masks[indices],
            all_relevant_masks=self.all_relevant_masks[indices],
            module_masks=self.module_masks[indices],
            num_modules=self.num_modules[indices],
            active_module_mask=self.active_module_mask[indices],
            module_indicators=self.module_indicators[indices],
            forbidden_indicators=self.forbidden_indicators[indices],
            relevant_indicators=self.relevant_indicators[indices],
            alias_values=alias_v,
            offsets=new_offsets,
            lengths=new_lengths,
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `ArchetypeSpecs` container."""
        n_roles = len(GeneRole)
        return cls(
            names=np.empty(0, dtype="S1"),
            categories=np.empty(0, dtype=np.int8),
            descriptions=np.empty(0, dtype="S1"),
            forbidden_masks=np.empty(0, dtype=np.uint32),
            cargo_masks=np.empty(0, dtype=np.uint32),
            required_masks=np.empty(0, dtype=np.uint32),
            all_relevant_masks=np.empty(0, dtype=np.uint32),
            module_masks=np.empty((0, 0), dtype=np.uint32),
            num_modules=np.empty(0, dtype=np.int32),
            active_module_mask=np.empty((0, 0), dtype=np.bool_),
            module_indicators=np.empty((0, 0, n_roles), dtype=np.float64),
            forbidden_indicators=np.empty((0, n_roles), dtype=np.float64),
            relevant_indicators=np.empty((0, n_roles), dtype=np.float64),
            alias_values=np.empty(0, dtype="S1"),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_list(cls, specs: Sequence[ArchetypeSpec]) -> Self:
        """Compile a sequence of `ArchetypeSpec` scalar views into a frozen `ArchetypeSpecs` SoA container."""
        n_arch = len(specs)
        if n_arch == 0:
            return cls.empty()

        n_roles = len(GeneRole)
        max_mods = max((s.num_modules for s in specs), default=0)

        names = np.array([s.name for s in specs], dtype="S16")
        categories = np.array([int(s.category) for s in specs], dtype=np.int8)
        descriptions = np.array([s.description for s in specs], dtype="S")
        forbidden_masks = np.array([s.forbidden_mask for s in specs], dtype=np.uint32)
        cargo_masks = np.array([s.cargo_mask for s in specs], dtype=np.uint32)
        required_masks = np.array([s.required_mask for s in specs], dtype=np.uint32)
        all_relevant_masks = np.array([s.all_relevant_mask for s in specs], dtype=np.uint32)

        module_masks = np.zeros((n_arch, max_mods), dtype=np.uint32)
        num_modules = np.zeros(n_arch, dtype=np.int32)
        for a_idx, spec in enumerate(specs):
            num_modules[a_idx] = spec.num_modules
            for m_idx, m_mask in enumerate(spec.module_masks):
                module_masks[a_idx, m_idx] = m_mask

        module_indicators = np.zeros((n_arch, max_mods, n_roles), dtype=np.float64)
        forbidden_indicators = np.zeros((n_arch, n_roles), dtype=np.float64)
        relevant_indicators = np.zeros((n_arch, n_roles), dtype=np.float64)

        for a_idx in range(n_arch):
            f_mask = int(forbidden_masks[a_idx])
            r_mask = int(all_relevant_masks[a_idx])
            n_m = int(num_modules[a_idx])
            for r_idx in range(n_roles):
                bit = 1 << r_idx
                if f_mask & bit:
                    forbidden_indicators[a_idx, r_idx] = 1.0
                if r_mask & bit:
                    relevant_indicators[a_idx, r_idx] = 1.0
                for m_idx in range(n_m):
                    if int(module_masks[a_idx, m_idx]) & bit:
                        module_indicators[a_idx, m_idx, r_idx] = 1.0

        active_module_mask = np.arange(max_mods)[None, :] < num_modules[:, None]

        alias_flat: list[bytes] = []
        alias_lens = np.empty(n_arch, dtype=np.int32)
        for i, s in enumerate(specs):
            alias_lens[i] = len(s.aliases)
            alias_flat.extend(s.aliases)
        alias_values = np.array(alias_flat, dtype="S") if alias_flat else np.empty(0, dtype="S1")
        offsets = lengths_to_offsets(alias_lens)

        for arr in (
            names,
            categories,
            descriptions,
            forbidden_masks,
            cargo_masks,
            required_masks,
            all_relevant_masks,
            module_masks,
            num_modules,
            active_module_mask,
            module_indicators,
            forbidden_indicators,
            relevant_indicators,
            alias_values,
            offsets,
            alias_lens,
        ):
            arr.setflags(write=False)

        return cls(
            names=names,
            categories=categories,
            descriptions=descriptions,
            forbidden_masks=forbidden_masks,
            cargo_masks=cargo_masks,
            required_masks=required_masks,
            all_relevant_masks=all_relevant_masks,
            module_masks=module_masks,
            num_modules=num_modules,
            active_module_mask=active_module_mask,
            module_indicators=module_indicators,
            forbidden_indicators=forbidden_indicators,
            relevant_indicators=relevant_indicators,
            alias_values=alias_values,
            offsets=offsets,
            lengths=alias_lens,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `ArchetypeSpecs` containers."""
        items: list[ArchetypeSpec] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)


class LocusArchetype(IntEnum):
    """Canonical biosynthetic locus archetypes (0..10).

    Defines the ontological classification and structural requirements of bacterial
    surface polysaccharide biosynthetic gene clusters. Scoring, completeness evaluation,
    and segment classification are encapsulated in [`ArchetypeClassifier`][seroref.loci.classifier.ArchetypeClassifier].
    """

    CPS_WZY = 0
    CPS_ABC = 1
    CPS_SYNTHASE = 2
    LPS_WZY = 3
    LPS_ABC = 4
    EPS_PEL = 5
    EPS_PGA = 6
    EPS_ALGINATE = 7
    EPS_BCS = 8
    TEICHOIC_ACID = 9
    UNKNOWN = 10

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the `ArchetypeSpec` scalar view for this archetype."""
        return self.specs()[int(self)]

    @property
    def category(self) -> ArchetypeCategory:
        """High-level `ArchetypeCategory` (`CAPSULE`, `O_ANTIGEN`, `EXOPOLYSACCHARIDE`, `OTHER`)."""
        return ArchetypeCategory(int(self.specs().categories[int(self)]))

    @property
    def forbidden_mask(self) -> int:
        """32-bit bitmask of `GeneRole` bits that are incompatible with this archetype."""
        return int(self.specs().forbidden_masks[int(self)])

    @property
    def cargo_mask(self) -> int:
        """32-bit bitmask of optional cargo/modifier/precursor `GeneRole` bits."""
        return int(self.specs().cargo_masks[int(self)])

    @property
    def required_mask(self) -> int:
        """Combined 32-bit union mask of all roles participating in any required module."""
        return int(self.specs().required_masks[int(self)])

    @property
    def all_relevant_mask(self) -> int:
        """Combined 32-bit bitmask of all required and optional cargo roles."""
        return int(self.specs().all_relevant_masks[int(self)])

    @property
    def description(self) -> bytes:
        """Human-readable biological description of the pathway."""
        return bytes(self.specs().descriptions[int(self)])

    @property
    def module_masks(self) -> tuple[int, ...]:
        """Tuple of 32-bit disjunctive module bitmasks (`M_1, ..., M_k`) required for completeness."""
        s = self.specs()
        idx = int(self)
        n_m = int(s.num_modules[idx])
        return tuple(int(s.module_masks[idx, m]) for m in range(n_m))

    @property
    def num_modules(self) -> int:
        """Number of required functional modules (`M_A`) for this archetype."""
        return int(self.specs().num_modules[int(self)])

    @property
    def aliases(self) -> tuple[bytes, ...]:
        """Known biological aliases for this archetype."""
        return self.spec.aliases

    @classmethod
    def specs(cls) -> ArchetypeSpecs:
        """Return the cached singleton `ArchetypeSpecs` SoA container over all `LocusArchetype` members."""
        return _get_archetype_specs()

    @classmethod
    def from_name(cls, key: str | bytes | int | LocusArchetype) -> LocusArchetype:
        """Resolve an archetype by integer ID, enum name, or biological alias."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        b_clean = b.strip().lower().replace(b"-", b"_").replace(b" ", b"_")
        lookup = _get_archetype_lookup()
        if b_clean in lookup:
            return lookup[b_clean]
        raise KeyError(f"Unknown LocusArchetype identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | LocusArchetype,
        default: LocusArchetype | None = None,
    ) -> LocusArchetype:
        """Resolve an archetype identifier, falling back to `default` (or `UNKNOWN`)."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback


@lru_cache(maxsize=1)
def _get_archetype_specs() -> ArchetypeSpecs:
    """Build and cache the singleton `ArchetypeSpecs` SoA container over all `LocusArchetype` members (0..10)."""
    common_precursors_and_mods = roles_to_mask(
        (
            GeneRole.GLYCOSYLTRANSFERASE,
            GeneRole.INITIATOR,
            GeneRole.PYRUVYLTRANSFERASE,
            GeneRole.METHYLTRANSFERASE,
            GeneRole.ACETYLTRANSFERASE,
            GeneRole.PHOSPHOTRANSFERASE,
            GeneRole.FORMYLTRANSFERASE,
            GeneRole.ALANINE_TRANSFERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.DEACETYLASE,
            GeneRole.EPIMERASE,
            GeneRole.OXIDOREDUCTASE,
            GeneRole.NUCLEOTIDYLTRANSFERASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.GLYCOSYLHYDROLASE,
        )
    )
    return ArchetypeSpecs.from_list(
        (
            # 0: CPS_WZY
            ArchetypeSpec(
                name=b"CPS_WZY",
                category=ArchetypeCategory.CAPSULE,
                module_masks=(
                    GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask | GeneRole.PHOSPHATASE.bitmask,
                    GeneRole.KINASE.bitmask,
                    GeneRole.FLIPPASE.bitmask,
                    GeneRole.POLYMERASE.bitmask | GeneRole.LIGASE.bitmask | GeneRole.COPOLYMERASE.bitmask,
                    GeneRole.INITIATOR.bitmask | GeneRole.GLYCOSYLTRANSFERASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods
                | roles_to_mask(
                    (
                        GeneRole.PHOSPHATASE,
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.COPOLYMERASE,
                        GeneRole.LIGASE,
                    )
                ),
                description=(
                    b"Group 1 / 4 Wzy-dependent capsular polysaccharide locus (K-locus); "
                    b"requires outer membrane export (Wza/Wzi), chain-length regulation (Wzc/Wzb/Wzz), "
                    b"flippase (Wzx), polymerase (Wzy), and initiating/repeating glycosyltransferases."
                ),
                aliases=(b"k_locus", b"klocus", b"cps_group1", b"cps_group4", b"wzy_capsule"),
            ),
            # 1: CPS_ABC
            ArchetypeSpec(
                name=b"CPS_ABC",
                category=ArchetypeCategory.CAPSULE,
                module_masks=(
                    GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask,
                    GeneRole.ABC_PERMEASE.bitmask,
                    GeneRole.ABC_ATPASE.bitmask,
                    GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods | GeneRole.CAPSULE_ANCHOR.bitmask,
                description=(
                    b"Group 2 / 3 ABC-transporter-dependent capsular polysaccharide locus (kps cluster); "
                    b"requires KpsD/KpsE translocon, KpsM permease, KpsT ATPase, and glycosyltransferases."
                ),
                aliases=(b"kps", b"cps_group2", b"cps_group3", b"abc_capsule"),
            ),
            # 2: CPS_SYNTHASE
            ArchetypeSpec(
                name=b"CPS_SYNTHASE",
                category=ArchetypeCategory.CAPSULE,
                module_masks=(GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                        GeneRole.DEACETYLASE,
                        GeneRole.GLYCOSYLHYDROLASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods
                | roles_to_mask(
                    (
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.PHOSPHATASE,
                        GeneRole.KINASE,
                        GeneRole.COPOLYMERASE,
                        GeneRole.PHOSPHOTRANSFERASE,
                    )
                ),
                description=(
                    b"Synthase-dependent capsule locus (e.g., HasA hyaluronic acid, Cps3S, or ectopic Tts synthase); "
                    b"coupled polymerization and transmembrane translocation by a single processive synthase."
                ),
                aliases=(b"synthase_capsule", b"has_locus", b"cps3", b"tts"),
            ),
            # 3: LPS_WZY
            ArchetypeSpec(
                name=b"LPS_WZY",
                category=ArchetypeCategory.O_ANTIGEN,
                module_masks=(
                    GeneRole.FLIPPASE.bitmask,
                    GeneRole.POLYMERASE.bitmask | GeneRole.LIGASE.bitmask | GeneRole.COPOLYMERASE.bitmask,
                    GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.TRANSLOCON,
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.KINASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods
                | roles_to_mask(
                    (
                        GeneRole.COPOLYMERASE,
                        GeneRole.LIGASE,
                    )
                ),
                description=(
                    b"Wzy-dependent lipopolysaccharide O-antigen locus (O-locus); "
                    b"requires Wzx flippase, Wzy polymerase, and glycosyltransferases without Wza/Wzc capsule export."
                ),
                aliases=(b"o_locus", b"olocus", b"wzy_o_antigen", b"rfb_wzy"),
            ),
            # 4: LPS_ABC
            ArchetypeSpec(
                name=b"LPS_ABC",
                category=ArchetypeCategory.O_ANTIGEN,
                module_masks=(
                    GeneRole.ABC_PERMEASE.bitmask,
                    GeneRole.ABC_ATPASE.bitmask,
                    GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.TRANSLOCON,
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.KINASE,
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                        GeneRole.ALANINE_TRANSFERASE,
                    )
                ),
                cargo_mask=(common_precursors_and_mods & ~GeneRole.ALANINE_TRANSFERASE.bitmask)
                | GeneRole.LIGASE.bitmask,
                description=(
                    b"ABC-transporter-dependent lipopolysaccharide O-antigen locus; "
                    b"requires Wzm permease, Wzt ATPase, and glycosyltransferases without outer-membrane Wza/KpsD translocon."
                ),
                aliases=(b"abc_o_antigen", b"wzm_wzt_o_antigen", b"rfb_abc"),
            ),
            # 5: EPS_PEL
            ArchetypeSpec(
                name=b"EPS_PEL",
                category=ArchetypeCategory.EXOPOLYSACCHARIDE,
                module_masks=(
                    GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                    GeneRole.DEACETYLASE.bitmask,
                    GeneRole.GLYCOSYLHYDROLASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
                description=b"Pel biofilm exopolysaccharide locus (PelA deacetylase/hydrolase, PelF GT, PelD/E/G machinery).",
                aliases=(b"pel", b"pel_locus"),
            ),
            # 6: EPS_PGA
            ArchetypeSpec(
                name=b"EPS_PGA",
                category=ArchetypeCategory.EXOPOLYSACCHARIDE,
                module_masks=(
                    GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                    GeneRole.DEACETYLASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
                description=b"PNAG / PgaABCD / IcaADBC poly-beta-1,6-GlcNAc biofilm exopolysaccharide locus.",
                aliases=(b"pga", b"ica", b"pnag"),
            ),
            # 7: EPS_ALGINATE
            ArchetypeSpec(
                name=b"EPS_ALGINATE",
                category=ArchetypeCategory.EXOPOLYSACCHARIDE,
                module_masks=(
                    GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                    GeneRole.OXIDOREDUCTASE.bitmask | GeneRole.EPIMERASE.bitmask,
                    GeneRole.GLYCOSYLHYDROLASE.bitmask,
                    GeneRole.ACETYLTRANSFERASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask,
                description=b"Alginate exopolysaccharide locus (Alg8 synthase, AlgD/AlgG dehydrogenase/epimerase, AlgL lyase, AlgI/J/F acetylation).",
                aliases=(b"alg", b"alginate"),
            ),
            # 8: EPS_BCS
            ArchetypeSpec(
                name=b"EPS_BCS",
                category=ArchetypeCategory.EXOPOLYSACCHARIDE,
                module_masks=(
                    GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                    GeneRole.GLYCOSYLHYDROLASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.FLIPPASE,
                        GeneRole.POLYMERASE,
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                        GeneRole.DEACETYLASE,
                        GeneRole.OXIDOREDUCTASE,
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.PHOSPHATASE,
                        GeneRole.COPOLYMERASE,
                        GeneRole.KINASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
                description=b"Bacterial cellulose synthesis (BcsABZC) exopolysaccharide locus.",
                aliases=(b"bcs", b"cellulose"),
            ),
            # 9: TEICHOIC_ACID
            ArchetypeSpec(
                name=b"TEICHOIC_ACID",
                category=ArchetypeCategory.OTHER,
                module_masks=(
                    GeneRole.ABC_PERMEASE.bitmask | GeneRole.FLIPPASE.bitmask,
                    GeneRole.ABC_ATPASE.bitmask | GeneRole.PHOSPHOTRANSFERASE.bitmask,
                    GeneRole.PHOSPHOTRANSFERASE.bitmask | GeneRole.ALANINE_TRANSFERASE.bitmask,
                    GeneRole.INITIATOR.bitmask
                    | GeneRole.GLYCOSYLTRANSFERASE.bitmask
                    | GeneRole.NUCLEOTIDYLTRANSFERASE.bitmask
                    | GeneRole.OXIDOREDUCTASE.bitmask,
                ),
                forbidden_mask=roles_to_mask(
                    (
                        GeneRole.TRANSLOCON,
                        GeneRole.CAPSULE_ANCHOR,
                        GeneRole.TRANSMEMBRANE_SYNTHASE,
                        GeneRole.POLYMERASE,
                        GeneRole.COPOLYMERASE,
                        GeneRole.KINASE,
                        GeneRole.LIGASE,
                    )
                ),
                cargo_mask=common_precursors_and_mods
                | roles_to_mask(
                    (
                        GeneRole.ABC_PERMEASE,
                        GeneRole.ABC_ATPASE,
                        GeneRole.FLIPPASE,
                    )
                ),
                description=b"Wall teichoic acid (WTA / tag / tar / dlt) or phosphocholine-decorated teichoic acid "
                b"(lic / tarIJL) locus.",
                aliases=(b"wta", b"teichoic", b"tag_tar", b"lic"),
            ),
            # 10: UNKNOWN
            ArchetypeSpec(
                name=b"UNKNOWN",
                category=ArchetypeCategory.OTHER,
                module_masks=(),
                forbidden_mask=0,
                cargo_mask=common_precursors_and_mods,
                description=b"Unclassified, incomplete, or orphan biosynthetic gene cluster.",
                aliases=(b"unclassified", b"orphan", b"other"),
            ),
        )
    )


@lru_cache(maxsize=1)
def _get_archetype_lookup() -> dict[bytes, LocusArchetype]:
    """Build and cache the name/alias lookup dictionary for `LocusArchetype`."""
    specs = _get_archetype_specs()
    lookup: dict[bytes, LocusArchetype] = {}
    for arch in LocusArchetype:
        spec = specs[int(arch)]
        lookup[spec.name.lower()] = arch
        for alias in spec.aliases:
            lookup[alias.strip().lower()] = arch
    return lookup
