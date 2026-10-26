"""Data-oriented bacterial species ontology, serotype grammars, and SoA `SpeciesSpecs` container."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Self, overload

import numpy as np
from seroverse.core.collections import (
    RaggedArrayContainer,
    ScalarView,
    _gather_ragged_kernel,
    lengths_to_offsets,
)

from seroref.loci.archetypes import ArchetypeCategory, LocusArchetype


def archetypes_to_mask(archetypes: Iterable[int | LocusArchetype]) -> int:
    """Pack a collection of `LocusArchetype` values into a 32-bit integer presence bitmask."""
    mask = 0
    for arch in archetypes:
        val = int(arch)
        if 0 <= val < len(LocusArchetype):
            mask |= 1 << val
    return mask


def mask_to_archetypes(mask: int) -> tuple[LocusArchetype, ...]:
    """Unpack a 32-bit integer presence bitmask into a sorted tuple of `LocusArchetype` enums."""
    return tuple(arch for arch in LocusArchetype if (mask & (1 << int(arch))) != 0)


class GramStain(IntEnum):
    """Cell-envelope Gram stain classification (`0..3`) for bacterial species."""

    NEGATIVE = 0
    POSITIVE = 1
    VARIABLE = 2
    UNKNOWN = 3

    @property
    def label(self) -> bytes:
        """Canonical ASCII byte-string label for this Gram stain class."""
        return _GRAM_STAIN_LABELS[int(self)]

    @property
    def label_str(self) -> str:
        """Canonical UTF-8 string label for this Gram stain class."""
        return self.label.decode("ascii")

    @classmethod
    def from_name(cls, key: str | bytes | int | GramStain) -> GramStain:
        """Resolve a `GramStain` by enum ID, enum name, or label."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        norm = b.strip().lower()
        for member in cls:
            if norm in (member.name.encode("ascii").lower(), member.label.lower()):
                return member
        raise KeyError(f"Unknown GramStain identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | GramStain,
        default: GramStain | None = None,
    ) -> GramStain:
        """Resolve a `GramStain` identifier, falling back to `default` (or `GramStain.UNKNOWN`)."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback


_GRAM_STAIN_LABELS: tuple[bytes, ...] = (
    b"Negative",
    b"Positive",
    b"Variable",
    b"Unknown",
)


@dataclass(frozen=True, slots=True)
class SerotypeRule(ScalarView):
    """Declarative rule mapping a serotype token prefix to its `ArchetypeCategory` and canonical prefix."""

    prefix: bytes
    category: ArchetypeCategory
    canonical_prefix: bytes = b""


_OK_TOKEN_RE = re.compile(rb"\b(O(?:CL|L)?\d+[a-zA-Z0-9_/-]*|K(?:L)?\d+[a-zA-Z0-9_/-]*|HS:\d+[a-zA-Z0-9_/-]*)\b", re.I)
_TYPED_NUM_RE = re.compile(
    rb"(?:serotype|serogroup|serovar|type|capsule|b-band|osa|iats|penner\s+serovar)\s+(?:O:?)?([0-9]{1,3}[A-Za-z]?)\b",
    re.I,
)
_LEADING_ZERO_O_RE = re.compile(rb"^O:?(?:0+)([1-9]\d*[A-Za-z]?)$", re.I)
_COLON_O_RE = re.compile(rb"^O:(\d+[A-Za-z]?)$", re.I)
_FLAGELLAR_ONLY_RE = re.compile(rb"^H\s*[-:]?\s*(?:\d+[a-zA-Z0-9_-]*|-|nt|nm)$", re.I)
_BARE_NUMBER_SERO_RE = re.compile(rb"^[0-9]{1,3}[A-Za-z]?(?:/[0-9]{1,2}[A-Za-z]?)?$")
_MUTANT_OR_LAB_STRAIN_RE = re.compile(
    rb"(?:\xe2\x88\x86|\xce\x94|::|\b(?:DH5(?:\xce\xb1|a)?|MG1655|BL21|W3110|BW25113|K-?12|ATCC|NCTC|DSM|CNCTC|PCM|CCUG|CIP|JCM|LMG|CECT|PrK|Prk)\b)",
    re.I,
)
_BAD_PLACEHOLDER_TOKENS: frozenset[bytes] = frozenset(
    {
        b"",
        b"na",
        b"n/a",
        b"none",
        b"unknown",
        b"-",
        b"nd",
        b"nt",
        b"?",
        b"not known",
        b"not available",
        b"untyped",
        b"untypeable",
        b"non-typeable",
        b"nontypeable",
        b"rough",
        b"autoagglutinating",
        b"missing",
        b"not collected",
        b"not applicable",
    }
)


@dataclass(frozen=True, slots=True)
class SpeciesSpec(ScalarView):
    """Immutable biological specification for a canonical `BacterialSpecies`.

    Encapsulates taxonomic identity, cell-envelope architecture, expected `LocusArchetype` bitmask,
    declarative serotype formula rules, serovar-to-O-group mappings, and landmark BGC accession series.
    """

    name: bytes
    ncbi_taxid: int
    gram_stain: GramStain
    default_category: ArchetypeCategory
    archetype_mask: int
    serotype_rules: tuple[SerotypeRule, ...] = ()
    serovar_map: tuple[tuple[bytes, bytes], ...] = ()
    aliases: tuple[bytes, ...] = ()
    landmark_accessions: tuple[tuple[bytes, bytes], ...] = ()
    known_serotypes: tuple[bytes, ...] = ()

    @property
    def scientific_name(self) -> bytes:
        """Canonical scientific name as ASCII bytes (alias for `self.name`)."""
        return self.name

    @property
    def synonyms(self) -> tuple[bytes, ...]:
        """Taxonomic synonyms and genus aliases as ASCII bytes (alias for `self.aliases`)."""
        return self.aliases

    @property
    def name_str(self) -> str:
        """UTF-8 scientific name of this bacterial species."""
        return self.name.decode("utf-8", errors="replace")

    @property
    def scientific_name_str(self) -> str:
        """UTF-8 scientific name of this bacterial species."""
        return self.name_str

    @property
    def genus(self) -> bytes:
        """Genus byte string extracted from `self.name`."""
        return self.name.split(b" ", 1)[0]

    @property
    def archetypes(self) -> tuple[LocusArchetype, ...]:
        """Sorted tuple of `LocusArchetype`s compatible with this species."""
        return mask_to_archetypes(self.archetype_mask)

    def supports_archetype(self, archetype: LocusArchetype) -> bool:
        """Return `True` if `archetype` is enabled in `self.archetype_mask`."""
        return bool(self.archetype_mask & (1 << int(archetype)))

    def _serovar_lookup(self) -> dict[bytes, bytes]:
        return {k.lower(): v for k, v in self.serovar_map}

    def is_strain_or_mutant_token(self, token: str | bytes) -> bool:
        """Return `True` if `token` is a laboratory strain, culture-collection ID, or genetic mutant."""
        b = token.encode("utf-8").strip() if isinstance(token, str) else bytes(token).strip()
        if not b:
            return False
        if _MUTANT_OR_LAB_STRAIN_RE.search(b):
            return True
        return b.lower() in self._serovar_lookup()

    def canonicalize_token(
        self,
        raw_token: str | bytes,
        *,
        archetype: LocusArchetype | None = None,
    ) -> tuple[ArchetypeCategory, bytes] | None:
        """Normalize a single serotype token into `(ArchetypeCategory, canonical_bytes)` or `None`."""
        b = raw_token.encode("utf-8").strip() if isinstance(raw_token, str) else bytes(raw_token).strip()
        low = b.lower()
        if not b or low in _BAD_PLACEHOLDER_TOKENS or _FLAGELLAR_ONLY_RE.match(b):
            return None

        serovars = self._serovar_lookup()
        if low in serovars:
            return (ArchetypeCategory.O_ANTIGEN, serovars[low])

        if _MUTANT_OR_LAB_STRAIN_RE.search(b):
            return None

        if m_num := _TYPED_NUM_RE.search(b):
            num_tok = m_num.group(1)
            cat = (
                archetype.category
                if (archetype is not None and archetype != LocusArchetype.UNKNOWN)
                else self.default_category
            )
            return (cat, self._format_bare_token(num_tok, cat))

        if m_zero := _LEADING_ZERO_O_RE.match(b):
            return (ArchetypeCategory.O_ANTIGEN, b"O%s" % m_zero.group(1))
        if m_col := _COLON_O_RE.match(b):
            return (ArchetypeCategory.O_ANTIGEN, b"O%s" % m_col.group(1))

        up = b.upper()
        for rule in self.serotype_rules:
            pref_up = rule.prefix.upper()
            if up.startswith(pref_up) and len(b) > len(pref_up):
                rest = b[len(pref_up) :].lstrip(b":- ")
                if rest and (rest[:1].isdigit() or len(rest) <= 4):
                    canon = b"%s%s" % (rule.canonical_prefix, rest) if rule.canonical_prefix else b
                    return (rule.category, canon)

        target_cat = (
            archetype.category
            if (archetype is not None and archetype != LocusArchetype.UNKNOWN)
            else self.default_category
        )
        if _BARE_NUMBER_SERO_RE.match(b):
            return (target_cat, self._format_bare_token(b, target_cat))

        return (target_cat, b)

    def _format_bare_token(self, bare: bytes, category: ArchetypeCategory) -> bytes:
        for rule in self.serotype_rules:
            if rule.category == category and rule.canonical_prefix:
                if not bare.upper().startswith(rule.canonical_prefix.upper()):
                    return b"%s%s" % (rule.canonical_prefix, bare)
                return bare
        return bare

    def parse_serotype(
        self,
        raw: str | bytes,
        *,
        archetype: LocusArchetype | None = None,
    ) -> tuple[tuple[ArchetypeCategory, bytes], ...]:
        """Parse a single or multi-antigen serotype formula (e.g. `b"O1:K1(L1):H7"`) into typed tokens."""
        b = raw.encode("utf-8").strip() if isinstance(raw, str) else bytes(raw).strip()
        if not b or b.lower() in _BAD_PLACEHOLDER_TOKENS:
            return ()

        cleaned = re.sub(rb"\([Ll]\d*\)", b"", b).strip()
        ok_hits = _OK_TOKEN_RE.findall(cleaned)
        if len(ok_hits) >= 2 or (len(ok_hits) == 1 and b":" in cleaned):
            results: list[tuple[ArchetypeCategory, bytes]] = []
            seen: set[tuple[ArchetypeCategory, bytes]] = set()
            for tok in ok_hits:
                parsed = self.canonicalize_token(tok, archetype=archetype)
                if parsed is not None and parsed not in seen:
                    seen.add(parsed)
                    results.append(parsed)
            if results:
                return tuple(results)

        single = self.canonicalize_token(cleaned, archetype=archetype)
        return (single,) if single is not None else ()

    def normalize_strain(
        self,
        raw_strain: str | bytes,
        raw_serotype: str | bytes = b"",
    ) -> bytes:
        """Normalize a strain isolate designation, rescuing strain/serovar tokens from `raw_serotype` when needed."""
        s_b = raw_strain.encode("utf-8").strip() if isinstance(raw_strain, str) else bytes(raw_strain).strip()
        sero_b = raw_serotype.encode("utf-8").strip() if isinstance(raw_serotype, str) else bytes(raw_serotype).strip()

        # Rescue mutant/lab strain or Salmonella serovar mistakenly placed in raw_serotype
        if not s_b and sero_b and self.is_strain_or_mutant_token(sero_b):
            s_b = sero_b

        if not s_b or s_b.lower() in _BAD_PLACEHOLDER_TOKENS or _FLAGELLAR_ONLY_RE.match(s_b):
            return b""

        # Reject if strain is identical to the canonicalized serotype token or raw serotype formula
        if sero_b and s_b.lower() == sero_b.lower() and not self.is_strain_or_mutant_token(s_b):
            return b""
        for _, canon_b in self.parse_serotype(sero_b):
            if s_b.lower() == canon_b.lower():
                return b""

        # Reject bare O/K antigen formulas masquerading as strain names unless accompanied by an isolate token
        if not self.is_strain_or_mutant_token(s_b):
            ok_tokens = _OK_TOKEN_RE.findall(s_b)
            if (
                ok_tokens
                and b":".join(ok_tokens).lower() == re.sub(rb":H\d+", b"", s_b, flags=re.I).strip(b":").lower()
            ):
                return b""
            if s_b in self.known_serotypes:
                return b""

        return s_b


def _gather_ragged_secondary(
    values: np.ndarray[tuple[int], Any],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    row_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[
    np.ndarray[tuple[int], Any],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    """Gather a secondary ragged array using `seroverse.core.collections._gather_ragged_kernel`."""
    if len(row_indices) == 0:
        empty_i32 = np.empty(0, dtype=np.int32)
        return np.empty(0, dtype=values.dtype), empty_i32, empty_i32
    new_lengths = lengths[row_indices].astype(np.int32, copy=False)
    new_offsets = lengths_to_offsets(new_lengths)
    tot = int(new_offsets[-1] + new_lengths[-1])
    if tot == 0:
        return np.empty(0, dtype=values.dtype), new_offsets, new_lengths
    if values.dtype.kind == "S":
        idx_pool = np.arange(len(values), dtype=np.int32)
        gathered_idx = _gather_ragged_kernel(idx_pool, offsets, new_offsets, new_lengths, row_indices, tot)
        return values[gathered_idx], new_offsets, new_lengths
    gathered = _gather_ragged_kernel(values, offsets, new_offsets, new_lengths, row_indices, tot)
    return gathered, new_offsets, new_lengths


@dataclass(frozen=True, slots=True)
class SpeciesSpecs(RaggedArrayContainer[SpeciesSpec, "SpeciesSpecs"]):
    """Ragged Structure-of-Arrays (SoA) batch container over `SpeciesSpec` specifications.

    All 1D fields share leading dimension `len(self)` (`52` for the canonical `BacterialSpecies` ontology).
    Variable-length per-species metadata (`aliases`, `serotype_rules`, `serovar_map`, `landmark_accessions`,
    and `known_serotypes`) are CSR-packed into contiguous NumPy arrays without Python object tuples.
    """

    names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    ncbi_taxids: np.ndarray[tuple[int], np.dtype[np.int32]]
    gram_stains: np.ndarray[tuple[int], np.dtype[np.int8]]
    default_categories: np.ndarray[tuple[int], np.dtype[np.int8]]
    archetype_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    alias_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    rule_prefixes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    rule_categories: np.ndarray[tuple[int], np.dtype[np.int8]]
    rule_canonical_prefixes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    rule_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    rule_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    serovar_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    serovar_targets: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    serovar_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    serovar_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    landmark_starts: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    landmark_ends: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    landmark_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    landmark_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    known_serotype_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    known_serotype_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    known_serotype_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[SpeciesSpec]:
        for i in range(len(self)):
            yield self[i]

    @property
    def scientific_names(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """1D byte-string array of canonical scientific names (alias for `self.names`)."""
        return self.names

    @property
    def alias_offsets(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Alias for `self.offsets` (starting indices into `self.alias_values`)."""
        return self.offsets

    @property
    def alias_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Alias for `self.lengths` (number of taxonomic aliases per species)."""
        return self.lengths

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> SpeciesSpec: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> SpeciesSpec | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> SpeciesSpec | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"SpeciesSpecs index {item} out of range for length {n}.")

            a_off, a_ln = int(self.offsets[idx]), int(self.lengths[idx])
            aliases = tuple(bytes(self.alias_values[k]) for k in range(a_off, a_off + a_ln))

            r_off, r_ln = int(self.rule_offsets[idx]), int(self.rule_lengths[idx])
            rules = tuple(
                SerotypeRule(
                    prefix=bytes(self.rule_prefixes[k]),
                    category=ArchetypeCategory(int(self.rule_categories[k])),
                    canonical_prefix=bytes(self.rule_canonical_prefixes[k]),
                )
                for k in range(r_off, r_off + r_ln)
            )

            s_off, s_ln = int(self.serovar_offsets[idx]), int(self.serovar_lengths[idx])
            serovars = tuple(
                (bytes(self.serovar_keys[k]), bytes(self.serovar_targets[k])) for k in range(s_off, s_off + s_ln)
            )

            l_off, l_ln = int(self.landmark_offsets[idx]), int(self.landmark_lengths[idx])
            landmarks = tuple(
                (bytes(self.landmark_starts[k]), bytes(self.landmark_ends[k])) for k in range(l_off, l_off + l_ln)
            )

            k_off, k_ln = int(self.known_serotype_offsets[idx]), int(self.known_serotype_lengths[idx])
            known_seros = tuple(bytes(self.known_serotype_values[k]) for k in range(k_off, k_off + k_ln))

            return SpeciesSpec(
                name=bytes(self.names[idx]),
                ncbi_taxid=int(self.ncbi_taxids[idx]),
                gram_stain=GramStain(int(self.gram_stains[idx])),
                default_category=ArchetypeCategory(int(self.default_categories[idx])),
                archetype_mask=int(self.archetype_masks[idx]),
                serotype_rules=rules,
                serovar_map=serovars,
                aliases=aliases,
                landmark_accessions=landmarks,
                known_serotypes=known_seros,
            )

        alias_idx_pool = np.arange(len(self.alias_values), dtype=np.int32)
        indices, new_offsets, new_lengths, (gathered_alias_idx,) = self._slice_ragged(item, alias_idx_pool)
        if len(indices) == 0:
            return self.empty()

        alias_v = self.alias_values[gathered_alias_idx] if len(gathered_alias_idx) > 0 else np.empty(0, dtype="S1")

        rule_idx_pool = np.arange(len(self.rule_prefixes), dtype=np.int32)
        rule_idx, r_off, r_ln = _gather_ragged_secondary(
            rule_idx_pool,
            self.rule_offsets,
            self.rule_lengths,
            indices,
        )
        r_pref = self.rule_prefixes[rule_idx] if len(rule_idx) > 0 else np.empty(0, dtype="S1")
        r_cats = self.rule_categories[rule_idx] if len(rule_idx) > 0 else np.empty(0, dtype=np.int8)
        r_canon = self.rule_canonical_prefixes[rule_idx] if len(rule_idx) > 0 else np.empty(0, dtype="S1")

        sero_idx_pool = np.arange(len(self.serovar_keys), dtype=np.int32)
        sero_idx, s_off, s_ln = _gather_ragged_secondary(
            sero_idx_pool,
            self.serovar_offsets,
            self.serovar_lengths,
            indices,
        )
        s_keys = self.serovar_keys[sero_idx] if len(sero_idx) > 0 else np.empty(0, dtype="S1")
        s_targs = self.serovar_targets[sero_idx] if len(sero_idx) > 0 else np.empty(0, dtype="S1")

        lm_idx_pool = np.arange(len(self.landmark_starts), dtype=np.int32)
        lm_idx, l_off, l_ln = _gather_ragged_secondary(
            lm_idx_pool,
            self.landmark_offsets,
            self.landmark_lengths,
            indices,
        )
        l_starts = self.landmark_starts[lm_idx] if len(lm_idx) > 0 else np.empty(0, dtype="S1")
        l_ends = self.landmark_ends[lm_idx] if len(lm_idx) > 0 else np.empty(0, dtype="S1")

        ks_v, ks_off, ks_ln = _gather_ragged_secondary(
            self.known_serotype_values,
            self.known_serotype_offsets,
            self.known_serotype_lengths,
            indices,
        )

        return type(self)(
            names=self.names[indices],
            ncbi_taxids=self.ncbi_taxids[indices],
            gram_stains=self.gram_stains[indices],
            default_categories=self.default_categories[indices],
            archetype_masks=self.archetype_masks[indices],
            alias_values=alias_v,
            offsets=new_offsets,
            lengths=new_lengths,
            rule_prefixes=r_pref,
            rule_categories=r_cats,
            rule_canonical_prefixes=r_canon,
            rule_offsets=r_off,
            rule_lengths=r_ln,
            serovar_keys=s_keys,
            serovar_targets=s_targs,
            serovar_offsets=s_off,
            serovar_lengths=s_ln,
            landmark_starts=l_starts,
            landmark_ends=l_ends,
            landmark_offsets=l_off,
            landmark_lengths=l_ln,
            known_serotype_values=ks_v,
            known_serotype_offsets=ks_off,
            known_serotype_lengths=ks_ln,
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `SpeciesSpecs` container."""
        return cls(
            names=np.empty(0, dtype="S1"),
            ncbi_taxids=np.empty(0, dtype=np.int32),
            gram_stains=np.empty(0, dtype=np.int8),
            default_categories=np.empty(0, dtype=np.int8),
            archetype_masks=np.empty(0, dtype=np.uint32),
            alias_values=np.empty(0, dtype="S1"),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
            rule_prefixes=np.empty(0, dtype="S1"),
            rule_categories=np.empty(0, dtype=np.int8),
            rule_canonical_prefixes=np.empty(0, dtype="S1"),
            rule_offsets=np.empty(0, dtype=np.int32),
            rule_lengths=np.empty(0, dtype=np.int32),
            serovar_keys=np.empty(0, dtype="S1"),
            serovar_targets=np.empty(0, dtype="S1"),
            serovar_offsets=np.empty(0, dtype=np.int32),
            serovar_lengths=np.empty(0, dtype=np.int32),
            landmark_starts=np.empty(0, dtype="S1"),
            landmark_ends=np.empty(0, dtype="S1"),
            landmark_offsets=np.empty(0, dtype=np.int32),
            landmark_lengths=np.empty(0, dtype=np.int32),
            known_serotype_values=np.empty(0, dtype="S1"),
            known_serotype_offsets=np.empty(0, dtype=np.int32),
            known_serotype_lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_list(cls, specs: Sequence[SpeciesSpec]) -> Self:
        """Compile a sequence of `SpeciesSpec` scalar views into a `SpeciesSpecs` SoA container."""
        n = len(specs)
        if n == 0:
            return cls.empty()

        names = np.array([s.name for s in specs], dtype="S64")
        taxids = np.array([s.ncbi_taxid for s in specs], dtype=np.int32)
        grams = np.array([int(s.gram_stain) for s in specs], dtype=np.int8)
        cats = np.array([int(s.default_category) for s in specs], dtype=np.int8)
        masks = np.array([s.archetype_mask for s in specs], dtype=np.uint32)

        alias_flat: list[bytes] = []
        alias_lens = np.empty(n, dtype=np.int32)
        r_pref_flat: list[bytes] = []
        r_cat_flat: list[int] = []
        r_canon_flat: list[bytes] = []
        r_lens = np.empty(n, dtype=np.int32)
        s_key_flat: list[bytes] = []
        s_targ_flat: list[bytes] = []
        s_lens = np.empty(n, dtype=np.int32)
        l_start_flat: list[bytes] = []
        l_end_flat: list[bytes] = []
        l_lens = np.empty(n, dtype=np.int32)
        ks_flat: list[bytes] = []
        ks_lens = np.empty(n, dtype=np.int32)

        for i, s in enumerate(specs):
            alias_lens[i] = len(s.aliases)
            alias_flat.extend(s.aliases)

            r_lens[i] = len(s.serotype_rules)
            for rule in s.serotype_rules:
                r_pref_flat.append(rule.prefix)
                r_cat_flat.append(int(rule.category))
                r_canon_flat.append(rule.canonical_prefix)

            s_lens[i] = len(s.serovar_map)
            for k, v in s.serovar_map:
                s_key_flat.append(k)
                s_targ_flat.append(v)

            l_lens[i] = len(s.landmark_accessions)
            for st, en in s.landmark_accessions:
                l_start_flat.append(st)
                l_end_flat.append(en)

            ks_lens[i] = len(s.known_serotypes)
            ks_flat.extend(s.known_serotypes)

        alias_values = np.array(alias_flat, dtype="S") if alias_flat else np.empty(0, dtype="S1")
        offsets = lengths_to_offsets(alias_lens)

        rule_prefixes = np.array(r_pref_flat, dtype="S") if r_pref_flat else np.empty(0, dtype="S1")
        rule_categories = np.array(r_cat_flat, dtype=np.int8) if r_cat_flat else np.empty(0, dtype=np.int8)
        rule_canonical_prefixes = np.array(r_canon_flat, dtype="S") if r_canon_flat else np.empty(0, dtype="S1")
        rule_offsets = lengths_to_offsets(r_lens)

        serovar_keys = np.array(s_key_flat, dtype="S") if s_key_flat else np.empty(0, dtype="S1")
        serovar_targets = np.array(s_targ_flat, dtype="S") if s_targ_flat else np.empty(0, dtype="S1")
        serovar_offsets = lengths_to_offsets(s_lens)

        landmark_starts = np.array(l_start_flat, dtype="S") if l_start_flat else np.empty(0, dtype="S1")
        landmark_ends = np.array(l_end_flat, dtype="S") if l_end_flat else np.empty(0, dtype="S1")
        landmark_offsets = lengths_to_offsets(l_lens)

        known_serotype_values = np.array(ks_flat, dtype="S") if ks_flat else np.empty(0, dtype="S1")
        known_serotype_offsets = lengths_to_offsets(ks_lens)

        for arr in (
            names,
            taxids,
            grams,
            cats,
            masks,
            alias_values,
            offsets,
            alias_lens,
            rule_prefixes,
            rule_categories,
            rule_canonical_prefixes,
            rule_offsets,
            r_lens,
            serovar_keys,
            serovar_targets,
            serovar_offsets,
            s_lens,
            landmark_starts,
            landmark_ends,
            landmark_offsets,
            l_lens,
            known_serotype_values,
            known_serotype_offsets,
            ks_lens,
        ):
            arr.setflags(write=False)

        return cls(
            names=names,
            ncbi_taxids=taxids,
            gram_stains=grams,
            default_categories=cats,
            archetype_masks=masks,
            alias_values=alias_values,
            offsets=offsets,
            lengths=alias_lens,
            rule_prefixes=rule_prefixes,
            rule_categories=rule_categories,
            rule_canonical_prefixes=rule_canonical_prefixes,
            rule_offsets=rule_offsets,
            rule_lengths=r_lens,
            serovar_keys=serovar_keys,
            serovar_targets=serovar_targets,
            serovar_offsets=serovar_offsets,
            serovar_lengths=s_lens,
            landmark_starts=landmark_starts,
            landmark_ends=landmark_ends,
            landmark_offsets=landmark_offsets,
            landmark_lengths=l_lens,
            known_serotype_values=known_serotype_values,
            known_serotype_offsets=known_serotype_offsets,
            known_serotype_lengths=ks_lens,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `SpeciesSpecs` containers."""
        items: list[SpeciesSpec] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)


_COMMON_GRAM_NEG_ARCHETYPES = archetypes_to_mask(
    (
        LocusArchetype.CPS_WZY,
        LocusArchetype.CPS_ABC,
        LocusArchetype.CPS_SYNTHASE,
        LocusArchetype.LPS_WZY,
        LocusArchetype.LPS_ABC,
        LocusArchetype.EPS_PEL,
        LocusArchetype.EPS_PGA,
        LocusArchetype.EPS_ALGINATE,
        LocusArchetype.EPS_BCS,
    )
)
_COMMON_GRAM_POS_ARCHETYPES = archetypes_to_mask(
    (
        LocusArchetype.CPS_WZY,
        LocusArchetype.CPS_SYNTHASE,
        LocusArchetype.EPS_PGA,
        LocusArchetype.TEICHOIC_ACID,
    )
)

_ECOLI_RULES = (
    SerotypeRule(b"O", ArchetypeCategory.O_ANTIGEN, b"O"),
    SerotypeRule(b"K", ArchetypeCategory.CAPSULE, b"K"),
)
_KLEBSIELLA_RULES = (
    SerotypeRule(b"KL", ArchetypeCategory.CAPSULE, b"K"),
    SerotypeRule(b"K", ArchetypeCategory.CAPSULE, b"K"),
    SerotypeRule(b"OL", ArchetypeCategory.O_ANTIGEN, b"O"),
    SerotypeRule(b"O", ArchetypeCategory.O_ANTIGEN, b"O"),
)
_ACINETOBACTER_RULES = (
    SerotypeRule(b"KL", ArchetypeCategory.CAPSULE, b"KL"),
    SerotypeRule(b"K", ArchetypeCategory.CAPSULE, b"K"),
    SerotypeRule(b"OCL", ArchetypeCategory.O_ANTIGEN, b"OCL"),
    SerotypeRule(b"OC", ArchetypeCategory.O_ANTIGEN, b"OC"),
    SerotypeRule(b"O", ArchetypeCategory.O_ANTIGEN, b"O"),
)
_O_ONLY_RULES = (SerotypeRule(b"O", ArchetypeCategory.O_ANTIGEN, b"O"),)
_CAMPY_RULES = (
    SerotypeRule(b"HS:", ArchetypeCategory.CAPSULE, b"HS:"),
    SerotypeRule(b"HS", ArchetypeCategory.CAPSULE, b"HS:"),
    SerotypeRule(b"O", ArchetypeCategory.CAPSULE, b"HS:"),
)
_STAPH_RULES = (
    SerotypeRule(b"CP", ArchetypeCategory.CAPSULE, b"CP"),
    SerotypeRule(b"type ", ArchetypeCategory.CAPSULE, b"CP"),
)
_CAPSULE_UNPREFIXED_RULES = (SerotypeRule(b"type ", ArchetypeCategory.CAPSULE, b""),)

_SALMONELLA_SEROVARS: tuple[tuple[bytes, bytes], ...] = (
    (b"paratyphi a", b"O2"),
    (b"paratyphi b", b"O4"),
    (b"typhimurium", b"O4"),
    (b"heidelberg", b"O4"),
    (b"saintpaul", b"O4"),
    (b"derby", b"O4"),
    (b"agona", b"O4"),
    (b"choleraesuis", b"O7"),
    (b"paratyphi c", b"O7"),
    (b"montevideo", b"O7"),
    (b"infantis", b"O7"),
    (b"thompson", b"O7"),
    (b"newport", b"O8"),
    (b"muenchen", b"O8"),
    (b"kentucky", b"O8"),
    (b"hadar", b"O8"),
    (b"typhi", b"O9"),
    (b"enteritidis", b"O9"),
    (b"dublin", b"O9"),
    (b"gallinarum", b"O9"),
    (b"pullorum", b"O9"),
    (b"anatum", b"O3,10"),
    (b"give", b"O3,10"),
    (b"london", b"O3,10"),
    (b"senftenberg", b"O1,3,19"),
    (b"aberdeen", b"O11"),
    (b"poona", b"O13"),
    (b"cerro", b"O18"),
    (b"minnesota", b"O21"),
    (b"urbana", b"O30"),
    (b"adelaide", b"O35"),
)


class BacterialSpecies(IntEnum):
    """Canonical bacterial species ontology (`0..51`) for `SeroRef` reference loci and strains.

    Follows the same data-oriented `IntEnum` + frozen `SpeciesSpec` + `SpeciesSpecs` pattern as
    [`LocusArchetype`][seroref.loci.archetypes.LocusArchetype] and [`GeneRole`][seroref.genes.roles.GeneRole].
    """

    ACINETOBACTER_BAUMANNII = 0
    ACINETOBACTER_CALCOACETICUS = 1
    ACTINOBACILLUS_PLEUROPNEUMONIAE = 2
    AEROMONAS_HYDROPHILA = 3
    AEROMONAS_SALMONICIDA = 4
    AGGREGATIBACTER_ACTINOMYCETEMCOMITANS = 5
    BACILLUS_CEREUS = 6
    BACTEROIDES_FRAGILIS = 7
    BURKHOLDERIA_CEPACIA = 8
    BURKHOLDERIA_MULTIVORANS = 9
    CAMPYLOBACTER_JEJUNI = 10
    CITROBACTER_BRAAKII = 11
    CITROBACTER_FREUNDII = 12
    CITROBACTER_WERKMANII = 13
    CRONOBACTER_DUBLINENSIS = 14
    CRONOBACTER_MALONATICUS = 15
    CRONOBACTER_SAKAZAKII = 16
    CRONOBACTER_TURICENSIS = 17
    ENTEROBACTER_CLOACAE = 18
    ENTEROCOCCUS_FAECALIS = 19
    ESCHERICHIA_ALBERTII = 20
    ESCHERICHIA_COLI = 21
    HAEMOPHILUS_INFLUENZAE = 22
    HELICOBACTER_PYLORI = 23
    KLEBSIELLA_PNEUMONIAE = 24
    LACTOBACILLUS_DELBRUECKII = 25
    NEISSERIA_MENINGITIDIS = 26
    PLESIOMONAS_SHIGELLOIDES = 27
    PROTEUS_MIRABILIS = 28
    PROTEUS_MYXOFACIENS = 29
    PROTEUS_PENNERI = 30
    PROTEUS_VULGARIS = 31
    PROVIDENCIA_RUSTIGIANII = 32
    PSEUDOMONAS_AERUGINOSA = 33
    RAOULTELLA_TERRIGENA = 34
    SALMONELLA_ENTERICA = 35
    SHIGELLA_BOYDII = 36
    SHIGELLA_DYSENTERIAE = 37
    SHIGELLA_FLEXNERI = 38
    SHIGELLA_SONNEI = 39
    STAPHYLOCOCCUS_AUREUS = 40
    STREPTOCOCCUS_ORALIS = 41
    STREPTOCOCCUS_PNEUMONIAE = 42
    STREPTOCOCCUS_SUIS = 43
    VIBRIO_CHOLERAE = 44
    YERSINIA_BERCOVIERI = 45
    YERSINIA_ENTEROCOLITICA = 46
    YERSINIA_FREDERIKSENII = 47
    YERSINIA_KRISTENSENII = 48
    YERSINIA_PSEUDOTUBERCULOSIS = 49
    YERSINIA_SIMILIS = 50
    UNKNOWN = 51

    @property
    def spec(self) -> SpeciesSpec:
        """Return the immutable `SpeciesSpec` metadata for this species."""
        return _get_species_specs()[int(self)]

    @property
    def name_bytes(self) -> bytes:
        """Canonical scientific name as ASCII bytes (e.g. `b"Escherichia coli"`)."""
        return self.spec.name

    @property
    def scientific_name(self) -> bytes:
        """Canonical scientific name as ASCII bytes (e.g. `b"Escherichia coli"`)."""
        return self.spec.name

    @property
    def name_str(self) -> str:
        """Canonical scientific name as UTF-8 `str`."""
        return self.spec.name_str

    @property
    def scientific_name_str(self) -> str:
        """Canonical scientific name as UTF-8 `str`."""
        return self.spec.name_str

    @property
    def ncbi_taxid(self) -> int:
        """Species-level NCBI Taxonomy ID."""
        return self.spec.ncbi_taxid

    @property
    def gram_stain(self) -> GramStain:
        """Gram stain classification (`GramStain.NEGATIVE` or `GramStain.POSITIVE`)."""
        return self.spec.gram_stain

    @property
    def default_category(self) -> ArchetypeCategory:
        """Default surface polysaccharide `ArchetypeCategory` when serotype tokens lack an explicit prefix."""
        return self.spec.default_category

    @property
    def archetype_mask(self) -> int:
        """32-bit bitmask of `LocusArchetype`s compatible with this species."""
        return self.spec.archetype_mask

    @property
    def aliases(self) -> tuple[bytes, ...]:
        """Taxonomic synonyms and genus aliases for this species."""
        return self.spec.aliases

    @property
    def synonyms(self) -> tuple[bytes, ...]:
        """Taxonomic synonyms and genus aliases for this species (alias for `self.aliases`)."""
        return self.spec.aliases

    @property
    def landmark_accessions(self) -> tuple[tuple[bytes, bytes], ...]:
        """Landmark INSDC BGC accession intervals for this species."""
        return self.spec.landmark_accessions

    def parse_serotype(
        self,
        raw: str | bytes,
        *,
        archetype: LocusArchetype | None = None,
    ) -> tuple[tuple[ArchetypeCategory, bytes], ...]:
        """Parse and normalize a single or multi-antigen serotype formula using this species' grammar."""
        return self.spec.parse_serotype(raw, archetype=archetype)

    def normalize_strain(
        self,
        raw_strain: str | bytes,
        raw_serotype: str | bytes = b"",
    ) -> bytes:
        """Normalize a strain isolate designation for this species."""
        return self.spec.normalize_strain(raw_strain, raw_serotype)

    @classmethod
    def from_name(cls, key: str | bytes | int | BacterialSpecies) -> BacterialSpecies:
        """Resolve a `BacterialSpecies` by enum ID, scientific name, enum name, or taxonomic alias."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        b_clean = b.split(b";", 1)[0].strip()
        lookup = _get_species_lookup()
        low = b_clean.lower()
        if low in lookup:
            return lookup[low]
        parts = b_clean.split()
        if len(parts) >= 2:
            binomial = (b"%s %s" % (parts[0], parts[1])).lower()
            if binomial in lookup:
                return lookup[binomial]
        raise KeyError(f"Unknown BacterialSpecies identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | BacterialSpecies,
        default: BacterialSpecies | None = None,
    ) -> BacterialSpecies:
        """Resolve a species identifier, falling back to `default` (or `BacterialSpecies.UNKNOWN`)."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback

    @classmethod
    def from_taxid(cls, taxid: int) -> BacterialSpecies:
        """Resolve a `BacterialSpecies` from an NCBI Taxonomy ID, or `BacterialSpecies.UNKNOWN`."""
        return _get_taxid_lookup().get(int(taxid), cls.UNKNOWN)

    @classmethod
    def specs(cls) -> SpeciesSpecs:
        """Return the cached singleton `SpeciesSpecs` SoA container over all `BacterialSpecies`."""
        return _get_species_specs()


@lru_cache(maxsize=1)
def _get_species_specs() -> SpeciesSpecs:
    """Build and cache the singleton `SpeciesSpecs` container over all `BacterialSpecies` (`0..51`)."""
    return SpeciesSpecs.from_list(
        (
            # 0: ACINETOBACTER_BAUMANNII
            SpeciesSpec(
                name=b"Acinetobacter baumannii",
                ncbi_taxid=470,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_ACINETOBACTER_RULES,
                aliases=(b"Acinetobacter sp.", b"Acinetobacter"),
                landmark_accessions=(
                    (b"KC526894", b"KC526921"),
                    (b"JX966412", b"JX966424"),
                    (b"MN166188", b"MN166198"),
                ),
            ),
            # 1: ACINETOBACTER_CALCOACETICUS
            SpeciesSpec(
                name=b"Acinetobacter calcoaceticus",
                ncbi_taxid=471,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 2: ACTINOBACILLUS_PLEUROPNEUMONIAE
            SpeciesSpec(
                name=b"Actinobacillus pleuropneumoniae",
                ncbi_taxid=715,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
                aliases=(b"Haemophilus pleuropneumoniae",),
            ),
            # 3: AEROMONAS_HYDROPHILA
            SpeciesSpec(
                name=b"Aeromonas hydrophila",
                ncbi_taxid=644,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Aeromonas sp.",),
            ),
            # 4: AEROMONAS_SALMONICIDA
            SpeciesSpec(
                name=b"Aeromonas salmonicida",
                ncbi_taxid=645,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
            ),
            # 5: AGGREGATIBACTER_ACTINOMYCETEMCOMITANS
            SpeciesSpec(
                name=b"Aggregatibacter actinomycetemcomitans",
                ncbi_taxid=714,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
                aliases=(b"Actinobacillus actinomycetemcomitans",),
            ),
            # 6: BACILLUS_CEREUS
            SpeciesSpec(
                name=b"Bacillus cereus",
                ncbi_taxid=1396,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.OTHER,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                aliases=(b"Bacillus sp.",),
            ),
            # 7: BACTEROIDES_FRAGILIS
            SpeciesSpec(
                name=b"Bacteroides fragilis",
                ncbi_taxid=817,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                aliases=(b"Bacteroides sp.",),
            ),
            # 8: BURKHOLDERIA_CEPACIA
            SpeciesSpec(
                name=b"Burkholderia cepacia",
                ncbi_taxid=292,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES | (1 << int(LocusArchetype.TEICHOIC_ACID)),
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Pseudomonas cepacia", b"Burkholderia sp."),
            ),
            # 9: BURKHOLDERIA_MULTIVORANS
            SpeciesSpec(
                name=b"Burkholderia multivorans",
                ncbi_taxid=87883,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 10: CAMPYLOBACTER_JEJUNI
            SpeciesSpec(
                name=b"Campylobacter jejuni",
                ncbi_taxid=197,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_CAMPY_RULES,
                serovar_map=((b"penner serovar 2", b"HS:2"),),
                aliases=(b"Campylobacter sp.",),
            ),
            # 11: CITROBACTER_BRAAKII
            SpeciesSpec(
                name=b"Citrobacter braakii",
                ncbi_taxid=57706,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 12: CITROBACTER_FREUNDII
            SpeciesSpec(
                name=b"Citrobacter freundii",
                ncbi_taxid=546,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Citrobacter sp.", b"Citrobacter"),
            ),
            # 13: CITROBACTER_WERKMANII
            SpeciesSpec(
                name=b"Citrobacter werkmanii",
                ncbi_taxid=67827,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 14: CRONOBACTER_DUBLINENSIS
            SpeciesSpec(
                name=b"Cronobacter dublinensis",
                ncbi_taxid=413497,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 15: CRONOBACTER_MALONATICUS
            SpeciesSpec(
                name=b"Cronobacter malonaticus",
                ncbi_taxid=413503,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 16: CRONOBACTER_SAKAZAKII
            SpeciesSpec(
                name=b"Cronobacter sakazakii",
                ncbi_taxid=28141,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Enterobacter sakazakii", b"Cronobacter sp.", b"Cronobacter"),
            ),
            # 17: CRONOBACTER_TURICENSIS
            SpeciesSpec(
                name=b"Cronobacter turicensis",
                ncbi_taxid=413502,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 18: ENTEROBACTER_CLOACAE
            SpeciesSpec(
                name=b"Enterobacter cloacae",
                ncbi_taxid=550,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Enterobacter sp.",),
            ),
            # 19: ENTEROCOCCUS_FAECALIS
            SpeciesSpec(
                name=b"Enterococcus faecalis",
                ncbi_taxid=1351,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                aliases=(b"Streptococcus faecalis", b"Enterococcus sp."),
            ),
            # 20: ESCHERICHIA_ALBERTII
            SpeciesSpec(
                name=b"Escherichia albertii",
                ncbi_taxid=208962,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 21: ESCHERICHIA_COLI
            SpeciesSpec(
                name=b"Escherichia coli",
                ncbi_taxid=562,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_ECOLI_RULES,
                aliases=(b"Escherichia sp.", b"Bacterium coli"),
                landmark_accessions=((b"AB811596", b"AB811624"), (b"AB812020", b"AB812084")),
            ),
            # 22: HAEMOPHILUS_INFLUENZAE
            SpeciesSpec(
                name=b"Haemophilus influenzae",
                ncbi_taxid=727,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
                aliases=(b"Haemophilus sp.",),
                known_serotypes=(b"a", b"b", b"c", b"d", b"e", b"f"),
            ),
            # 23: HELICOBACTER_PYLORI
            SpeciesSpec(
                name=b"Helicobacter pylori",
                ncbi_taxid=210,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Campylobacter pylori", b"Helicobacter sp."),
            ),
            # 24: KLEBSIELLA_PNEUMONIAE
            SpeciesSpec(
                name=b"Klebsiella pneumoniae",
                ncbi_taxid=573,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_KLEBSIELLA_RULES,
                aliases=(
                    b"Klebsiella sp.",
                    b"Klebsiella",
                    b"Klebsiella quasipneumoniae",
                    b"Klebsiella variicola",
                ),
                landmark_accessions=((b"AB924547", b"AB924624"), (b"LT174538", b"LT174607")),
            ),
            # 25: LACTOBACILLUS_DELBRUECKII
            SpeciesSpec(
                name=b"Lactobacillus delbrueckii",
                ncbi_taxid=1584,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                aliases=(b"Lactobacillus sp.",),
            ),
            # 26: NEISSERIA_MENINGITIDIS
            SpeciesSpec(
                name=b"Neisseria meningitidis",
                ncbi_taxid=487,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
                aliases=(b"Neisseria sp.",),
                known_serotypes=(b"A", b"B", b"C", b"W", b"W135", b"X", b"Y", b"Z", b"29E", b"I"),
            ),
            # 27: PLESIOMONAS_SHIGELLOIDES
            SpeciesSpec(
                name=b"Plesiomonas shigelloides",
                ncbi_taxid=703,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Aeromonas shigelloides",),
            ),
            # 28: PROTEUS_MIRABILIS
            SpeciesSpec(
                name=b"Proteus mirabilis",
                ncbi_taxid=584,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Proteus sp.", b"Proteus"),
                landmark_accessions=((b"KY710685", b"KY710739"), (b"HM146785", b"HM146790")),
            ),
            # 29: PROTEUS_MYXOFACIENS
            SpeciesSpec(
                name=b"Proteus myxofaciens",
                ncbi_taxid=184072,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Cosenzaea myxofaciens",),
            ),
            # 30: PROTEUS_PENNERI
            SpeciesSpec(
                name=b"Proteus penneri",
                ncbi_taxid=102862,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 31: PROTEUS_VULGARIS
            SpeciesSpec(
                name=b"Proteus vulgaris",
                ncbi_taxid=585,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Proteus hauseri",),
            ),
            # 32: PROVIDENCIA_RUSTIGIANII
            SpeciesSpec(
                name=b"Providencia rustigianii",
                ncbi_taxid=158850,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Providencia alcalifaciens", b"Providencia stuartii", b"Providencia sp."),
                landmark_accessions=((b"HQ880272", b"HQ880283"),),
            ),
            # 33: PSEUDOMONAS_AERUGINOSA
            SpeciesSpec(
                name=b"Pseudomonas aeruginosa",
                ncbi_taxid=287,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Pseudomonas pyocyanea", b"Pseudomonas sp."),
                landmark_accessions=((b"AF498400", b"AF498420"),),
            ),
            # 34: RAOULTELLA_TERRIGENA
            SpeciesSpec(
                name=b"Raoultella terrigena",
                ncbi_taxid=577,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Klebsiella terrigena", b"Raoultella sp."),
            ),
            # 35: SALMONELLA_ENTERICA
            SpeciesSpec(
                name=b"Salmonella enterica",
                ncbi_taxid=28901,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES | (1 << int(LocusArchetype.TEICHOIC_ACID)),
                serotype_rules=_O_ONLY_RULES,
                serovar_map=_SALMONELLA_SEROVARS,
                aliases=(
                    b"Salmonella typhimurium",
                    b"Salmonella typhi",
                    b"Salmonella enteritidis",
                    b"Salmonella paratyphi",
                    b"Salmonella choleraesuis",
                    b"Salmonella bongori",
                    b"Salmonella sp.",
                    b"Salmonella",
                ),
                landmark_accessions=((b"JX975325", b"JX975352"),),
            ),
            # 36: SHIGELLA_BOYDII
            SpeciesSpec(
                name=b"Shigella boydii",
                ncbi_taxid=621,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
                landmark_accessions=((b"EU296402", b"EU296419"),),
            ),
            # 37: SHIGELLA_DYSENTERIAE
            SpeciesSpec(
                name=b"Shigella dysenteriae",
                ncbi_taxid=622,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
                landmark_accessions=((b"EU294162", b"EU294178"),),
            ),
            # 38: SHIGELLA_FLEXNERI
            SpeciesSpec(
                name=b"Shigella flexneri",
                ncbi_taxid=623,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
                aliases=(b"Shigella sp.", b"Shigella"),
                known_serotypes=(
                    b"1a",
                    b"1b",
                    b"2",
                    b"2a",
                    b"2a2",
                    b"2a3",
                    b"2b",
                    b"3a",
                    b"3b",
                    b"4a",
                    b"5a",
                    b"6",
                    b"X",
                    b"Y",
                ),
            ),
            # 39: SHIGELLA_SONNEI
            SpeciesSpec(
                name=b"Shigella sonnei",
                ncbi_taxid=624,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=(),
                known_serotypes=(b"Phase I", b"Phase II"),
            ),
            # 40: STAPHYLOCOCCUS_AUREUS
            SpeciesSpec(
                name=b"Staphylococcus aureus",
                ncbi_taxid=1280,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                serotype_rules=_STAPH_RULES,
                aliases=(b"Staphylococcus sp.",),
            ),
            # 41: STREPTOCOCCUS_ORALIS
            SpeciesSpec(
                name=b"Streptococcus oralis",
                ncbi_taxid=1303,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
            ),
            # 42: STREPTOCOCCUS_PNEUMONIAE
            SpeciesSpec(
                name=b"Streptococcus pneumoniae",
                ncbi_taxid=1313,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
                aliases=(b"Diplococcus pneumoniae", b"Pneumococcus", b"Streptococcus sp."),
                landmark_accessions=((b"CR931632", b"CR931722"),),
            ),
            # 43: STREPTOCOCCUS_SUIS
            SpeciesSpec(
                name=b"Streptococcus suis",
                ncbi_taxid=1307,
                gram_stain=GramStain.POSITIVE,
                default_category=ArchetypeCategory.CAPSULE,
                archetype_mask=_COMMON_GRAM_POS_ARCHETYPES,
                serotype_rules=_CAPSULE_UNPREFIXED_RULES,
            ),
            # 44: VIBRIO_CHOLERAE
            SpeciesSpec(
                name=b"Vibrio cholerae",
                ncbi_taxid=666,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Vibrio sp.",),
            ),
            # 45: YERSINIA_BERCOVIERI
            SpeciesSpec(
                name=b"Yersinia bercovieri",
                ncbi_taxid=634,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 46: YERSINIA_ENTEROCOLITICA
            SpeciesSpec(
                name=b"Yersinia enterocolitica",
                ncbi_taxid=630,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 47: YERSINIA_FREDERIKSENII
            SpeciesSpec(
                name=b"Yersinia frederiksenii",
                ncbi_taxid=29484,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 48: YERSINIA_KRISTENSENII
            SpeciesSpec(
                name=b"Yersinia kristensenii",
                ncbi_taxid=28152,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 49: YERSINIA_PSEUDOTUBERCULOSIS
            SpeciesSpec(
                name=b"Yersinia pseudotuberculosis",
                ncbi_taxid=633,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
                aliases=(b"Pasteurella pseudotuberculosis", b"Yersinia sp.", b"Yersinia"),
                landmark_accessions=((b"AJ536295", b"AJ536305"),),
            ),
            # 50: YERSINIA_SIMILIS
            SpeciesSpec(
                name=b"Yersinia similis",
                ncbi_taxid=367190,
                gram_stain=GramStain.NEGATIVE,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES,
                serotype_rules=_O_ONLY_RULES,
            ),
            # 51: UNKNOWN
            SpeciesSpec(
                name=b"Unknown bacterium",
                ncbi_taxid=0,
                gram_stain=GramStain.UNKNOWN,
                default_category=ArchetypeCategory.O_ANTIGEN,
                archetype_mask=_COMMON_GRAM_NEG_ARCHETYPES | _COMMON_GRAM_POS_ARCHETYPES,
                serotype_rules=_ECOLI_RULES,
                aliases=(b"unknown", b"bacterium", b"unclassified"),
            ),
        )
    )


@lru_cache(maxsize=1)
def _get_species_lookup() -> dict[bytes, BacterialSpecies]:
    """Build and cache the case-insensitive name/alias lookup dictionary for `BacterialSpecies`."""
    specs = _get_species_specs()
    lookup: dict[bytes, BacterialSpecies] = {}
    for sp in BacterialSpecies:
        spec = specs[int(sp)]
        lookup[sp.name.encode("ascii").lower()] = sp
        lookup[spec.name.lower()] = sp
        for alias in spec.aliases:
            lookup[alias.strip().lower()] = sp
    return lookup


@lru_cache(maxsize=1)
def _get_taxid_lookup() -> dict[int, BacterialSpecies]:
    """Build and cache the NCBI TaxID lookup dictionary for `BacterialSpecies`."""
    specs = _get_species_specs()
    return {int(specs.ncbi_taxids[int(sp)]): sp for sp in BacterialSpecies if int(specs.ncbi_taxids[int(sp)]) > 0}
