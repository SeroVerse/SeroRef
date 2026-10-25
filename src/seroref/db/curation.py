"""Multi-tier reference strain, assembly, and SeroRef locus discovery and curation."""

from __future__ import annotations

import asyncio
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from seroverse.client import (
    ENAClient,
    NCBIClient,
    NCTCReferenceStrain,
    NuccoreSummary,
    StrainInfoClient,
    StrainPassport,
    load_msgpack,
    save_msgpack,
)
from seroverse.core.collections import RaggedArrayContainer, ScalarView, lengths_to_offsets
from seroverse.core.genome import GenomeAssembly

from seroref.db.models import SeroRefRecord
from seroref.db.records import SeroRefRecords, loci_from_npz_dict, loci_to_npz_dict
from seroref.loci.archetypes import ArchetypeCategory, LocusArchetype
from seroref.loci.models import Loci, Locus
from seroref.loci.predictor import LocusPredictor

DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / ".cache"
DEFAULT_LOCI_CACHE_PATH: Path = DEFAULT_CACHE_DIR / "predicted_loci.npz"

OGROUP_TO_SALMONELLA_SEROVARS: dict[str, tuple[str, ...]] = {
    "O2": ("Paratyphi A",),
    "O4": ("Typhimurium", "Paratyphi B", "Heidelberg", "Agona", "Derby", "Saintpaul"),
    "O7": ("Choleraesuis", "Montevideo", "Infantis", "Thompson", "Paratyphi C"),
    "O8": ("Newport", "Muenchen", "Kentucky", "Hadar"),
    "O9": ("Typhi", "Enteritidis", "Dublin", "Gallinarum", "Pullorum"),
    "O3,10": ("Anatum", "Give", "London"),
    "O1,3,19": ("Senftenberg",),
    "O11": ("Aberdeen",),
    "O13": ("Poona",),
    "O18": ("Cerro",),
    "O21": ("Minnesota",),
    "O30": ("Urbana",),
    "O35": ("Adelaide",),
}

_WORD_SPLIT_RE = re.compile(r"[A-Za-z0-9]+(?::[A-Za-z0-9]+)?")
_NUM_SERO_RE = re.compile(r"(?:serotype|serogroup|type|capsule|b-band|osa|iats)\s+O?([0-9]{1,3}[A-Za-z]?)\b", re.I)
_COLLECTION_RE = re.compile(
    r"\b(ATCC|NCTC|CCUG|CNCTC|PCM|CCM|IMET|IAM|CIP|JCM|DSM|LMG|CECT|NCIMB|VKM|Prague|PrK)\b", re.I
)
_ECOLI_OK_SPLIT_RE = re.compile(r"\b(O\d+[a-z]?|K\d+[a-z]?)\b", re.I)
_GENUS_LEVEL_TAXA = frozenset(
    "Klebsiella Acinetobacter Salmonella Shigella Proteus Yersinia Campylobacter Cronobacter Burkholderia "
    "Providencia Raoultella Serratia Citrobacter Enterobacter Morganella Hafnia Edwardsiella Vibrio Aeromonas "
    "Shewanella Pseudoalteromonas Rhodococcus Rhizobium Sinorhizobium Bradyrhizobium Agrobacterium Azospirillum "
    "Ochrobactrum Xanthomonas Pantoea Erwinia Pectobacterium Dickeya Bacteroides Clostridium Bacillus "
    "Lactobacillus Bifidobacterium".split()
)


def extract_multi_antigen_tokens(organism: str, raw_serotype: str) -> list[tuple[bytes, bytes]]:
    """Extract `(antigen_type_bytes, normalized_serotype_bytes)` pairs from a raw ENA/StrainInfo serotype string."""
    s = raw_serotype.strip()
    if not s:
        return []
    org_low = organism.lower()
    out: list[tuple[bytes, bytes]] = []
    if "escherichia" in org_low:
        for m in _ECOLI_OK_SPLIT_RE.finditer(s):
            tok = m.group(1).upper()
            at = b"CPS" if tok.startswith("K") else b"O-antigen"
            pair = (at, tok.encode("utf-8"))
            if pair not in out:
                out.append(pair)
        if out:
            return out
    if "klebsiella" in org_low and (
        m_k := re.search(r"(?:capsular\s+serotype|serotype|type|k)\s*[-:]?\s*(\d{1,3}[a-z]?)", s, re.I)
    ):
        return [(b"CPS", f"K{m_k.group(1)}".encode())]
    if "pseudomonas aeruginosa" in org_low and (
        m_o := re.search(r"(?:serotype|type|o)\s*[-:]?\s*(\d{1,2}[a-z]?)", s, re.I)
    ):
        return [(b"O-antigen", f"O{m_o.group(1)}".encode())]
    if "salmonella" in org_low:
        sv = s.split(",")[0].removeprefix("serotype ").removeprefix("serovar ").strip()
        if sv:
            return [(b"O-antigen", sv.encode("utf-8"))]
    at_default = b"CPS" if s.upper().startswith(("K", "KL", "TYPE ")) else b"O-antigen"
    return [(at_default, s.encode("utf-8"))]


@dataclass(frozen=True, slots=True)
class SerotypeTokenIndex:
    """Compiled multi-tier lexical & regex index for matching serotype/strain tokens in NCBI summaries."""

    num_map: dict[str, tuple[str, ...]]
    word_map: dict[str, tuple[str, ...]]
    multi_map: dict[str, tuple[tuple[str, re.Pattern[str]], ...]]

    @classmethod
    def build(cls, targets: Iterable[str]) -> Self:
        num_acc: dict[str, list[str]] = defaultdict(list)
        word_acc: dict[str, list[str]] = defaultdict(list)
        multi_acc: dict[str, list[tuple[str, re.Pattern[str]]]] = defaultdict(list)

        for tok_s in targets:
            if not tok_s:
                continue
            if tok_s.isdigit():
                num_acc[tok_s.upper()].append(tok_s)
                continue
            if not (words := _WORD_SPLIT_RE.findall(tok_s)):
                continue
            if len(words) == 1 and words[0] == tok_s:
                u_tok = tok_s.upper()
                word_acc[u_tok].append(tok_s)
                if u_tok.startswith("K") and u_tok[1:].isdigit():
                    word_acc["KL" + u_tok[1:]].append(tok_s)
                elif u_tok.startswith("KL") and u_tok[2:].isdigit():
                    word_acc["K" + u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O:") and u_tok[2:].isdigit():
                    word_acc["O" + u_tok[2:]].append(tok_s)
                    num_acc[u_tok[2:]].append(tok_s)
                elif u_tok.startswith("O") and u_tok[1:].isdigit():
                    word_acc["O:" + u_tok[1:]].append(tok_s)
                    num_acc[u_tok[1:]].append(tok_s)
                for sv_alias in OGROUP_TO_SALMONELLA_SEROVARS.get(u_tok, ()):
                    if len(sv_words := _WORD_SPLIT_RE.findall(sv_alias)) == 1:
                        word_acc[sv_words[0].upper()].append(tok_s)
            else:
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(tok_s)}(?![A-Za-z0-9])", re.I)
                multi_acc[max(words, key=len).upper()].append((tok_s, pat))

        return cls(
            {k: tuple(v) for k, v in num_acc.items()},
            {k: tuple(v) for k, v in word_acc.items()},
            {k: tuple(v) for k, v in multi_acc.items()},
        )

    def match(self, summary: NuccoreSummary) -> set[str]:
        matched: set[str] = set()
        blob_txt = summary.blob_text
        for w in {w.upper() for w in _WORD_SPLIT_RE.findall(blob_txt)}:
            if w in self.word_map:
                matched.update(self.word_map[w])
            for orig_tok, pat in self.multi_map.get(w, ()):
                if orig_tok not in matched and pat.search(blob_txt):
                    matched.add(orig_tok)
        if self.num_map:
            for m_num in _NUM_SERO_RE.findall(summary.title):
                if (u_num := m_num.upper()) in self.num_map:
                    matched.update(self.num_map[u_num])
            for st_k, sn_v in summary.sub_pairs:
                if st_k.strip().lower() in ("serotype", "serogroup", "serovar"):
                    u_sn = sn_v.strip().upper()
                    if u_sn in self.num_map:
                        matched.update(self.num_map[u_sn])
                    elif u_sn.startswith("O") and u_sn[1:] in self.num_map:
                        matched.update(self.num_map[u_sn[1:]])
        return matched


@dataclass(frozen=True, slots=True)
class LociCacheEntry(ScalarView):
    """Scalar view for a single cached `(cache_key, assembly_id, Loci)` row in `LociCacheBundle`."""

    cache_key: bytes
    assembly_id: bytes
    loci: Loci


@dataclass(slots=True)
class LociCacheBundle(RaggedArrayContainer[LociCacheEntry, "LociCacheBundle"]):
    """Zero-pickle `RaggedArrayContainer` storing flat `Loci` and offsets/lengths for zero-copy `.npz` load/save."""

    cache_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))
    asm_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S32"))
    loci: Loci = field(default_factory=Loci.empty)
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    strain_orgs: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S64"))
    strain_toks: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S64"))
    strain_keys: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))
    doi_pmids: np.ndarray[tuple[int], np.dtype[np.int64]] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    doi_vals: np.ndarray[tuple[int], np.dtype[np.bytes_]] = field(default_factory=lambda: np.empty(0, dtype="S96"))

    _key_to_idx: dict[bytes, int] | None = field(default=None, init=False, repr=False)
    _base_to_idx: dict[bytes, int] | None = field(default=None, init=False, repr=False)
    _entry_memo: dict[int, LociCacheEntry] = field(default_factory=dict, init=False, repr=False)
    _assembly_loci_dict: dict[bytes, tuple[bytes, Loci]] | None = field(default=None, init=False, repr=False)
    _strain_dict: dict[tuple[bytes, bytes], bytes] | None = field(default=None, init=False, repr=False)
    _doi_dict: dict[int, bytes] | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.offsets) if self._assembly_loci_dict is None else len(self._assembly_loci_dict)

    @property
    def is_warm(self) -> bool:
        if self._assembly_loci_dict is not None:
            return bool(self.assembly_loci and self.strain_to_assembly and self.pmid_to_doi)
        return len(self.cache_keys) > 0 and len(self.strain_keys) > 0 and len(self.doi_pmids) > 0

    def _ensure_indices(self) -> tuple[dict[bytes, int], dict[bytes, int]]:
        if self._key_to_idx is None or self._base_to_idx is None:
            k_map: dict[bytes, int] = {}
            b_map: dict[bytes, int] = {}
            for i, k_raw in enumerate(self.cache_keys):
                k_b = bytes(k_raw).rstrip(b"\x00")
                k_map[k_b] = i
                if not k_b.startswith((b"nuccore:", b"pmid:", b"uid:")):
                    b_map[k_b.split(b".")[0]] = i
            self._key_to_idx = k_map
            self._base_to_idx = b_map
        return self._key_to_idx, self._base_to_idx

    def _entry_at(self, idx: int) -> LociCacheEntry:
        if (cached := self._entry_memo.get(idx)) is not None:
            return cached
        off, ln = int(self.offsets[idx]), int(self.lengths[idx])
        loci_slice = self.loci[off : off + ln] if ln > 0 else Loci.empty()
        entry = LociCacheEntry(
            cache_key=bytes(self.cache_keys[idx]).rstrip(b"\x00"),
            assembly_id=bytes(self.asm_ids[idx]).rstrip(b"\x00"),
            loci=loci_slice,
        )
        self._entry_memo[idx] = entry
        return entry

    def get_entry(self, key: bytes, *, allow_base_acc: bool = False) -> LociCacheEntry | None:
        if self._assembly_loci_dict is not None:
            hit = self._assembly_loci_dict.get(key) or (self.base_acc_lookup.get(key) if allow_base_acc else None)
            return LociCacheEntry(key, hit[0], hit[1]) if hit is not None else None
        k_map, b_map = self._ensure_indices()
        idx = k_map.get(key)
        if idx is None and allow_base_acc:
            idx = b_map.get(key)
        return self._entry_at(idx) if idx is not None else None

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> LociCacheEntry: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> LociCacheBundle: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> LociCacheEntry | LociCacheBundle: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> LociCacheEntry | LociCacheBundle:
        if self._assembly_loci_dict is not None:
            self._sync_arrays_from_dicts()
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self.offsets)
            if idx < 0 or idx >= len(self.offsets):
                raise IndexError("LociCacheBundle index out of range")
            return self._entry_at(idx)

        indices, new_offsets, new_lengths, (loci_idx,) = self._slice_ragged(
            item, np.arange(len(self.loci), dtype=np.int32)
        )
        if len(indices) == 0:
            return self.empty(0)
        return LociCacheBundle(
            cache_keys=self.cache_keys[indices],
            asm_ids=self.asm_ids[indices],
            loci=self.loci[loci_idx] if len(loci_idx) > 0 else Loci.empty(),
            offsets=new_offsets,
            lengths=new_lengths,
            strain_orgs=self.strain_orgs,
            strain_toks=self.strain_toks,
            strain_keys=self.strain_keys,
            doi_pmids=self.doi_pmids,
            doi_vals=self.doi_vals,
        )

    @classmethod
    def empty(cls, length: int = 0) -> Self:
        z_i32 = np.zeros(length, dtype=np.int32) if length > 0 else np.empty(0, dtype=np.int32)
        return cls(
            cache_keys=np.zeros(length, dtype="S96") if length > 0 else np.empty(0, dtype="S96"),
            asm_ids=np.zeros(length, dtype="S32") if length > 0 else np.empty(0, dtype="S32"),
            loci=Loci.empty(),
            offsets=z_i32,
            lengths=z_i32.copy(),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty(0)
        if len(batch_list) == 1:
            return batch_list[0]
        for b in batch_list:
            if b._assembly_loci_dict is not None:
                b._sync_arrays_from_dicts()
        new_lens = np.concatenate([b.lengths for b in batch_list])
        loci_non = [b.loci for b in batch_list if len(b.loci) > 0]
        return cls(
            cache_keys=np.concatenate([b.cache_keys for b in batch_list]),
            asm_ids=np.concatenate([b.asm_ids for b in batch_list]),
            loci=Loci.concat(loci_non) if loci_non else Loci.empty(),
            offsets=lengths_to_offsets(new_lens),
            lengths=new_lens,
            strain_orgs=np.concatenate([b.strain_orgs for b in batch_list if len(b.strain_orgs) > 0])
            if any(len(b.strain_orgs) > 0 for b in batch_list)
            else np.empty(0, dtype="S64"),
            strain_toks=np.concatenate([b.strain_toks for b in batch_list if len(b.strain_toks) > 0])
            if any(len(b.strain_toks) > 0 for b in batch_list)
            else np.empty(0, dtype="S64"),
            strain_keys=np.concatenate([b.strain_keys for b in batch_list if len(b.strain_keys) > 0])
            if any(len(b.strain_keys) > 0 for b in batch_list)
            else np.empty(0, dtype="S96"),
            doi_pmids=np.concatenate([b.doi_pmids for b in batch_list if len(b.doi_pmids) > 0])
            if any(len(b.doi_pmids) > 0 for b in batch_list)
            else np.empty(0, dtype=np.int64),
            doi_vals=np.concatenate([b.doi_vals for b in batch_list if len(b.doi_vals) > 0])
            if any(len(b.doi_vals) > 0 for b in batch_list)
            else np.empty(0, dtype="S96"),
        )

    @classmethod
    def from_dicts(
        cls,
        assembly_loci: dict[bytes, tuple[bytes, Loci]],
        strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
        pmid_to_doi: dict[int, bytes] | None = None,
    ) -> Self:
        items = list(assembly_loci.items())
        n = len(items)
        keys_arr = np.array([k[:96] for k, _ in items], dtype="S96") if n > 0 else np.empty(0, dtype="S96")
        asm_arr = np.array([a[:32] for _, (a, _) in items], dtype="S32") if n > 0 else np.empty(0, dtype="S32")
        lengths = np.fromiter((len(loc) for _, (_, loc) in items), dtype=np.int32, count=n)
        offsets = lengths_to_offsets(lengths)
        non_empty = [loc for _, (_, loc) in items if len(loc) > 0]
        flat_loci = Loci.concat(non_empty) if non_empty else Loci.empty()

        s_items = sorted((strain_to_assembly or {}).items())
        s_orgs = np.array([o[:64] for (o, _), _ in s_items], dtype="S64") if s_items else np.empty(0, dtype="S64")
        s_toks = np.array([t[:64] for (_, t), _ in s_items], dtype="S64") if s_items else np.empty(0, dtype="S64")
        s_keys = np.array([ck[:96] for _, ck in s_items], dtype="S96") if s_items else np.empty(0, dtype="S96")

        d_items = sorted((pmid_to_doi or {}).items())
        d_pmids = np.array([p for p, _ in d_items], dtype=np.int64) if d_items else np.empty(0, dtype=np.int64)
        d_vals = np.array([d[:96] for _, d in d_items], dtype="S96") if d_items else np.empty(0, dtype="S96")

        return cls(keys_arr, asm_arr, flat_loci, offsets, lengths, s_orgs, s_toks, s_keys, d_pmids, d_vals)

    def _sync_arrays_from_dicts(self) -> None:
        if self._assembly_loci_dict is None and self._strain_dict is None and self._doi_dict is None:
            return
        rebuilt = self.from_dicts(self.assembly_loci, self.strain_to_assembly, self.pmid_to_doi)
        self.cache_keys, self.asm_ids, self.loci = rebuilt.cache_keys, rebuilt.asm_ids, rebuilt.loci
        self.offsets, self.lengths = rebuilt.offsets, rebuilt.lengths
        self.strain_orgs, self.strain_toks, self.strain_keys = (
            rebuilt.strain_orgs,
            rebuilt.strain_toks,
            rebuilt.strain_keys,
        )
        self.doi_pmids, self.doi_vals = rebuilt.doi_pmids, rebuilt.doi_vals
        self._key_to_idx = None
        self._base_to_idx = None
        self._entry_memo.clear()
        self._assembly_loci_dict = None

    @property
    def assembly_loci(self) -> dict[bytes, tuple[bytes, Loci]]:
        if self._assembly_loci_dict is None:
            self._assembly_loci_dict = {
                (e := self._entry_at(i)).cache_key: (e.assembly_id, e.loci) for i in range(len(self.offsets))
            }
        return self._assembly_loci_dict

    @property
    def strain_to_assembly(self) -> dict[tuple[bytes, bytes], bytes]:
        if self._strain_dict is None:
            s_map: dict[tuple[bytes, bytes], bytes] = {
                (p[1], p[2]): k_b
                for k_raw in self.cache_keys
                if (k_b := bytes(k_raw).rstrip(b"\x00")).startswith(b"nuccore:")
                and len(p := k_b.split(b":", 2)) == 3
                and p[1]
                and p[2]
            }
            for o_r, t_r, ck_r in zip(self.strain_orgs, self.strain_toks, self.strain_keys, strict=False):
                o_b, t_b, ck_b = bytes(o_r).rstrip(b"\x00"), bytes(t_r).rstrip(b"\x00"), bytes(ck_r).rstrip(b"\x00")
                if o_b and t_b and ck_b:
                    s_map[(o_b, t_b)] = ck_b
            self._strain_dict = s_map
        return self._strain_dict

    @property
    def pmid_to_doi(self) -> dict[int, bytes]:
        if self._doi_dict is None:
            self._doi_dict = {
                int(p): bytes(d).rstrip(b"\x00")
                for p, d in zip(self.doi_pmids, self.doi_vals, strict=False)
                if int(p) > 0 and bytes(d).rstrip(b"\x00")
            }
        return self._doi_dict

    @property
    def base_acc_lookup(self) -> dict[bytes, tuple[bytes, Loci]]:
        return {
            k_acc.split(b".")[0]: val
            for k_acc, val in self.assembly_loci.items()
            if not k_acc.startswith((b"nuccore:", b"pmid:", b"uid:"))
        }

    def invalidate_index(self) -> None:
        self._key_to_idx = None
        self._base_to_idx = None
        self._entry_memo.clear()

    def as_tuple(self) -> tuple[dict[bytes, tuple[bytes, Loci]], dict[tuple[bytes, bytes], bytes], dict[int, bytes]]:
        return self.assembly_loci, self.strain_to_assembly, self.pmid_to_doi

    @classmethod
    def load(cls, path: Path = DEFAULT_LOCI_CACHE_PATH) -> Self:
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys, asm_ids, indptr = data["cache_keys"], data["asm_ids"], data["key_offsets"]
                    flat_loci = loci_from_npz_dict(data, prefix="loci_")
                    s_orgs = data["strain_orgs"] if "strain_orgs" in data else np.empty(0, dtype="S64")
                    s_toks = data["strain_toks"] if "strain_toks" in data else np.empty(0, dtype="S64")
                    s_keys = data["strain_keys"] if "strain_keys" in data else np.empty(0, dtype="S96")
                    d_pmids = data["doi_pmids"] if "doi_pmids" in data else np.empty(0, dtype=np.int64)
                    d_vals = data["doi_vals"] if "doi_vals" in data else np.empty(0, dtype="S96")
                offsets = indptr[:-1].astype(np.int32, copy=False) if len(indptr) > 1 else np.empty(0, dtype=np.int32)
                lengths = (
                    np.diff(indptr).astype(np.int32, copy=False) if len(indptr) > 1 else np.empty(0, dtype=np.int32)
                )
                return cls(keys, asm_ids, flat_loci, offsets, lengths, s_orgs, s_toks, s_keys, d_pmids, d_vals)
            except Exception:
                pass
        return cls.empty(0)

    def save(self, path: Path = DEFAULT_LOCI_CACHE_PATH) -> None:
        self._sync_arrays_from_dicts()
        if len(self.cache_keys) == 0:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            extra: dict[str, Any] = {}
            if len(self.strain_keys) > 0:
                extra["strain_orgs"] = self.strain_orgs
                extra["strain_toks"] = self.strain_toks
                extra["strain_keys"] = self.strain_keys
            if len(self.doi_pmids) > 0:
                extra["doi_pmids"] = self.doi_pmids
                extra["doi_vals"] = self.doi_vals
            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
            loci_arrays = loci_to_npz_dict(self.loci, prefix="loci_")
            np.savez_compressed(
                tmp_path,
                cache_keys=self.cache_keys,
                asm_ids=self.asm_ids,
                key_offsets=self.indptr,
                **extra,
                **loci_arrays,
            )
            os.replace(tmp_path, path)
        except Exception:
            pass


def load_predicted_loci_cache(path: Path = DEFAULT_LOCI_CACHE_PATH) -> dict[bytes, tuple[bytes, Loci]]:
    """Load zero-pickle cached `(asm_id, Loci)` entries from `.cache/predicted_loci.npz`."""
    return LociCacheBundle.load(path).assembly_loci


def save_predicted_loci_cache(
    path: Path,
    cache: dict[bytes, tuple[bytes, Loci]],
    *,
    strain_to_assembly: dict[tuple[bytes, bytes], bytes] | None = None,
    pmid_to_doi: dict[int, bytes] | None = None,
) -> None:
    """Persist `(asm_id, Loci)` entries (plus optional strain/DOI maps) to `.cache/predicted_loci.npz` without pickle."""
    LociCacheBundle.from_dicts(cache, strain_to_assembly, pmid_to_doi).save(path)


def _clean_fasta_acc(raw_header: bytes) -> str:
    first_tok = raw_header.split()[0].decode("utf-8", errors="replace").lstrip(">")
    if "|" in first_tok:
        parts = [
            p for p in first_tok.split("|") if p and not p.isdigit() and p not in ("gb", "emb", "dbj", "ref", "gi")
        ]
        if parts:
            first_tok = parts[-1]
    return first_tok[:32]


def _catalog_taxon(org_b: bytes) -> str:
    if org_b in (b"Unknown bacterium", b"Bacteria (KEGG pathway)"):
        return ""
    parts = org_b.decode("utf-8", errors="replace").strip().removesuffix(" sp.").strip().split()
    if not parts:
        return ""
    return parts[0] if parts[0] in _GENUS_LEVEL_TAXA else " ".join(parts[:2])


def _is_true_strain_isolate(org_b: bytes, str_b: bytes, sero_b: bytes) -> bool:
    bad_orgs = (b"Unknown bacterium", b"Bacteria (KEGG pathway)")
    if not str_b or len(str_b) < 2 or str_b == sero_b or b"sp." in org_b or org_b in bad_orgs:
        return False
    pat = r"(?:type\s+|serotype\s+|serogroup\s+)?(?:K|KL|O|OL|OC|OCL|VI|\d{1,3}[A-Za-z]?|[IVX]{1,4}[abAB]?)"
    return not bool(re.fullmatch(pat, str_b.decode("utf-8", errors="replace").strip(), re.I))


@dataclass(slots=True)
class DiscoveryState:
    """Shared mutable state across discovery tiers during `resolve_and_predict_loci`."""

    bundle: LociCacheBundle
    base_acc_in_cache: dict[str, bytes]
    tier1_uids: set[str] = field(default_factory=set)
    pmid_to_nuccore_uids: dict[int, list[str]] = field(default_factory=dict)
    uid_to_summary: dict[str, NuccoreSummary] = field(default_factory=dict)
    uid_to_accver: dict[str, str] = field(default_factory=dict)
    selected_key_uids: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    uid_priority: Counter[str] = field(default_factory=Counter)
    uid_to_predicted: dict[str, tuple[bytes, list[Locus]]] = field(default_factory=dict)
    passports_by_org: dict[bytes, list[StrainPassport]] = field(default_factory=dict)

    def pool_predicted_uids(self, uids: Iterable[str]) -> tuple[bytes, Loci] | None:
        pool: list[Locus] = []
        first_acc = b""
        for u in uids:
            if u in self.uid_to_predicted:
                acc_b, l_list = self.uid_to_predicted[u]
                if not first_acc:
                    first_acc = acc_b
                pool.extend(l_list)
        return (first_acc, Loci.from_list(pool)) if pool and first_acc else None


@dataclass(slots=True)
class NCBILocusRunner:
    """Shared rate-limited `NCBIClient` wrapper & multi-FASTA `LocusPredictor` runner."""

    ncbi: NCBIClient
    batch_predictor: LocusPredictor = field(
        default_factory=lambda: LocusPredictor(min_completeness=0.65, stitch_contig_splits=False)
    )
    whole_genome_predictor: LocusPredictor = field(default_factory=lambda: LocusPredictor(min_completeness=0.75))

    def predict_multi_fasta_batch(self, fasta_blob: bytes) -> dict[str, tuple[bytes, list[Locus]]]:
        grouped: dict[str, tuple[bytes, list[Locus]]] = {}
        stripped = fasta_blob.lstrip()
        if not stripped.startswith(b">"):
            return grouped
        try:
            bgc_entries: list[bytes] = []
            for idx_e, chunk in enumerate(stripped.split(b"\n>")):
                entry = chunk if idx_e == 0 else b">" + chunk
                nl_pos = entry.find(b"\n")
                if nl_pos <= 1 or not (acc_tok := _clean_fasta_acc(entry[1:nl_pos])):
                    continue
                grouped.setdefault(acc_tok.split(".")[0], (acc_tok.encode("utf-8")[:32], []))
                if 1_500 <= (len(entry) - nl_pos) <= 95_000:
                    bgc_entries.append(entry)
            if not bgc_entries:
                return grouped
            asm = GenomeAssembly.from_fasta(b"\n".join(bgc_entries), id_=b"batch", truncate_id=True)
            if len(asm) == 0 or len(pred_loci := self.batch_predictor.predict(asm)) == 0:
                return grouped
            by_cidx: dict[int, list[Locus]] = defaultdict(list)
            for j in range(len(pred_loci)):
                loc = pred_loci[j].canonicalize_strand()
                by_cidx[int(loc.contig_index)].append(loc)
            for c_idx, loc_list in by_cidx.items():
                if 0 <= c_idx < len(asm.ids):
                    acc_tok = _clean_fasta_acc(bytes(asm.ids[c_idx]).strip(b"\x00").strip())
                    tagged = [replace(loc.with_name(acc_tok), contig_index=0) for loc in loc_list]
                    grouped[acc_tok.split(".")[0]] = (acc_tok.encode("utf-8")[:32], tagged)
        except Exception:
            pass
        return grouped

    async def fetch_and_predict_nuccore_uids(
        self, uids: Sequence[str], state: DiscoveryState, *, batch_size: int = 40
    ) -> None:
        empty_loci = Loci.empty()
        accbase_to_uids: dict[str, list[str]] = defaultdict(list)
        uids_to_fetch: list[str] = []

        for u in uids:
            base_av = state.uid_to_accver.get(u, "").split(".")[0]
            if base_av:
                accbase_to_uids[base_av].append(u)
            uid_key = b"uid:" + u.encode("utf-8")
            cache_k = state.base_acc_in_cache[base_av] if base_av and base_av in state.base_acc_in_cache else uid_key
            if (hit := state.bundle.assembly_loci.get(cache_k)) is not None:
                if len(hit[1]) > 0:
                    state.uid_to_predicted[u] = (hit[0], [hit[1][j] for j in range(len(hit[1]))])
            else:
                uids_to_fetch.append(u)

        for i in range(0, len(uids_to_fetch), batch_size):
            batch_uids = uids_to_fetch[i : i + batch_size]
            content = await self.ncbi.efetch_fasta(batch_uids, db="nuccore")
            if content.startswith(b">"):
                for base_tok, (acc_b, tagged_list) in self.predict_multi_fasta_batch(content).items():
                    loci_obj = Loci.from_list(tagged_list) if tagged_list else empty_loci
                    state.bundle.assembly_loci[acc_b] = (acc_b, loci_obj)
                    state.base_acc_in_cache[base_tok] = acc_b
                    for matched_u in accbase_to_uids.get(base_tok, ()):
                        state.bundle.assembly_loci[b"uid:" + matched_u.encode("utf-8")] = (acc_b, loci_obj)
                        if tagged_list:
                            state.uid_to_predicted[matched_u] = (acc_b, tagged_list)
                for u_b in batch_uids:
                    u_key = b"uid:" + u_b.encode("utf-8")
                    if u_key not in state.bundle.assembly_loci:
                        base_u = state.uid_to_accver.get(u_b, "").split(".")[0]
                        acc_u_b = base_u.encode("utf-8")[:32] if base_u else b"empty"
                        state.bundle.assembly_loci[u_key] = (acc_u_b, empty_loci)
                        if base_u and base_u not in state.base_acc_in_cache:
                            state.bundle.assembly_loci[acc_u_b] = (acc_u_b, empty_loci)
                            state.base_acc_in_cache[base_u] = acc_u_b
        state.bundle.invalidate_index()


class ENANCTC3000Tier:
    """Bulk ENA `PRJEB6403` (NCTC3000) reference strain and serotype resolver."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_path = cache_dir / "ena_nctc3000.msgpack"

    async def discover(self, ena: ENAClient, state: DiscoveryState) -> list[NCTCReferenceStrain]:
        if self.cache_path.is_file():
            try:
                refs = load_msgpack(self.cache_path, list[NCTCReferenceStrain])
            except Exception:
                refs = []
        else:
            refs = await ena.fetch_nctc3000_references(include_untyped=True)
            if refs:
                save_msgpack(self.cache_path, refs)

        for ref in refs:
            org_b = ref.organism.encode("utf-8")
            asm_b = ref.assembly_accession.encode("utf-8")[:32]
            if ref.strain:
                state.bundle.strain_to_assembly.setdefault((org_b, ref.strain.encode("utf-8")), asm_b)
                nospace = ref.strain.replace(" ", "").encode("utf-8")
                state.bundle.strain_to_assembly.setdefault((org_b, nospace), asm_b)
            if ref.serotype:
                for _, sero_tok_b in extract_multi_antigen_tokens(ref.organism, ref.serotype):
                    state.bundle.strain_to_assembly.setdefault((org_b, sero_tok_b), asm_b)
        return refs


class StrainInfoTier:
    """DSMZ StrainInfo v2 culture-collection equivalence clique & literature PMID bridge."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_path = cache_dir / "straininfo_passports.msgpack"

    async def discover(
        self, raw_records: list[dict[str, Any]], si_client: StrainInfoClient, state: DiscoveryState
    ) -> dict[bytes, list[StrainPassport]]:
        if self.cache_path.is_file():
            try:
                cached_map = load_msgpack(self.cache_path, dict[str, list[StrainPassport]])
                state.passports_by_org = {k.encode("utf-8"): v for k, v in cached_map.items()}
            except Exception:
                state.passports_by_org = {}

        species_needed = sorted(
            {
                r["organism"].decode("utf-8", errors="replace")
                for r in raw_records
                if b" " in r["organism"]
                and not r["organism"].endswith(b" sp.")
                and r["organism"] not in state.passports_by_org
            }
        )[:40]

        for sp_name in species_needed:
            passports = await si_client.fetch_taxon_passports(sp_name)
            state.passports_by_org[sp_name.encode("utf-8")] = passports

        if species_needed and state.passports_by_org:
            save_msgpack(
                self.cache_path,
                {k.decode("utf-8", errors="replace"): v for k, v in state.passports_by_org.items()},
            )

        for org_b, plist in state.passports_by_org.items():
            for p in plist:
                # Find any assembly already known for any synonym in this clique
                matched_asm = b""
                for g_acc in p.genome_accessions:
                    matched_asm = g_acc.encode("utf-8")[:32]
                    break
                if not matched_asm:
                    for des in p.designations:
                        des_b = des.encode("utf-8")
                        if (hit_asm := state.bundle.strain_to_assembly.get((org_b, des_b))) is not None:
                            matched_asm = hit_asm
                            break
                if matched_asm:
                    for des in p.designations:
                        state.bundle.strain_to_assembly.setdefault((org_b, des.encode("utf-8")), matched_asm)
                for s_acc in p.sequence_accessions:
                    if len(s_acc) >= 6 and s_acc.split(".")[0] not in state.base_acc_in_cache:
                        for des in p.designations:
                            state.bundle.strain_to_assembly.setdefault(
                                (org_b, des.encode("utf-8")), s_acc.encode("utf-8")[:32]
                            )
        return state.passports_by_org


class ExplicitAccessionTier:
    """Tier 1: Explicit GenBank/RefSeq nucleotide or assembly accession hints."""

    async def discover(self, raw_records: list[dict[str, Any]], runner: NCBILocusRunner, state: DiscoveryState) -> None:
        explicit_nuccore = {
            acc_s
            for r in raw_records
            if (acc_b := r.get("accession_hint", b""))
            and len(acc_s := acc_b.decode("utf-8", errors="replace").strip()) >= 6
            and not acc_s.startswith(("GCF_", "GCA_", "NC_", "NZ_"))
        }
        uncached = [acc for acc in sorted(explicit_nuccore)[:300] if acc.split(".")[0] not in state.base_acc_in_cache]
        for i in range(0, len(uncached), 35):
            clause = " OR ".join(f"{acc}[ACCN]" for acc in uncached[i : i + 35])
            state.tier1_uids.update(
                await runner.ncbi.esearch("nuccore", f"({clause}) AND 2000:85000[SLEN]", retmax=100)
            )


class PubMedElinkTier:
    """Tier 1.5 (token-matched PubMed->nuccore) and Tier 4 (untagged PubMed paper BGC pool)."""

    async def discover(self, raw_records: list[dict[str, Any]], runner: NCBILocusRunner, state: DiscoveryState) -> None:
        pmid_weights: Counter[int] = Counter(
            p_val
            for r in raw_records
            if (p_val := int(r.get("pmid", 0))) > 0
            and not (
                r.get("accession_hint", b"")
                and r["accession_hint"].decode("utf-8", errors="replace").split(".")[0] in state.base_acc_in_cache
            )
        )
        top_pmids = [p for p, _ in pmid_weights.most_common(600)]
        elink_map = await runner.ncbi.elink_pubmed_to_nuccore(top_pmids, batch_size=60)
        for p, links in elink_map.items():
            if 1 <= len(links) <= 45:
                state.pmid_to_nuccore_uids[p] = links[:20]

        elink_uids = sorted(state.tier1_uids | {u for u_list in state.pmid_to_nuccore_uids.values() for u in u_list})
        for u, s_obj in (
            await runner.ncbi.fetch_nuccore_summaries(elink_uids, slen_range=(2500, 65000), chunk_size=200)
        ).items():
            state.uid_to_summary[u] = s_obj
            state.uid_to_accver[u] = s_obj.accession_version

        for u in state.tier1_uids:
            if u in state.uid_to_summary:
                state.uid_priority[u] += 50
        for _, u_list in sorted(state.pmid_to_nuccore_uids.items()):
            for idx_u, u in enumerate([u for u in u_list if u in state.uid_to_summary][:8]):
                state.uid_priority[u] += 25 - idx_u

    def populate_cache_from_predictions(self, raw_records: list[dict[str, Any]], state: DiscoveryState) -> None:
        for p_val, u_list in state.pmid_to_nuccore_uids.items():
            if pooled := state.pool_predicted_uids([u for u in u_list if u in state.uid_to_predicted][:8]):
                state.bundle.assembly_loci[b"pmid:%d" % p_val] = pooled

        for r in raw_records:
            p_val = int(r.get("pmid", 0))
            pred_u = [
                u
                for u in state.pmid_to_nuccore_uids.get(p_val, ())
                if u in state.uid_to_predicted and u in state.uid_to_summary
            ]
            if p_val <= 0 or not pred_u:
                continue
            if not (
                r_toks := {
                    t.decode("utf-8", errors="replace").strip(): t
                    for t in (r.get("serotype", b""), r.get("strain", b""))
                    if t
                }
            ):
                continue
            tok_index = SerotypeTokenIndex.build(r_toks.keys())
            tok_to_uids: dict[bytes, list[str]] = defaultdict(list)
            for u in pred_u:
                for m_s in tok_index.match(state.uid_to_summary[u]):
                    tok_to_uids[r_toks[m_s]].append(u)
            for tok_b, matched_uids in tok_to_uids.items():
                if pooled := state.pool_predicted_uids(matched_uids):
                    state.bundle.assembly_loci.setdefault(b"pmid:%d:%s" % (p_val, tok_b), pooled)


class TaxonCatalogTier:
    """Tier 3: Taxon-wide `nuccore` BGC title search and landmark sweeps indexed by `SerotypeTokenIndex`."""

    _BGC_TITLE_FILTER = (
        "AND 3000:60000[SLEN] AND ("
        '"capsular"[Title] OR "capsule"[Title] OR "cps"[Title] OR "kps"[Title] '
        'OR "K-antigen"[Title] OR "O-antigen"[Title] OR "lipopolysaccharide"[Title] '
        'OR "lipooligosaccharide"[Title] OR "polysaccharide"[Title] OR "exopolysaccharide"[Title] '
        'OR "teichoic"[Title] OR "outer core"[Title] OR "B-band"[Title] '
        'OR "O antigen"[Title] OR "K antigen"[Title] OR "wb"[Title] OR "waa"[Title] OR "wba"[Title]'
        ') NOT "whole genome shotgun"[Title] NOT "patent"[Filter] NOT "targeted locus study"[Title] '
        'NOT "pyocin"[Title] NOT "16S"[Title] NOT "plasmid"[Title] NOT "phage"[Title]'
    )
    _LANDMARK_SWEEPS: dict[str, str] = {
        "Streptococcus pneumoniae": "CR931632:CR931722[ACCN]",
        "Pseudomonas aeruginosa": "AF498400:AF498420[ACCN]",
        "Klebsiella": "AB924547:AB924624[ACCN]",
        "Escherichia coli": "AB811596:AB812084[ACCN]",
        "Proteus": "KY710685:KY710739[ACCN]",
        "Providencia": "HQ880272:HQ880283[ACCN]",
    }

    async def discover(self, raw_records: list[dict[str, Any]], runner: NCBILocusRunner, state: DiscoveryState) -> None:
        taxon_counts: Counter[str] = Counter()
        taxon_targets: dict[str, set[str]] = defaultdict(set)
        key_weights: Counter[tuple[str, str]] = Counter()

        for r in raw_records:
            if (
                r.get("accession_hint", b"")
                and r["accession_hint"].decode("utf-8", errors="replace").split(".")[0] in state.base_acc_in_cache
            ):
                continue
            org_b, str_b, sero_b = r["organism"], r.get("strain", b""), r.get("serotype", b"")
            if (org_b, str_b) in state.bundle.strain_to_assembly or not (taxon_q := _catalog_taxon(org_b)):
                continue
            taxon_counts[taxon_q] += 1
            if sero_b:
                sero_s = sero_b.decode("utf-8", errors="replace").strip()
                taxon_targets[taxon_q].add(sero_s)
                key_weights[(taxon_q, sero_s)] += 1
                if m_base := re.match(r"^([KO]\d+)[a-z].*$", sero_s):
                    taxon_targets[taxon_q].add(m_base.group(1))
                    key_weights[(taxon_q, m_base.group(1))] += 1
            if (
                str_b
                and len(str_b) >= 3
                and str_b != sero_b
                and not (str_s := str_b.decode("utf-8", errors="replace").strip()).isdigit()
            ):
                taxon_targets[taxon_q].add(str_s)
                key_weights[(taxon_q, str_s)] += 1

        key_to_candidates: dict[tuple[str, str], list[tuple[str, str, bool]]] = defaultdict(list)

        async def _catalog_taxon_uids(taxon_q: str) -> None:
            idlist = await runner.ncbi.esearch(
                "nuccore", f'"{taxon_q}"[Organism] {self._BGC_TITLE_FILTER}', retmax=1000
            )
            if taxon_q in self._LANDMARK_SWEEPS:
                seen = set(idlist)
                for lid in await runner.ncbi.esearch("nuccore", self._LANDMARK_SWEEPS[taxon_q], retmax=200):
                    if lid not in seen:
                        idlist.append(lid)
                        seen.add(lid)
            if not idlist:
                return
            summaries = await runner.ncbi.fetch_nuccore_summaries(idlist, slen_range=None, chunk_size=350)
            tok_index = SerotypeTokenIndex.build(taxon_targets[taxon_q])
            for uid in idlist:
                if (s_obj := summaries.get(uid)) is not None:
                    state.uid_to_accver[uid] = s_obj.accession_version
                    for m_tok in tok_index.match(s_obj):
                        cand_list = key_to_candidates[(taxon_q, m_tok)]
                        if not any(u == uid for u, _, _ in cand_list):
                            cand_list.append((uid, s_obj.accession_version, s_obj.is_cps))

        await asyncio.gather(*[_catalog_taxon_uids(tq) for tq, _ in taxon_counts.most_common()])
        for key_pair, cands in sorted(key_to_candidates.items()):
            merged: list[str] = []
            for u in [u for u, _, is_c in cands if is_c][:2] + [u for u, _, is_c in cands if not is_c][:2]:
                if u not in merged and len(merged) < 3:
                    merged.append(u)
            if merged:
                state.selected_key_uids[key_pair] = merged
                w = key_weights.get(key_pair, 1)
                for idx_u, u in enumerate(merged):
                    state.uid_priority[u] += w * (3 - idx_u)

    def populate_cache_from_predictions(self, raw_records: list[dict[str, Any]], state: DiscoveryState) -> None:
        for r in raw_records:
            org_b = r["organism"]
            if not (taxon_q := _catalog_taxon(org_b)):
                continue
            for tok_b in (r.get("strain", b""), r.get("serotype", b"")):
                if not tok_b or (org_b, tok_b) in state.bundle.strain_to_assembly:
                    continue
                tok_s = tok_b.decode("utf-8", errors="replace").strip()
                uids = state.selected_key_uids.get((taxon_q, tok_s), [])
                if not uids and (m_base := re.match(r"^([KO]\d+)[a-z].*$", tok_s)):
                    uids = state.selected_key_uids.get((taxon_q, m_base.group(1)), [])
                if uids and (pooled := state.pool_predicted_uids(uids)):
                    cache_key = b"nuccore:%s:%s" % (taxon_q.encode("utf-8"), tok_b)
                    state.bundle.assembly_loci.setdefault(cache_key, pooled)
                    state.bundle.strain_to_assembly[(org_b, tok_b)] = cache_key


def build_seroref_records_from_bundle(bundle: LociCacheBundle) -> SeroRefRecords:
    """Compile `SeroRefRecords` from a populated `LociCacheBundle`."""
    records: list[SeroRefRecord] = []
    seen: set[tuple[bytes, bytes, bytes, bytes]] = set()

    for (org_b, tok_b), cache_key in sorted(bundle.strain_to_assembly.items()):
        entry = bundle.get_entry(cache_key, allow_base_acc=True)
        if entry is None or len(entry.loci) == 0:
            continue
        asm_id = entry.assembly_id or cache_key
        for j in range(len(entry.loci)):
            loc = entry.loci[j]
            cat = loc.archetype.category
            if cat == ArchetypeCategory.CAPSULE:
                at = b"CPS"
            elif cat == ArchetypeCategory.EXOPOLYSACCHARIDE:
                at = b"EPS"
            elif loc.archetype == LocusArchetype.TEICHOIC_ACID:
                at = b"WTA"
            else:
                at = b"O-antigen"
            sig = (org_b, at, tok_b, asm_id)
            if sig in seen:
                continue
            seen.add(sig)
            records.append(
                SeroRefRecord(
                    organism=org_b,
                    antigen_type=at,
                    serotype=tok_b,
                    strain=tok_b,
                    strain_synonyms=(tok_b,),
                    assembly_id=asm_id,
                    locus=loc,
                )
            )
    return SeroRefRecords.from_list(records)
