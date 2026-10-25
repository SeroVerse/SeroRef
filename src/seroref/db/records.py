"""Structure-of-Arrays (SoA) batch container for `SeroRefRecord` reference loci."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from seroverse.core.collections import BatchedContainer, lengths_to_offsets
from seroverse.core.interval import Intervals
from seroverse.core.seq.alphabet import GeneticCode
from seroverse.core.seq.seq import Sequences

from seroref.db.models import SeroRefRecord
from seroref.genes.models import Genes, Indels
from seroref.loci.archetypes import LocusArchetype
from seroref.loci.models import Loci, Locus

DEFAULT_SEROREF_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "seroref.npz"


def loci_to_npz_dict(loci: Loci, prefix: str = "locus_") -> dict[str, Any]:
    """Serialize a `Loci` SoA batch (and its embedded `Genes`) into a flat `.npz`-ready dictionary."""
    g = loci.genes
    return {
        f"{prefix}node_types": loci.node_types,
        f"{prefix}node_offsets": loci.node_offsets,
        f"{prefix}edge_sources": loci.edge_sources,
        f"{prefix}edge_targets": loci.edge_targets,
        f"{prefix}edge_types": loci.edge_types,
        f"{prefix}edge_offsets": loci.edge_offsets,
        f"{prefix}starts": loci.intervals.starts,
        f"{prefix}ends": loci.intervals.ends,
        f"{prefix}strands": loci.intervals.strands,
        f"{prefix}contig_indices": loci.contig_indices,
        f"{prefix}archetypes": loci.archetypes,
        f"{prefix}completeness": loci.completeness,
        f"{prefix}scores": loci.scores,
        f"{prefix}is_tandem_split": loci.is_tandem_split,
        f"{prefix}names": np.char.encode(loci.names, "utf-8") if len(loci.names) > 0 else np.empty(0, dtype="S1"),
        f"{prefix}gene_starts": g.intervals.starts,
        f"{prefix}gene_ends": g.intervals.ends,
        f"{prefix}gene_strands": g.intervals.strands,
        f"{prefix}gene_contig_indices": g.contig_indices,
        f"{prefix}gene_is_partial": g.is_partial,
        f"{prefix}gene_genetic_code_id": np.array([g.genetic_code.table_id], dtype=np.int32),
        f"{prefix}gene_seq_ids": g.seqs.ids,
        f"{prefix}gene_seq_bytes": g.seqs.seqs,
        f"{prefix}gene_seq_offsets": g.seqs.offsets,
        f"{prefix}gene_seq_lengths": g.seqs.lengths,
        f"{prefix}gene_indel_positions": g.indels.positions,
        f"{prefix}gene_indel_kinds": g.indels.kinds,
        f"{prefix}gene_indel_offsets": g.indels.offsets,
        f"{prefix}gene_indel_lengths": g.indels.lengths,
    }


def loci_from_npz_dict(data: Any, prefix: str = "locus_") -> Loci:
    """Deserialize a `Loci` SoA batch from a flat `.npz` archive dictionary."""
    files = set(data.files) if hasattr(data, "files") else set(data.keys())
    g_starts = data[f"{prefix}gene_starts"]
    n_genes = len(g_starts)
    gene_intervals = Intervals(
        starts=g_starts,
        ends=data[f"{prefix}gene_ends"],
        strands=data[f"{prefix}gene_strands"],
    )
    seq_lengths = data[f"{prefix}gene_seq_lengths"]
    gene_seqs = (
        Sequences(
            ids=data[f"{prefix}gene_seq_ids"],
            seqs=data[f"{prefix}gene_seq_bytes"],
            offsets=data[f"{prefix}gene_seq_offsets"],
            lengths=seq_lengths,
        )
        if len(seq_lengths) > 0
        else Sequences.empty()
    )
    gene_indels = Indels(
        positions=data[f"{prefix}gene_indel_positions"],
        kinds=data[f"{prefix}gene_indel_kinds"],
        offsets=data[f"{prefix}gene_indel_offsets"],
        lengths=data[f"{prefix}gene_indel_lengths"],
    )
    is_partial = (
        data[f"{prefix}gene_is_partial"].astype(np.bool_)
        if f"{prefix}gene_is_partial" in files
        else np.zeros(n_genes, dtype=np.bool_)
    )
    gc_id = int(data[f"{prefix}gene_genetic_code_id"][0]) if f"{prefix}gene_genetic_code_id" in files else 11
    genes = Genes(
        intervals=gene_intervals,
        seqs=gene_seqs,
        contig_indices=data[f"{prefix}gene_contig_indices"],
        indels=gene_indels,
        is_partial=is_partial,
        genetic_code=GeneticCode.from_table(gc_id),
    )
    node_offsets = data[f"{prefix}node_offsets"]
    n_loci = max(0, len(node_offsets) - 1)
    names_raw = data[f"{prefix}names"] if f"{prefix}names" in files else np.full(n_loci, "", dtype="U1")
    if names_raw.dtype.kind == "S":
        names_arr = np.char.decode(names_raw, "utf-8", errors="replace")
    else:
        names_arr = names_raw
    return Loci(
        node_types=data[f"{prefix}node_types"],
        node_offsets=node_offsets,
        edge_sources=data[f"{prefix}edge_sources"],
        edge_targets=data[f"{prefix}edge_targets"],
        edge_types=data[f"{prefix}edge_types"],
        edge_offsets=data[f"{prefix}edge_offsets"],
        genes=genes,
        intervals=Intervals(
            starts=data[f"{prefix}starts"],
            ends=data[f"{prefix}ends"],
            strands=data[f"{prefix}strands"],
        ),
        contig_indices=data[f"{prefix}contig_indices"],
        archetypes=data[f"{prefix}archetypes"],
        completeness=data[f"{prefix}completeness"],
        scores=data[f"{prefix}scores"],
        is_tandem_split=data[f"{prefix}is_tandem_split"],
        names=names_arr,
    )


def _slice_ragged_1d(
    values: np.ndarray[tuple[int], Any],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    row_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[
    np.ndarray[tuple[int], Any], np.ndarray[tuple[int], np.dtype[np.int32]], np.ndarray[tuple[int], np.dtype[np.int32]]
]:
    new_lengths = lengths[row_indices].astype(np.int32, copy=False)
    new_offsets = lengths_to_offsets(new_lengths)
    tot = int(new_offsets[-1] + new_lengths[-1]) if len(new_lengths) > 0 else 0
    if tot == 0:
        return values[:0], new_offsets, new_lengths
    gather_idx = np.empty(tot, dtype=np.int32)
    cur = 0
    for r in row_indices:
        off = int(offsets[int(r)])
        ln = int(lengths[int(r)])
        if ln > 0:
            gather_idx[cur : cur + ln] = np.arange(off, off + ln, dtype=np.int32)
            cur += ln
    return values[gather_idx], new_offsets, new_lengths


@dataclass(frozen=True, slots=True)
class SeroRefRecords(BatchedContainer[SeroRefRecord, "SeroRefRecords"]):
    """Structure-of-Arrays (SoA) batch container for curated `SeroRefRecord` reference antigen loci."""

    organisms: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    antigen_types: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    serotypes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    strains: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    assembly_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    ncbi_taxids: np.ndarray[tuple[int], np.dtype[np.int32]]
    straininfo_ids: np.ndarray[tuple[int], np.dtype[np.int32]]
    bacdive_ids: np.ndarray[tuple[int], np.dtype[np.int32]]
    synonym_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    synonym_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    synonym_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    pmid_values: np.ndarray[tuple[int], np.dtype[np.int64]]
    pmid_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    pmid_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    loci: Loci

    def __len__(self) -> int:
        return len(self.organisms)

    def __iter__(self) -> Iterator[SeroRefRecord]:
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> SeroRefRecord: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> SeroRefRecord | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> SeroRefRecord | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"SeroRefRecords index {item} out of range for batch of length {n}.")
            s_off, s_ln = int(self.synonym_offsets[idx]), int(self.synonym_lengths[idx])
            syns = tuple(bytes(self.synonym_values[k]) for k in range(s_off, s_off + s_ln))
            p_off, p_ln = int(self.pmid_offsets[idx]), int(self.pmid_lengths[idx])
            pms = tuple(int(self.pmid_values[k]) for k in range(p_off, p_off + p_ln))
            loc_obj = self.loci[idx]
            return SeroRefRecord(
                organism=bytes(self.organisms[idx]),
                antigen_type=bytes(self.antigen_types[idx]),
                serotype=bytes(self.serotypes[idx]),
                strain=bytes(self.strains[idx]),
                strain_synonyms=syns,
                assembly_id=bytes(self.assembly_ids[idx]),
                locus=loc_obj if len(loc_obj.nodes) > 0 else None,
                ncbi_taxid=int(self.ncbi_taxids[idx]),
                straininfo_id=int(self.straininfo_ids[idx]),
                bacdive_id=int(self.bacdive_ids[idx]),
                pmids=pms,
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return self.empty()
        syn_v, syn_o, syn_l = _slice_ragged_1d(self.synonym_values, self.synonym_offsets, self.synonym_lengths, indices)
        pm_v, pm_o, pm_l = _slice_ragged_1d(self.pmid_values, self.pmid_offsets, self.pmid_lengths, indices)
        return type(self)(
            organisms=self.organisms[indices],
            antigen_types=self.antigen_types[indices],
            serotypes=self.serotypes[indices],
            strains=self.strains[indices],
            assembly_ids=self.assembly_ids[indices],
            ncbi_taxids=self.ncbi_taxids[indices],
            straininfo_ids=self.straininfo_ids[indices],
            bacdive_ids=self.bacdive_ids[indices],
            synonym_values=syn_v,
            synonym_offsets=syn_o,
            synonym_lengths=syn_l,
            pmid_values=pm_v,
            pmid_offsets=pm_o,
            pmid_lengths=pm_l,
            loci=self.loci[indices],
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(
            organisms=np.empty(0, dtype="S1"),
            antigen_types=np.empty(0, dtype="S1"),
            serotypes=np.empty(0, dtype="S1"),
            strains=np.empty(0, dtype="S1"),
            assembly_ids=np.empty(0, dtype="S1"),
            ncbi_taxids=np.empty(0, dtype=np.int32),
            straininfo_ids=np.empty(0, dtype=np.int32),
            bacdive_ids=np.empty(0, dtype=np.int32),
            synonym_values=np.empty(0, dtype="S1"),
            synonym_offsets=np.empty(0, dtype=np.int32),
            synonym_lengths=np.empty(0, dtype=np.int32),
            pmid_values=np.empty(0, dtype=np.int64),
            pmid_offsets=np.empty(0, dtype=np.int32),
            pmid_lengths=np.empty(0, dtype=np.int32),
            loci=Loci.empty(),
        )

    @classmethod
    def from_list(cls, records: Sequence[SeroRefRecord]) -> Self:
        n = len(records)
        if n == 0:
            return cls.empty()
        empty_locus = Locus(nodes=(), edges=())
        syn_flat: list[bytes] = []
        syn_lens = np.empty(n, dtype=np.int32)
        pm_flat: list[int] = []
        pm_lens = np.empty(n, dtype=np.int32)
        loci_list: list[Locus] = []

        for i, r in enumerate(records):
            syn_lens[i] = len(r.strain_synonyms)
            syn_flat.extend(r.strain_synonyms)
            pm_lens[i] = len(r.pmids)
            pm_flat.extend(r.pmids)
            loci_list.append(r.locus if r.locus is not None else empty_locus)

        return cls(
            organisms=np.array([r.organism for r in records], dtype="S"),
            antigen_types=np.array([r.antigen_type for r in records], dtype="S"),
            serotypes=np.array([r.serotype for r in records], dtype="S"),
            strains=np.array([r.strain for r in records], dtype="S"),
            assembly_ids=np.array([r.assembly_id for r in records], dtype="S"),
            ncbi_taxids=np.array([r.ncbi_taxid for r in records], dtype=np.int32),
            straininfo_ids=np.array([r.straininfo_id for r in records], dtype=np.int32),
            bacdive_ids=np.array([r.bacdive_id for r in records], dtype=np.int32),
            synonym_values=np.array(syn_flat, dtype="S") if syn_flat else np.empty(0, dtype="S1"),
            synonym_offsets=lengths_to_offsets(syn_lens),
            synonym_lengths=syn_lens,
            pmid_values=np.array(pm_flat, dtype=np.int64) if pm_flat else np.empty(0, dtype=np.int64),
            pmid_offsets=lengths_to_offsets(pm_lens),
            pmid_lengths=pm_lens,
            loci=Loci.from_list(loci_list),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        items: list[SeroRefRecord] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)

    def where_organism(self, organism: bytes | str) -> Self:
        org_b = organism.encode("utf-8") if isinstance(organism, str) else organism
        return self[self.organisms == org_b]

    def where_serotype(self, serotype: bytes | str) -> Self:
        sero_b = serotype.encode("utf-8") if isinstance(serotype, str) else serotype
        return self[self.serotypes == sero_b]

    def where_antigen_type(self, antigen_type: bytes | str) -> Self:
        at_b = antigen_type.encode("utf-8") if isinstance(antigen_type, str) else antigen_type
        return self[self.antigen_types == at_b]

    def where_assembly_id(self, assembly_id: bytes | str) -> Self:
        asm_b = assembly_id.encode("utf-8") if isinstance(assembly_id, str) else assembly_id
        return self[self.assembly_ids == asm_b]

    def where_archetype(self, archetype: LocusArchetype | int | str | bytes) -> Self:
        arch = LocusArchetype.from_name(archetype)
        return self[self.loci.archetypes == int(arch)]

    def save(self, path: Path = DEFAULT_SEROREF_DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            organisms=self.organisms,
            antigen_types=self.antigen_types,
            serotypes=self.serotypes,
            strains=self.strains,
            assembly_ids=self.assembly_ids,
            ncbi_taxids=self.ncbi_taxids,
            straininfo_ids=self.straininfo_ids,
            bacdive_ids=self.bacdive_ids,
            synonym_values=self.synonym_values,
            synonym_offsets=self.synonym_offsets,
            synonym_lengths=self.synonym_lengths,
            pmid_values=self.pmid_values,
            pmid_offsets=self.pmid_offsets,
            pmid_lengths=self.pmid_lengths,
            **loci_to_npz_dict(self.loci, prefix="locus_"),
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_SEROREF_DB_PATH) -> Self:
        if not path.exists():
            return cls.empty()
        with np.load(path, allow_pickle=False) as data:
            return cls(
                organisms=data["organisms"],
                antigen_types=data["antigen_types"],
                serotypes=data["serotypes"],
                strains=data["strains"],
                assembly_ids=data["assembly_ids"],
                ncbi_taxids=data["ncbi_taxids"],
                straininfo_ids=data["straininfo_ids"],
                bacdive_ids=data["bacdive_ids"],
                synonym_values=data["synonym_values"],
                synonym_offsets=data["synonym_offsets"],
                synonym_lengths=data["synonym_lengths"],
                pmid_values=data["pmid_values"],
                pmid_offsets=data["pmid_offsets"],
                pmid_lengths=data["pmid_lengths"],
                loci=loci_from_npz_dict(data, prefix="locus_"),
            )


def load_seroref_records(path: Path = DEFAULT_SEROREF_DB_PATH) -> SeroRefRecords:
    """Load curated `SeroRefRecords` from `seroref.npz`."""
    return SeroRefRecords.load(path)
