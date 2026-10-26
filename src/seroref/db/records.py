"""Structure-of-Arrays (SoA) batch containers for `ReferenceStrain` and `SeroRefRecord`."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from seroverse.core.collections import (
    BatchedContainer,
    RaggedArrayContainer,
    _gather_ragged_kernel,
    lengths_to_offsets,
)
from seroverse.core.graph.csr import CSR
from seroverse.core.interval import Intervals
from seroverse.core.seq.alphabet import GeneticCode
from seroverse.core.seq.seq import Sequences

from seroref.db.models import (
    INVARIANT_ARCHETYPE_GLYCAN_NAMES,
    ReferenceStrain,
    SeroRefRecord,
    archetype_to_antigen_type,
)
from seroref.db.species import BacterialSpecies
from seroref.genes.models import Genes, Indels
from seroref.loci.archetypes import LocusArchetype
from seroref.loci.models import Loci, Locus

DEFAULT_SEROREF_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "seroref.npz"

_ARCHETYPE_TO_ANTIGEN_BYTES: np.ndarray[tuple[int], np.dtype[np.bytes_]] = np.array(
    [archetype_to_antigen_type(arch) for arch in LocusArchetype],
    dtype="S12",
)


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
        f"{prefix}names": loci.names if len(loci.names) > 0 else np.empty(0, dtype="S1"),
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
    names_raw = data[f"{prefix}names"] if f"{prefix}names" in files else np.full(n_loci, b"", dtype="S1")
    names_arr = np.char.encode(names_raw, "utf-8") if names_raw.dtype.kind == "U" else names_raw
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
class ReferenceStrains(RaggedArrayContainer[ReferenceStrain, "ReferenceStrains"]):
    """Ragged Structure-of-Arrays (SoA) container for deduplicated [`ReferenceStrain`][seroref.db.models.ReferenceStrain] records.

    Stores per-strain species IDs, primary strain names, INSDC assembly accessions, taxonomy/culture
    collection IDs, and CSR-packed ragged arrays for culture-collection `synonym_values` (`offsets`,
    `lengths`) and literature `pmid_values` (`pmid_offsets`, `pmid_lengths`).
    """

    species_ids: np.ndarray[tuple[int], np.dtype[np.int16]]
    names: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    assembly_ids: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    ncbi_taxids: np.ndarray[tuple[int], np.dtype[np.int32]]
    straininfo_ids: np.ndarray[tuple[int], np.dtype[np.int32]]
    bacdive_ids: np.ndarray[tuple[int], np.dtype[np.int32]]
    synonym_values: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    pmid_values: np.ndarray[tuple[int], np.dtype[np.int64]]
    pmid_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    pmid_lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        return len(self.species_ids)

    def __iter__(self) -> Iterator[ReferenceStrain]:
        for i in range(len(self)):
            yield self[i]

    @property
    def organisms(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """Vectorized 1D array of canonical scientific names resolved from `self.species_ids`."""
        if len(self) == 0:
            return np.empty(0, dtype="S1")
        specs = BacterialSpecies.specs()
        clipped = np.clip(self.species_ids, 0, len(specs) - 1)
        return specs.names[clipped]

    @property
    def synonym_offsets(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Alias for `self.offsets` (starting indices into `self.synonym_values`)."""
        return self.offsets

    @property
    def synonym_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Alias for `self.lengths` (number of synonyms per strain record)."""
        return self.lengths

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> ReferenceStrain: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> ReferenceStrain | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> ReferenceStrain | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            n = len(self)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(f"ReferenceStrains index {item} out of range for batch of length {n}.")
            s_off, s_ln = int(self.offsets[idx]), int(self.lengths[idx])
            syns = tuple(bytes(self.synonym_values[k]) for k in range(s_off, s_off + s_ln))
            p_off, p_ln = int(self.pmid_offsets[idx]), int(self.pmid_lengths[idx])
            pms = tuple(int(self.pmid_values[k]) for k in range(p_off, p_off + p_ln))
            sp_val = int(self.species_ids[idx])
            sp = BacterialSpecies(sp_val) if 0 <= sp_val < len(BacterialSpecies) else BacterialSpecies.UNKNOWN
            return ReferenceStrain(
                species=sp,
                name=bytes(self.names[idx]),
                synonyms=syns,
                assembly_id=bytes(self.assembly_ids[idx]),
                ncbi_taxid=int(self.ncbi_taxids[idx]),
                straininfo_id=int(self.straininfo_ids[idx]),
                bacdive_id=int(self.bacdive_ids[idx]),
                pmids=pms,
            )

        syn_idx_pool = np.arange(len(self.synonym_values), dtype=np.int32)
        indices, new_syn_offsets, new_syn_lengths, (gathered_syn_idx,) = self._slice_ragged(item, syn_idx_pool)
        if len(indices) == 0:
            return self.empty()
        syn_v = self.synonym_values[gathered_syn_idx] if len(gathered_syn_idx) > 0 else np.empty(0, dtype="S1")
        pm_v, pm_o, pm_l = _gather_ragged_secondary(
            self.pmid_values,
            self.pmid_offsets,
            self.pmid_lengths,
            indices,
        )
        return type(self)(
            species_ids=self.species_ids[indices],
            names=self.names[indices],
            assembly_ids=self.assembly_ids[indices],
            ncbi_taxids=self.ncbi_taxids[indices],
            straininfo_ids=self.straininfo_ids[indices],
            bacdive_ids=self.bacdive_ids[indices],
            synonym_values=syn_v,
            offsets=new_syn_offsets,
            lengths=new_syn_lengths,
            pmid_values=pm_v,
            pmid_offsets=pm_o,
            pmid_lengths=pm_l,
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `ReferenceStrains` container."""
        return cls(
            species_ids=np.empty(0, dtype=np.int16),
            names=np.empty(0, dtype="S1"),
            assembly_ids=np.empty(0, dtype="S1"),
            ncbi_taxids=np.empty(0, dtype=np.int32),
            straininfo_ids=np.empty(0, dtype=np.int32),
            bacdive_ids=np.empty(0, dtype=np.int32),
            synonym_values=np.empty(0, dtype="S1"),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
            pmid_values=np.empty(0, dtype=np.int64),
            pmid_offsets=np.empty(0, dtype=np.int32),
            pmid_lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_list(cls, strains: Sequence[ReferenceStrain]) -> Self:
        """Compile a sequence of `ReferenceStrain` scalar views into a `ReferenceStrains` SoA container."""
        m = len(strains)
        if m == 0:
            return cls.empty()
        syn_flat: list[bytes] = []
        syn_lens = np.empty(m, dtype=np.int32)
        pm_flat: list[int] = []
        pm_lens = np.empty(m, dtype=np.int32)

        for i, s in enumerate(strains):
            syn_lens[i] = len(s.synonyms)
            syn_flat.extend(s.synonyms)
            pm_lens[i] = len(s.pmids)
            pm_flat.extend(s.pmids)

        return cls(
            species_ids=np.array([int(s.species) for s in strains], dtype=np.int16),
            names=np.array([s.name for s in strains], dtype="S"),
            assembly_ids=np.array([s.assembly_id for s in strains], dtype="S"),
            ncbi_taxids=np.array([s.effective_taxid for s in strains], dtype=np.int32),
            straininfo_ids=np.array([s.straininfo_id for s in strains], dtype=np.int32),
            bacdive_ids=np.array([s.bacdive_id for s in strains], dtype=np.int32),
            synonym_values=np.array(syn_flat, dtype="S") if syn_flat else np.empty(0, dtype="S1"),
            offsets=lengths_to_offsets(syn_lens),
            lengths=syn_lens,
            pmid_values=np.array(pm_flat, dtype=np.int64) if pm_flat else np.empty(0, dtype=np.int64),
            pmid_offsets=lengths_to_offsets(pm_lens),
            pmid_lengths=pm_lens,
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `ReferenceStrains` containers."""
        items: list[ReferenceStrain] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)

    def where_species(self, species: str | bytes | int | BacterialSpecies) -> Self:
        """Filter strains by canonical `BacterialSpecies`."""
        sp = BacterialSpecies.resolve(species)
        return self[self.species_ids == int(sp)]

    def where_assembly_id(self, assembly_id: bytes | str) -> Self:
        """Filter strains by INSDC `assembly_id`."""
        asm_b = assembly_id.encode("utf-8") if isinstance(assembly_id, str) else assembly_id
        return self[self.assembly_ids == asm_b]

    def _equivalence_edges(
        self,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int32]], np.ndarray[tuple[int], np.dtype[np.int32]]]:
        """Extract undirected equivalence edges `(edge_u, edge_v)` across shared assembly, StrainInfo, BacDive, or synonym keys."""
        m = len(self)
        if m <= 1:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty

        buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for i in range(m):
            sp = int(self.species_ids[i])
            asm = bytes(self.assembly_ids[i])
            if asm:
                buckets[("asm", sp, asm.split(b".")[0].upper())].append(i)
            si = int(self.straininfo_ids[i])
            if si > 0:
                buckets[("si", sp, si)].append(i)
            bd = int(self.bacdive_ids[i])
            if bd > 0:
                buckets[("bd", sp, bd)].append(i)
            name_b = bytes(self.names[i])
            if name_b and not asm:
                norm_n = re.sub(rb"[\s\-_/]+", b"", name_b.upper())
                if norm_n:
                    buckets[("syn", sp, norm_n)].append(i)
            s_off, s_ln = int(self.offsets[i]), int(self.lengths[i])
            for k in range(s_off, s_off + s_ln):
                syn_b = bytes(self.synonym_values[k])
                if syn_b and not asm:
                    norm_s = re.sub(rb"[\s\-_/]+", b"", syn_b.upper())
                    if norm_s:
                        buckets[("syn", sp, norm_s)].append(i)

        edge_u: list[int] = []
        edge_v: list[int] = []
        for members in buckets.values():
            if len(members) > 1:
                root = members[0]
                for other in members[1:]:
                    if other != root:
                        edge_u.append(root)
                        edge_v.append(other)
        if not edge_u:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        return np.array(edge_u, dtype=np.int32), np.array(edge_v, dtype=np.int32)

    def equivalence_csr(self) -> CSR:
        """Build the `(M, M)` symmetric `CSR` adjacency matrix of `ReferenceStrain` equivalence cliques."""
        m = len(self)
        if m == 0:
            return CSR.empty((0, 0), dtype=np.int8)
        u, v = self._equivalence_edges()
        if len(u) == 0:
            return CSR.empty((m, m), dtype=np.int8)
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        return CSR.from_coo(rows, cols, np.ones(len(rows), dtype=np.int8), shape=(m, m))

    def merge_cliques(self) -> tuple[Self, np.ndarray[tuple[int], np.dtype[np.int32]]]:
        """Resolve transitive `ReferenceStrain` equivalence cliques via `CSR.connected_components_from_edges` (`seroverse.core.graph.csr.CSR`).

        Returns:
            tuple[ReferenceStrains, np.ndarray]: `(merged_strains, component_labels)` where `component_labels`
            is a 1D `int32` array of length `M` mapping each input strain index to its merged clique index.
        """
        m = len(self)
        if m == 0:
            return self, np.empty(0, dtype=np.int32)
        u, v = self._equivalence_edges()
        n_comp, labels = CSR.connected_components_from_edges(m, u, v)
        if n_comp == m:
            return self, np.arange(m, dtype=np.int32)
        merged_list: list[ReferenceStrain | None] = [None] * n_comp
        for i in range(m):
            c = int(labels[i])
            sr = self[i]
            cur = merged_list[c]
            merged_list[c] = sr if cur is None else cur.merge(sr)
        return self.from_list([s for s in merged_list if s is not None]), labels

    def to_npz_dict(self, prefix: str = "strain_") -> dict[str, Any]:
        """Serialize this `ReferenceStrains` container into a flat `.npz`-ready dictionary."""
        return {
            f"{prefix}species_ids": self.species_ids,
            f"{prefix}names": self.names,
            f"{prefix}assembly_ids": self.assembly_ids,
            f"{prefix}ncbi_taxids": self.ncbi_taxids,
            f"{prefix}straininfo_ids": self.straininfo_ids,
            f"{prefix}bacdive_ids": self.bacdive_ids,
            f"{prefix}synonym_values": self.synonym_values,
            f"{prefix}synonym_offsets": self.offsets,
            f"{prefix}synonym_lengths": self.lengths,
            f"{prefix}pmid_values": self.pmid_values,
            f"{prefix}pmid_offsets": self.pmid_offsets,
            f"{prefix}pmid_lengths": self.pmid_lengths,
        }

    @classmethod
    def from_npz_dict(cls, data: Any, prefix: str = "strain_") -> Self:
        """Deserialize a `ReferenceStrains` container from a flat `.npz` archive dictionary."""
        return cls(
            species_ids=data[f"{prefix}species_ids"].astype(np.int16, copy=False),
            names=data[f"{prefix}names"],
            assembly_ids=data[f"{prefix}assembly_ids"],
            ncbi_taxids=data[f"{prefix}ncbi_taxids"].astype(np.int32, copy=False),
            straininfo_ids=data[f"{prefix}straininfo_ids"].astype(np.int32, copy=False),
            bacdive_ids=data[f"{prefix}bacdive_ids"].astype(np.int32, copy=False),
            synonym_values=data[f"{prefix}synonym_values"],
            offsets=data[f"{prefix}synonym_offsets"].astype(np.int32, copy=False),
            lengths=data[f"{prefix}synonym_lengths"].astype(np.int32, copy=False),
            pmid_values=data[f"{prefix}pmid_values"].astype(np.int64, copy=False),
            pmid_offsets=data[f"{prefix}pmid_offsets"].astype(np.int32, copy=False),
            pmid_lengths=data[f"{prefix}pmid_lengths"].astype(np.int32, copy=False),
        )


@dataclass(frozen=True, slots=True)
class SeroRefRecords(BatchedContainer[SeroRefRecord, "SeroRefRecords"]):
    """Normalized relational Structure-of-Arrays (SoA) container for [`SeroRefRecord`][seroref.db.models.SeroRefRecord] loci.

    Deduplicates reference isolates and genome assemblies into `self.strains` (`ReferenceStrains`, length `M`)
    and links each of the `N` predicted surface polysaccharide loci (`self.loci`, `self.serotypes`) to its parent
    strain via `self.strain_indices` (`int32[N]`, analogous to `Genes.contig_indices`).
    """

    strains: ReferenceStrains
    strain_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    serotypes: np.ndarray[tuple[int], np.dtype[np.bytes_]]
    loci: Loci

    def __len__(self) -> int:
        return len(self.serotypes)

    def __iter__(self) -> Iterator[SeroRefRecord]:
        for i in range(len(self)):
            yield self[i]

    @property
    def species_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int16]]:
        """1D `int16` array (`(N,)`) of `BacterialSpecies` IDs for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype=np.int16)
        return self.strains.species_ids[self.strain_indices]

    @property
    def organisms(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """1D byte-string array (`(N,)`) of canonical species names for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype="S1")
        return self.strains.organisms[self.strain_indices]

    @property
    def strain_names(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """1D byte-string array (`(N,)`) of primary reference strain designations for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype="S1")
        return self.strains.names[self.strain_indices]

    @property
    def assembly_ids(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """1D byte-string array (`(N,)`) of INSDC assembly/nucleotide accessions for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype="S1")
        return self.strains.assembly_ids[self.strain_indices]

    @property
    def ncbi_taxids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array (`(N,)`) of NCBI Taxonomy IDs for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype=np.int32)
        return self.strains.ncbi_taxids[self.strain_indices]

    @property
    def straininfo_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array (`(N,)`) of DSMZ StrainInfo IDs for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype=np.int32)
        return self.strains.straininfo_ids[self.strain_indices]

    @property
    def bacdive_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array (`(N,)`) of DSMZ BacDive IDs for each locus record."""
        if len(self) == 0:
            return np.empty(0, dtype=np.int32)
        return self.strains.bacdive_ids[self.strain_indices]

    @property
    def archetypes(self) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        """1D `int8` array (`(N,)`) of `LocusArchetype` integer IDs across all records in the batch."""
        return self.loci.archetypes

    @property
    def antigen_types(self) -> np.ndarray[tuple[int], np.dtype[np.bytes_]]:
        """1D byte-string array (`(N,)`) of surface antigen classes (`b"CPS"`, `b"O-antigen"`, `b"EPS"`, `b"WTA"`)."""
        if len(self) == 0:
            return np.empty(0, dtype="S1")
        arch_ids = np.clip(self.loci.archetypes, 0, len(_ARCHETYPE_TO_ANTIGEN_BYTES) - 1)
        return _ARCHETYPE_TO_ANTIGEN_BYTES[arch_ids]

    @property
    def strain_locus_csr(self) -> CSR:
        """Sparse `(M, N)` `CSR` (`seroverse.core.graph.csr.CSR`) incidence matrix mapping `ReferenceStrains` to `Loci`."""
        m = len(self.strains)
        n = len(self)
        if m == 0 or n == 0:
            return CSR.empty((m, n), dtype=np.int8)
        locus_idx = np.arange(n, dtype=np.int32)
        return CSR.from_coo(self.strain_indices, locus_idx, np.ones(n, dtype=np.int8), shape=(m, n))

    @property
    def species_locus_csr(self) -> CSR:
        """Sparse `(S, N)` `CSR` (`seroverse.core.graph.csr.CSR`) incidence matrix mapping `BacterialSpecies` to `Loci`."""
        s_count = len(BacterialSpecies)
        n = len(self)
        if n == 0:
            return CSR.empty((s_count, 0), dtype=np.int8)
        locus_idx = np.arange(n, dtype=np.int32)
        return CSR.from_coo(
            self.species_ids.astype(np.int32, copy=False),
            locus_idx,
            np.ones(n, dtype=np.int8),
            shape=(s_count, n),
        )

    @property
    def species_strain_csr(self) -> CSR:
        """Sparse `(S, M)` `CSR` (`seroverse.core.graph.csr.CSR`) incidence matrix mapping `BacterialSpecies` to `ReferenceStrains`."""
        s_count = len(BacterialSpecies)
        m = len(self.strains)
        if m == 0:
            return CSR.empty((s_count, 0), dtype=np.int8)
        strain_idx = np.arange(m, dtype=np.int32)
        return CSR.from_coo(
            self.strains.species_ids.astype(np.int32, copy=False),
            strain_idx,
            np.ones(m, dtype=np.int8),
            shape=(s_count, m),
        )

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
            s_idx = int(self.strain_indices[idx])
            strain_rec = self.strains[s_idx]
            loc_obj = self.loci[idx]
            has_loc = len(loc_obj.nodes) > 0 or loc_obj.archetype != LocusArchetype.UNKNOWN
            return SeroRefRecord(
                strain_record=strain_rec,
                serotype=bytes(self.serotypes[idx]),
                locus=loc_obj if has_loc else None,
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return self.empty()
        sub_strain_idx = self.strain_indices[indices]
        unique_sidx, inverse_sidx = np.unique(sub_strain_idx, return_inverse=True)
        return type(self)(
            strains=self.strains[unique_sidx.astype(np.int32)],
            strain_indices=inverse_sidx.astype(np.int32),
            serotypes=self.serotypes[indices],
            loci=self.loci[indices],
        )

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `SeroRefRecords` container."""
        return cls(
            strains=ReferenceStrains.empty(),
            strain_indices=np.empty(0, dtype=np.int32),
            serotypes=np.empty(0, dtype="S1"),
            loci=Loci.empty(),
        )

    @classmethod
    def from_list(cls, records: Sequence[SeroRefRecord], *, deduplicate: bool = True) -> Self:
        """Compile a sequence of `SeroRefRecord` scalar views into a normalized `SeroRefRecords` container.

        Deduplicates `ReferenceStrain` entries into `self.strains`, merges untyped/typed duplicate
        physical loci when `deduplicate=True`, and synchronizes `Loci.names` with `self.serotypes`.
        """
        if len(records) == 0:
            return cls.empty()

        empty_locus = Locus(nodes=(), edges=())
        normalized_records: list[SeroRefRecord] = []
        for r in records:
            loc = r.locus if r.locus is not None else empty_locus
            sero_b = r.serotype
            if sero_b:
                parsed = r.species.parse_serotype(sero_b, archetype=loc.archetype)
                sero_b = parsed[0][1] if parsed else b""
            if not sero_b and loc.archetype in INVARIANT_ARCHETYPE_GLYCAN_NAMES:
                sero_b = INVARIANT_ARCHETYPE_GLYCAN_NAMES[loc.archetype]
            if sero_b:
                if loc.name != sero_b:
                    loc = loc.with_name(sero_b)
            elif loc.name:
                loc = loc.with_name(b"")
            has_loc = len(loc.nodes) > 0 or loc.archetype != LocusArchetype.UNKNOWN
            normalized_records.append(
                SeroRefRecord(
                    strain_record=r.strain_record,
                    serotype=sero_b,
                    locus=loc if has_loc else None,
                )
            )

        if deduplicate:
            by_key: dict[tuple[Any, ...], SeroRefRecord] = {}
            for r in normalized_records:
                loc = r.locus
                arch = int(loc.archetype) if loc is not None else int(LocusArchetype.UNKNOWN)
                start = int(loc.start) if loc is not None else 0
                end = int(loc.end) if loc is not None else 0
                if r.assembly_id and loc is not None and len(loc.nodes) > 0:
                    key: tuple[Any, ...] = ("asm_loc", int(r.species), r.assembly_id, arch, start, end, r.serotype)
                else:
                    key = ("meta", int(r.species), r.assembly_id, r.strain, arch, r.serotype, start, end)
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = r
                else:
                    by_key[key] = existing.merge(r)

            # Fold untyped (serotype == b"") records into typed records on the exact same physical locus
            typed_phys: dict[tuple[int, bytes, int, int, int], tuple[Any, ...]] = {}
            for k, r in by_key.items():
                if r.serotype and r.assembly_id and r.locus is not None:
                    phys = (int(r.species), r.assembly_id, int(r.locus.archetype), int(r.locus.start), int(r.locus.end))
                    typed_phys[phys] = k

            deduped_list: list[SeroRefRecord] = []
            for k, r in list(by_key.items()):
                if not r.serotype and r.assembly_id and r.locus is not None:
                    phys = (int(r.species), r.assembly_id, int(r.locus.archetype), int(r.locus.start), int(r.locus.end))
                    target_k = typed_phys.get(phys)
                    if target_k is not None and target_k in by_key:
                        by_key[target_k] = by_key[target_k].merge(r)
                        del by_key[k]
            deduped_list = list(by_key.values())
        else:
            deduped_list = normalized_records

        # Build deduplicated ReferenceStrains table via CSR connected components (`merge_cliques`)
        raw_strains = ReferenceStrains.from_list([r.strain_record for r in deduped_list])
        merged_strains, strain_labels = raw_strains.merge_cliques()

        n = len(deduped_list)
        loci_list = [r.locus if r.locus is not None else empty_locus for r in deduped_list]
        serotypes_arr = np.array([r.serotype for r in deduped_list], dtype="S") if n > 0 else np.empty(0, dtype="S1")

        return cls(
            strains=merged_strains,
            strain_indices=strain_labels,
            serotypes=serotypes_arr,
            loci=Loci.from_list(loci_list),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate and normalize multiple `SeroRefRecords` batches."""
        items: list[SeroRefRecord] = []
        for b in batches:
            items.extend(b)
        return cls.from_list(items)

    def where_species(self, species: str | bytes | int | BacterialSpecies) -> Self:
        """Filter records by canonical `BacterialSpecies`."""
        sp = BacterialSpecies.resolve(species)
        return self[self.species_ids == int(sp)]

    def where_organism(self, organism: bytes | str | BacterialSpecies) -> Self:
        """Filter records by organism name or `BacterialSpecies`."""
        if isinstance(organism, BacterialSpecies):
            return self[self.species_ids == int(organism)]
        sp = BacterialSpecies.resolve(organism)
        if sp != BacterialSpecies.UNKNOWN:
            return self[self.species_ids == int(sp)]
        org_b = organism.encode("utf-8") if isinstance(organism, str) else organism
        return self[self.organisms == org_b]

    def where_serotype(self, serotype: bytes | str) -> Self:
        """Filter records by canonical `serotype` byte string."""
        sero_b = serotype.encode("utf-8") if isinstance(serotype, str) else serotype
        return self[self.serotypes == sero_b]

    def where_antigen_type(self, antigen_type: bytes | str) -> Self:
        """Filter records by derived surface polysaccharide `antigen_type` (`CPS`, `O-antigen`, `EPS`, `WTA`)."""
        at_b = antigen_type.encode("utf-8") if isinstance(antigen_type, str) else antigen_type
        return self[self.antigen_types == at_b]

    def where_assembly_id(self, assembly_id: bytes | str) -> Self:
        """Filter records by INSDC genome or locus `assembly_id`."""
        asm_b = assembly_id.encode("utf-8") if isinstance(assembly_id, str) else assembly_id
        return self[self.assembly_ids == asm_b]

    def where_archetype(self, archetype: LocusArchetype | int | str | bytes) -> Self:
        """Filter records by `LocusArchetype`."""
        arch = LocusArchetype.from_name(archetype)
        return self[self.loci.archetypes == int(arch)]

    def save(self, path: Path = DEFAULT_SEROREF_DB_PATH) -> Path:
        """Atomically serialize this `SeroRefRecords` container to a compressed `.npz` archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            strain_indices=self.strain_indices,
            serotypes=self.serotypes,
            **self.strains.to_npz_dict(prefix="strain_"),
            **loci_to_npz_dict(self.loci, prefix="locus_"),
        )
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_SEROREF_DB_PATH) -> Self:
        """Load a `SeroRefRecords` container from `.npz` (supporting both relational and legacy flat schemas)."""
        if not path.exists():
            return cls.empty()
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            loci = loci_from_npz_dict(data, prefix="locus_")
            if "strain_indices" in files and "strain_species_ids" in files:
                return cls(
                    strains=ReferenceStrains.from_npz_dict(data, prefix="strain_"),
                    strain_indices=data["strain_indices"].astype(np.int32, copy=False),
                    serotypes=data["serotypes"],
                    loci=loci,
                )

            # Transparent migration path from legacy flat schema
            organisms = data["organisms"]
            serotypes = data["serotypes"]
            strains = data["strains"]
            assembly_ids = data["assembly_ids"]
            ncbi_taxids = data["ncbi_taxids"]
            straininfo_ids = data["straininfo_ids"]
            bacdive_ids = data["bacdive_ids"]
            syn_v = data["synonym_values"]
            syn_o = data["synonym_offsets"]
            syn_l = data["synonym_lengths"]
            pm_v = data["pmid_values"]
            pm_o = data["pmid_offsets"]
            pm_l = data["pmid_lengths"]

            migrated: list[SeroRefRecord] = []
            for i in range(len(organisms)):
                s_off, s_ln = int(syn_o[i]), int(syn_l[i])
                syns = tuple(bytes(syn_v[k]) for k in range(s_off, s_off + s_ln))
                p_off, p_ln = int(pm_o[i]), int(pm_l[i])
                pms = tuple(int(pm_v[k]) for k in range(p_off, p_off + p_ln))
                loc_obj = loci[i]
                has_loc = len(loc_obj.nodes) > 0 or loc_obj.archetype != LocusArchetype.UNKNOWN
                migrated.append(
                    SeroRefRecord.from_fields(
                        bytes(organisms[i]),
                        serotype=bytes(serotypes[i]),
                        strain=bytes(strains[i]),
                        strain_synonyms=syns,
                        assembly_id=bytes(assembly_ids[i]),
                        locus=loc_obj if has_loc else None,
                        ncbi_taxid=int(ncbi_taxids[i]),
                        straininfo_id=int(straininfo_ids[i]),
                        bacdive_id=int(bacdive_ids[i]),
                        pmids=pms,
                    )
                )
            return cls.from_list(migrated, deduplicate=True)


def load_seroref_records(path: Path = DEFAULT_SEROREF_DB_PATH) -> SeroRefRecords:
    """Load curated `SeroRefRecords` from `seroref.npz`."""
    return SeroRefRecords.load(path)
