---
title: Architecture & Algorithms
subtitle: ORF HMM, 1D-CNN, ACP-DP Segmentation & 4-Tier Reference DB
description: Deep dive into SeroRef's 29-state profile-HMM ORF caller, polysaccharide-specific 1D-CNN role classifier, ACP-DP locus discovery, and 4-tier Strain–Assembly–Locus curation pipeline.
author: Tom Stanton
icon: lucide/layers
status: new
comments: true
tags:
  - architecture
  - algorithms
  - soa
  - cnn
  - mcl
  - curation
---

# Architecture & Algorithms

Within the 4-domain **SeroVerse** diamond architecture (`seroverse` $\rightarrow$ (`seroref`, `kbank`) $\rightarrow$ `serogen`), **SeroRef** serves as the genomic antigen engine and Strain–Assembly–Locus reference database. It is organized into three strictly layered modules built on `seroverse.core` and `seroverse.client`:

1. **[`seroref.genes`](reference/genes.md)** — Single-pass 29-state frameshift- and operon-overlap-aware profile-HMM Viterbi ORF prediction ([`OrfPredictor`][seroref.genes.orf.predictor.OrfPredictor], supporting bacterial, viral/prophage, and alternative translation tables), functional gene role ontology ([`GeneRole`][seroref.genes.roles.GeneRole], [`RoleSpec`][seroref.genes.roles.RoleSpec], [`RoleSpecs`][seroref.genes.roles.RoleSpecs]), and polysaccharide-antigen-specific multi-scale 1D-CNN protein role classification ([`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier]).
2. **[`seroref.loci`](reference/loci.md)** — Biosynthetic locus archetype ontologies ([`LocusArchetype`][seroref.loci.archetypes.LocusArchetype], [`ArchetypeCategory`][seroref.loci.archetypes.ArchetypeCategory], [`ArchetypeSpec`][seroref.loci.archetypes.ArchetypeSpec], [`ArchetypeSpecs`][seroref.loci.archetypes.ArchetypeSpecs]), exact All-Contiguous-Partitions Dynamic Programming (ACP-DP) + multi-locus Viterbi segmentation ([`LocusSegmenter`][seroref.loci.segmenter.LocusSegmenter]), MGE insertion bridging/healing, cross-contig boundary stitching ([`LocusPredictor`][seroref.loci.predictor.LocusPredictor]), and de novo database construction & structural variant matching ([`LocusMatcher`][seroref.loci.matcher.LocusMatcher]).
3. **[`seroref.db`](reference/db.md)** — Data-oriented species ontology ([`BacterialSpecies`][seroref.db.species.BacterialSpecies], [`SpeciesSpec`][seroref.db.species.SpeciesSpec], [`SpeciesSpecs`][seroref.db.species.SpeciesSpecs]), deduplicated [`ReferenceStrain`][seroref.db.models.ReferenceStrain] and [`ReferenceStrains`][seroref.db.records.ReferenceStrains] containers, and the relational [`SeroRefRecord`][seroref.db.models.SeroRefRecord] / [`SeroRefRecords`][seroref.db.records.SeroRefRecords] reference database (`src/seroref/data/seroref.npz`).

---

## 1. Single-Pass 29-State Bacterial & Prophage ORF Calling (`seroref.genes.orf`)

[`OrfPredictor`][seroref.genes.orf.predictor.OrfPredictor] combines sliding local GC adaptation with a unified, single-mode 29-state frameshift- and operon-overlap-aware profile-HMM Viterbi decoder (`FragGeneScanRs` port):

1. **Precomputed 64-State Trinucleotide LUT**: `Alphabet.DNA.encode` (`seroverse.core.seq.alphabet`) maps raw ASCII FASTA bytes into 2-bit integers (`0..3`) via a Numba LUT kernel, and builds a 1D array of 6-bit trinucleotide indices (`0..63`) shared across the Viterbi pass and downstream extraction passes.
2. **Sliding 4-kb Prefix-Sum Local GC Adaptation**: Computes a 4-kb local GC bin (`0..43`) at every nucleotide via $O(1)$ prefix sums (`local_gc_window=4_000`) so codon log-odds adapt seamlessly across AT-rich polysaccharide islands and integrated prophages (`34%–44%` GC) embedded inside higher-GC host chromosomes.
3. **29-State Frameshift & Operon-Overlap Profile-HMM**:
   - Models forward and reverse coding phases alongside 12 explicit frameshift indel states (`1-bp` and `2-bp` insertions/deletions) to read through sequencing errors and frameshifted pseudogenes (`IS_FRAMESHIFT`) without splitting the ORF.
   - Models direct `1-bp` (`TGATG`/`TAATG`) and `4-bp` (`ATGA`) translational coupling overlaps via dedicated stop-to-start transitions (`E → M1` and `Er → M1r`), scored by upstream Shine-Dalgarno RBS binding energy and start-codon context.

---

## 2. Polysaccharide-Specific Dual-Head 1D-CNN Gene Classification (`seroref.genes.cnn`)

Why does SeroRef use a domain-specific neural architecture ([`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier]) instead of `BLAST`, `MMseqs2`, or generic profile HMMs?

- **Breaking the Sequence Twilight Zone (`< 20–25%` Identity)**: In bacterial surface polysaccharide loci (`CPS`, `O-antigen`, `WTA`, `EPS`), diversifying selection drives `Wzx` flippases (`FLIPPASE`), `Wzy` polymerases (`POLYMERASE`), `WaaL` ligases (`LIGASE`), and glycosyltransferases (`GLYCOSYLTRANSFERASE`) into extreme sequence divergence where k-mer seeding and alignment bitscores fail, even though catalytic triads (`DxD`, `ED`, `EX7E`), Rossmann `GxGxxG` dinucleotide-binding loops, and transmembrane hydropathy profiles remain strictly conserved.
- **Pure Emergent Multi-Scale 1D-CNN (`[conv_max, conv_avg]`)**:
  1. Learnable amino-acid embeddings (`embed_dim=64`) feed three parallel 1D convolutional branches (`num_filters=192`, `kernel_size=(5, 9, 13)`) with LUT-factored forward and backward Numba kernels (`_conv1d_forward_lut` / `_conv1d_backward_lut`).
  2. Dual global max-pooling (`conv_max`, isolating peak catalytic and viral motif activations regardless of sequence length) and length-normalized mean-pooling (`conv_avg`, capturing global fold and transmembrane composition) are concatenated directly into a pure emergent `384`-dimensional penultimate vector (`feature_dim = 2 * num_filters = 384`).
- **Joint 29-Role Biosynthetic Head & 1-Bit Bacteriophage/Prophage Source Head**:
  1. **29-Role Classification + Open-Set Biological Decoy Rejection (`dense_w_`)**: Classifies proteins across **29 functional [`GeneRole`][seroref.genes.roles.GeneRole]s** (`87.89%` holdout Macro F1 across `7,832` non-redundant `<= 80%` CD-HIT sequences). Training with curated bacterial housekeeping and phage structural/packaging/lysis decoys drives out-of-distribution proteins toward uniform softmax entropy so `predict(confidence_threshold=0.50)` rejects **75.3%** of real biological decoys as open-set `GeneRole.UNKNOWN` (`Role 29`).
  2. **1-Bit Bacterial vs. Bacteriophage/Prophage Origin Head (`source_w_`)**: Trained jointly via binary cross-entropy (`source_loss_weight=0.25`) on `taxonomy_id:2` (`0` = bacterial) vs. `taxonomy_id:10239` (`1` = bacteriophage/prophage), [`GeneRoleClassifier.predict_source`][seroref.genes.cnn.GeneRoleClassifier.predict_source] and [`GeneRoleClassifier.predict_source_proba`][seroref.genes.cnn.GeneRoleClassifier.predict_source_proba] achieve **90.4% holdout accuracy**, distinguishing phage-borne seroconverting enzymes (*gtrABC*, *oac*, tailspike lyases/depolymerases) from core chromosomal biosynthetic machinery.
- **11 Internal Representation Layers (`transform`)**: [`GeneRoleClassifier.transform`][seroref.genes.cnn.GeneRoleClassifier.transform] exposes 11 representations (`penultimate`, `task_weighted`, `task_weighted_conv_max`, `conv_max`, `conv_avg`, `embed_mean`, `logits`, `proba`, and per-kernel slices `conv_k5`, `conv_k9`, `conv_k13`). Unweighted `conv_max` (`87.68%` centroid F1) is used directly by [`Loci.cluster_genes`][seroref.loci.models.Loci.cluster_genes] (`embedding_layer="conv_max"`) for role-gated MCL motif rescue.

---

## 3. De Novo Polysaccharide Locus Discovery & Database Matching (`seroref.loci`)

[`LocusPredictor`][seroref.loci.predictor.LocusPredictor] and [`LocusSegmenter`][seroref.loci.segmenter.LocusSegmenter] discover surface polysaccharide loci *de novo* directly from unannotated genomes — even when disrupted by mobile elements or split across draft contigs:

1. **Candidate Island Detection & MGE Bridging**: [`LocusSegmenter`][seroref.loci.segmenter.LocusSegmenter] identifies genomic islands enriched for polysaccharide biosynthesis, processing, and transport roles while bridging across interspersed mobile genetic elements (`GeneRole.TRANSPOSASE` and `GeneRole.INTEGRASE`) and tolerating atypical `UNKNOWN` gaps inside extended operons.
2. **All-Contiguous-Partitions Dynamic Programming (ACP-DP) & Multi-Locus Viterbi**: Evaluates contiguous sub-intervals against 11 [`LocusArchetype`][seroref.loci.archetypes.LocusArchetype] definitions (`CPS_WZY`, `CPS_ABC`, `CPS_SYNTHASE`, `LPS_WZY`, `LPS_ABC`, `EPS_PEL`, `EPS_PGA`, `EPS_ALGINATE`, `EPS_BCS`, `TEICHOIC_ACID`, `UNKNOWN`) using [`ArchetypeClassifier`][seroref.loci.classifier.ArchetypeClassifier] log-odds scoring, anchor boundary snapping, and forbidden-role penalties, partitioning tandem loci (e.g., adjacent `K` and `O` operons) cleanly.
3. **MGE Excision/Healing & Cross-Contig Boundary Stitching**:
   - Calling `LocusPredictor(heal_mobile_elements=True)` (or [`Locus.heal_mobile_elements()`][seroref.loci.models.Locus.heal_mobile_elements]) excises `TRANSPOSASE`/`INTEGRASE` insertions and reconnects flanking biosynthetic genes in the locus graph.
   - Calling `LocusPredictor(stitch_contig_splits=True)` detects partial biosynthetic modules located near contig boundaries (`edge_margin=5000 bp`), re-orients reverse-complemented contigs, merges split partial ORFs across the break (`GAP_BIN = 14`), and reconstructs the complete locus (`is_fragmented=True`).
4. **De Novo Database Construction & Vectorized Structural Variant Matching ([`LocusMatcher`][seroref.loci.matcher.LocusMatcher])**:
   - Given any collection of de novo-discovered [`Loci`][seroref.loci.models.Loci], [`LocusMatcher`][seroref.loci.matcher.LocusMatcher] clusters their proteins via `seroverse.core.seq.clustering.Clusterer` (`cluster_mode="mcl"` or `"cdhit"`), indexes the centroids with `RandstrobeIndex`, and executes a parallel Numba kernel (`_match_loci_kernel`) that scores gene-cluster Jaccard/Dice similarity, collinear synteny conservation, and amino-acid identity while classifying exact structural differences into composable [`MatchType`][seroref.loci.matcher.MatchType] bitflags (`EXACT`, `STRAND_VARIANT`, `MGE_VARIANT`, `FRAMESHIFT_VARIANT`, `SYNTENY_VARIANT`, `GENE_VARIANT`, `ROLE_VARIANT`, `ARCHETYPE_MISMATCH`, `FRAGMENTED_VARIANT`).

---

## 4. Data-Oriented Species Ontology & Relational Reference Database (`seroref.db`)

[`seroref.db`](reference/db.md) models bacterial species, reference strain isolates, and biosynthetic antigen loci using a strictly data-oriented Structure-of-Arrays (SoA) relational design built on `seroverse.core.collections`:

1. **Species Ontology & Serotype Grammar ([`BacterialSpecies`][seroref.db.species.BacterialSpecies], [`SpeciesSpec`][seroref.db.species.SpeciesSpec], [`SpeciesSpecs`][seroref.db.species.SpeciesSpecs])**: Encodes 51 canonical bacterial species alongside their NCBI Taxonomy IDs, Gram-stain classifications ([`GramStain`][seroref.db.species.GramStain]), supported [`LocusArchetype`][seroref.loci.archetypes.LocusArchetype] bitmasks, declarative [`SerotypeRule`][seroref.db.species.SerotypeRule] regexes, Kauffmann-White serovar mappings, and landmark INSDC BGC accession series.
2. **Deduplicated Reference Strains ([`ReferenceStrain`][seroref.db.models.ReferenceStrain], [`ReferenceStrains`][seroref.db.records.ReferenceStrains])**: Subclasses `RaggedArrayContainer` (`_gather_ragged_kernel`) to store `M` deduplicated reference strain/assembly rows (`species_ids`, `names`, `assembly_ids`, `ncbi_taxids`, `straininfo_ids`, `bacdive_ids`) alongside CSR-packed culture-collection equivalence cliques (`synonym_values`, `offsets`, `lengths`) and PubMed citations (`pmid_values`, `pmid_offsets`, `pmid_lengths`).
3. **Multi-Antigen Locus Binding ([`ReferenceStrain.bind_loci`][seroref.db.models.ReferenceStrain.bind_loci])**: Binds de novo predicted [`Loci`][seroref.loci.models.Loci] on a reference assembly into [`SeroRefRecord`][seroref.db.models.SeroRefRecord] views, splitting multi-antigen formulas (`O1:K1:H7` $\rightarrow$ `O1` `O-antigen` + `K1` `CPS`) across [`ArchetypeCategory`][seroref.loci.archetypes.ArchetypeCategory] groups.
4. **Relational `SeroRefRecords` Container ([`SeroRefRecords`][seroref.db.records.SeroRefRecords])**: Links `N` de novo segmented biosynthetic [`Loci`][seroref.loci.models.Loci] and canonical `serotypes` to the `M` deduplicated [`ReferenceStrains`][seroref.db.records.ReferenceStrains] via foreign-key `strain_indices` (`int32[N]`), persisted atomically in `src/seroref/data/seroref.npz`.
