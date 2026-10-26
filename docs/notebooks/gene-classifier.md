---
title: CNN Gene Role Classifier
subtitle: Multi-Scale 1D-CNN & Layer Sweep
description: Polysaccharide-specific multi-scale 1D-CNN + biophysical protein role classifier training, Optuna TPE optimization, and Numba layer representation sweep.
author: Tom Stanton
icon: lucide/cpu
status: new
comments: true
tags:
  - benchmarks
  - cnn
  - gene-roles
  - optuna
---

# Polysaccharide-Specific CNN Gene Role Classifier

This presentation documents the multi-source reference curation, **Optuna TPE hyperparameter optimization**, **Numba-accelerated layer representation sweep**, and held-out validation of [`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier] across **non-redundant (`<= 80%` CD-HIT) bacterial (`taxonomy_id:2`) and bacteriophage/prophage (`taxonomy_id:10239`) proteins** spanning all **29 functional [`GeneRole`][seroref.genes.roles.GeneRole]s**, a joint **1-bit bacterial/phage source head**, and open-set `UNKNOWN` (`Role 29`) biological decoy rejection.

## Why Polysaccharide-Specific CNN Classification Beats Remote Homology

Bacterial surface polysaccharide loci (capsular polysaccharides `CPS`, O-antigens `O-antigen`, wall teichoic acids `WTA`, and exopolysaccharides `EPS`) evolve under intense diversifying selection from bacteriophages and host immunity:

1. **Sequence Divergence Beyond BLAST / MMseqs2 / Profile-HMM Sensitivity**: Essential membrane machinery — such as `Wzx`/`GtrA` flippases (`FLIPPASE`), `Wzy` polymerases (`POLYMERASE`), `WaaL` O-antigen ligases (`LIGASE`), and serotype-defining glycosyltransferases (`GLYCOSYLTRANSFERASE`) — routinely diverge into the sequence "twilight and midnight zones" (`< 20–25%` amino-acid identity) where k-mer seeding and alignment bitscores fail.
2. **Conserved Catalytic Motifs & Emergent Biophysical Topology**: Despite extreme global sequence divergence, these enzymes strictly conserve short local catalytic signatures (e.g., `DxD`, `ED`, and `EX7E` glycosyltransferase catalytic triads, Rossmann-fold `GxGxxG` nucleotide-sugar binding loops, and `LytR-CpsA-Psr` motifs) alongside whole-protein transmembrane hydropathy and fold repeat profiles.
3. **Pure Emergent Dual-Pooled 1D-CNN + Joint 1-Bit Source Head**: [`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier] combines learnable amino-acid embeddings (`embed_dim=64`), parallel multi-scale 1D convolutional branches (`num_filters=192`, `kernel_size=(5, 9, 13)`), and dual global max (`conv_max`, peak local catalytic and viral motifs) + length-normalized mean (`conv_avg`, global fold/transmembrane density) pooling (`feature_dim = 2 * num_filters = 384`), driving both the 29-class `GeneRole` readout (`dense_w_`) and a joint **1-bit genomic source readout** (`source_w_`, `predict_source` / `predict_source_proba`: `0` = bacterial, `1` = bacteriophage/prophage) while keeping [`ArchetypeSpecs`][seroref.loci.archetypes.ArchetypeSpecs] 32-bit `uint32` role bitmasks untouched.
4. **Active-Site Embeddings for Twilight-Zone Clustering Rescue**: Intermediate representations extracted via [`GeneRoleClassifier.transform(seqs, layer="conv_max")`][seroref.genes.cnn.GeneRoleClassifier.transform] (default in [`Loci.cluster_genes`][seroref.loci.models.Loci.cluster_genes]) capture catalytic motif activations and feed directly into `seroverse.core.seq.clustering.Clusterer` (`embedding_weight`, `motif_rescue_threshold`) to rescue divergent intra-role homologs during Markov Clustering (MCL).

<div class="serogen-slide-frame">
  <iframe src="assets/train_gene_classifier.html" title="Polysaccharide-Specific CNN Gene Role Classifier Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Architectural Assessments & Conclusions

### 1. 11-Layer Representation Sweep (`transform` Benchmark on 1,565 Held-Out Proteins)

Across all 11 representations exposed by [`GeneRoleClassifier.transform`][seroref.genes.cnn.GeneRoleClassifier.transform], **unweighted `penultimate`** (`[conv_max, conv_avg]`, `384-D`) and **unweighted `conv_max`** (`192-D`) achieve the highest overall clustering and retrieval scores, outperforming class-separation-weighted `task_weighted`:

| Layer | Dim | 1-NN Cosine Recall | Top-5 Role Purity | Nearest-Centroid Macro F1 | Fisher Ratio | Composite Score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`penultimate` (`[conv_max, conv_avg]`)** | 384 | **84.66%** | 82.84% | **87.69%** | 1.339 | **85.06** |
| **`conv_max` (default in `Loci.cluster_genes`)** | 192 | **84.60%** | 82.75% | **87.68%** | 1.309 | **85.01** |
| `task_weighted` | 384 | 83.26% | 83.14% | 87.39% | 1.044 | 84.60 |
| `task_weighted_conv_max` | 192 | 83.51% | 82.65% | 86.90% | 1.031 | 84.35 |
| `conv_k13` (single-kernel max+avg) | 128 | 81.98% | 79.31% | 86.26% | 1.384 | 82.52 |
| `conv_k9` (single-kernel max+avg) | 128 | 81.34% | 79.08% | 85.47% | 1.374 | 81.96 |
| `logits` | 29 | 82.56% | 83.48% | 74.08% | 1.036 | 80.04 |
| `proba` | 29 | 78.79% | 82.77% | 76.29% | 1.883 | 79.28 |
| `conv_k5` (single-kernel max+avg) | 128 | 78.40% | 76.23% | 82.61% | 1.430 | 79.08 |
| `conv_avg` | 192 | 70.99% | 67.39% | 52.81% | 0.631 | 63.73 |
| `embed_mean` | 64 | 61.98% | 55.64% | 34.43% | 0.253 | 50.68 |

### 2. Dual-Head Weight-Norm Division of Labor (`dense_w_` vs. `source_w_`)

Inspecting the Frobenius squared weight norms (`||W||_F²`) of the trained linear readouts reveals how the shared `[conv_max, conv_avg]` representation serves two distinct biological tasks simultaneously:

| Component / Receptive Field | 29-Role Biosynthetic Head (`dense_w_`) | 1-Bit Phage Source Head (`source_w_`) | Biological Interpretation |
| :--- | ---: | ---: | :--- |
| **`conv_max` (`0:192`)** | **65.75%** | **82.23%** | Peak local catalytic (`DxD`, `EX7E`, Walker A/B) & viral signature motifs |
| **`conv_avg` (`192:384`)** | **34.25%** | **17.77%** | Whole-protein fold & transmembrane helix density (`Wzx`/`Wzy`, β-barrels) |
| **Kernel `k = 5 aa`** | 24.03% | 21.89% | Short active-site turns & catalytic triads |
| **Kernel `k = 9 aa`** | 34.35% | 34.85% | Medium Rossmann loops & HTH DNA-binding helices |
| **Kernel `k = 13 aa`** | **41.63%** | **43.26%** | Extended transmembrane spans & viral structural/tailspike motifs |

---

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair --with optuna marimo run scripts/train_gene_classifier.py

# Edit notebook cells
uv run --with marimo --with altair --with optuna marimo edit scripts/train_gene_classifier.py
```
