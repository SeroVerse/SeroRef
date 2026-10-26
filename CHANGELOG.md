# Release Notes

All notable changes to **`SeroRef`** are documented on this page. This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) specification.

---

## [Unreleased]

### Added

- **Single-Pass ORF & CNN Role Engines (`seroref.genes`)**:
  - 29-state frameshift- and operon-overlap-aware profile-HMM Viterbi `OrfPredictor` with sliding 4-kb GC adaptation and single-pass viral/prophage gap rescue (`seroref.genes.orf.predictor`).
  - 30-class `GeneRole` ontology (`29` functional polysaccharide biosynthesis, transport, modification, regulatory, and MGE roles + open-set `UNKNOWN`) and pure NumPy/Numba multi-scale 1D-CNN `GeneRoleClassifier` with a joint 1-bit bacterial vs. bacteriophage/prophage origin head (`seroref.genes.cnn`).
- **De Novo Polysaccharide Locus Discovery (`seroref.loci`)**:
  - 11-class `LocusArchetype` and `ArchetypeCategory` ontology (`seroref.loci.archetypes`) and `ArchetypeClassifier` / `LocusClassifier` (`seroref.loci.classifier`).
  - Exact All-Contiguous-Partitions Dynamic Programming (`ACP-DP`) + multi-locus Viterbi `LocusSegmenter` (`seroref.loci.segmenter`).
  - `LocusPredictor` with `TRANSPOSASE`/`INTEGRASE` MGE insertion bridging, optional coordinate healing (`heal_mobile_elements`), and cross-contig boundary stitching (`stitch_contig_splits`).
  - `LocusMatcher` for de novo orthogroup database construction and `MatchType` structural variant bitflag classification (`seroref.loci.matcher`).
- **4-Tier De Novo Reference Database (`seroref.db`)**:
  - `SeroRefRecord` and `SeroRefRecords` Structure-of-Arrays container (`src/seroref/data/seroref.npz`) linking organism, reference strain, DSMZ StrainInfo/BacDive equivalence cliques (`strain_synonyms`), NCBI assembly/nuccore accession, serotype/glycan antigen token, and de novo predicted `Locus`.
  - 4-tier async curation pipeline (`seroref.db.curation`: `ENANCTC3000Tier`, `StrainInfoTier`, `TaxonCatalogTier`, and de novo target resolution via `seed_discovery_targets_from_sources`, `ExplicitAccessionTier`, `PubMedElinkTier`, and `StrainAssemblyTier`).
- **Interactive Benchmarks & Documentation**:
  - Reproducible 16:9 Marimo slide decks (`curate_seroref.py`, `benchmark_prophage_orfs.py`, `train_gene_classifier.py`, `benchmark_spneumo_db.py`) and unified SeroVerse Zensical documentation.
