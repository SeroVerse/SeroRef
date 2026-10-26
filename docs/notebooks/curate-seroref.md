---
title: 4-Tier SeroRef Database Curation
subtitle: ENA NCTC3000, DSMZ StrainInfo v2 & NCBI BGC Resolution
description: Four-tier de novo bacterial surface polysaccharide locus discovery and Strain–Assembly–Locus curation pipeline producing src/seroref/data/seroref.npz.
author: Tom Stanton
icon: lucide/database
status: new
comments: true
tags:
  - benchmarks
  - curation
  - nctc3000
  - straininfo
  - ncbi
---

# 4-Tier De Novo `SeroRef` Reference Database Curation

This presentation documents the de novo discovery, DSMZ StrainInfo/BacDive equivalence resolution, multi-antigen formula splitting ([`BacterialSpecies.parse_serotype`][seroref.db.species.BacterialSpecies.parse_serotype] and [`ReferenceStrain.bind_loci`][seroref.db.models.ReferenceStrain.bind_loci]), and [`LocusPredictor`][seroref.loci.predictor.LocusPredictor] pipeline producing the relational Structure-of-Arrays [`SeroRefRecords`][seroref.db.records.SeroRefRecords] reference archive (`src/seroref/data/seroref.npz`).

## Key Takeaways

- **Species Ontology (`seroref.db.species`)**: [`BacterialSpecies`][seroref.db.species.BacterialSpecies], [`SpeciesSpec`][seroref.db.species.SpeciesSpec], and [`SpeciesSpecs`][seroref.db.species.SpeciesSpecs] encode 51 canonical bacterial species, supported [`LocusArchetype`][seroref.loci.archetypes.LocusArchetype] bitmasks, declarative [`SerotypeRule`][seroref.db.species.SerotypeRule] regexes, Kauffmann-White serovar mappings, and landmark BGC accession ranges.
- **ENA NCTC3000 (`PRJEB6403`) Reference Assemblies**: Harvests PacBio reference assemblies and serovar/strain metadata via `ENAClient`, normalizes historical species synonyms, and splits multi-antigen serotype formulas (`O1:K1:H7` $\rightarrow$ `O1` `O-antigen` + `K1` `CPS`) into one [`SeroRefRecord`][seroref.db.models.SeroRefRecord] per biological antigen locus via [`ReferenceStrain.bind_loci`][seroref.db.models.ReferenceStrain.bind_loci].
- **DSMZ StrainInfo v2 & BacDive v2 Equivalence Cliques**: Resolves `StrainPassport` equivalence cliques (`synonym_values`: `ATCC`, `NCTC`, `DSM`, `CNCTC`, `PCM`, `CCUG`, `CIP`, `JCM`), `straininfo_id`, `bacdive_id`, `ncbi_taxid`, and literature `pmids` into [`ReferenceStrains`][seroref.db.records.ReferenceStrains].
- **Relational `SeroRefRecords` Archive**: Links `N` de novo segmented [`Loci`][seroref.loci.models.Loci] to `M` deduplicated [`ReferenceStrains`][seroref.db.records.ReferenceStrains] via foreign-key `strain_indices` (`int32[N]`).

## Run Locally

```bash
# Execute CLI curation & compile src/seroref/data/seroref.npz
uv run python scripts/curate_seroref.py

# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/curate_seroref.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/curate_seroref.py
```
