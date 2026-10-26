---
title: API Reference
subtitle: Public Modules & Classes
description: Complete API reference for SeroRef's ORF and CNN gene engines (seroref.genes), de novo locus discovery (seroref.loci), and 4-tier reference database (seroref.db).
author: Tom Stanton
icon: lucide/book-open
comments: true
tags:
  - api
  - reference
---

# API Reference

The **SeroRef** API is organized into three modules:

- **[`seroref.genes`](genes.md)** — Single-pass bacterial & prophage ORF calling ([`OrfPredictor`][seroref.genes.orf.predictor.OrfPredictor]), functional role ontology ([`GeneRole`][seroref.genes.roles.GeneRole], [`RoleSpec`][seroref.genes.roles.RoleSpec], [`RoleSpecs`][seroref.genes.roles.RoleSpecs]), and multi-scale 1D-CNN protein classification ([`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier]).
- **[`seroref.loci`](loci.md)** — Biosynthetic locus archetypes ([`LocusArchetype`][seroref.loci.archetypes.LocusArchetype], [`ArchetypeCategory`][seroref.loci.archetypes.ArchetypeCategory], [`ArchetypeSpec`][seroref.loci.archetypes.ArchetypeSpec], [`ArchetypeSpecs`][seroref.loci.archetypes.ArchetypeSpecs]), ACP-DP locus segmentation ([`LocusSegmenter`][seroref.loci.segmenter.LocusSegmenter]), high-level locus prediction ([`LocusPredictor`][seroref.loci.predictor.LocusPredictor]), and serotype matching ([`LocusMatcher`][seroref.loci.matcher.LocusMatcher]).
- **[`seroref.db`](db.md)** — Species ontology ([`BacterialSpecies`][seroref.db.species.BacterialSpecies], [`SpeciesSpec`][seroref.db.species.SpeciesSpec], [`SpeciesSpecs`][seroref.db.species.SpeciesSpecs]), deduplicated reference strain isolates ([`ReferenceStrain`][seroref.db.models.ReferenceStrain], [`ReferenceStrains`][seroref.db.records.ReferenceStrains]), and relational reference locus records ([`SeroRefRecord`][seroref.db.models.SeroRefRecord], [`SeroRefRecords`][seroref.db.records.SeroRefRecords]).
