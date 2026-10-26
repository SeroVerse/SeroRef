<div align="center">

# 🧬 SeroRef

**De novo bacterial surface polysaccharide locus discovery, twilight-zone 1D-CNN gene role classification, syntenic pan-locus matching, and the Strain–Assembly–Locus reference database of the SeroVerse suite.**

[![CI](https://img.shields.io/github/actions/workflow/status/SeroVerse/SeroRef/ci.yml?branch=main&label=CI&logo=github)](https://github.com/SeroVerse/SeroRef/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/SeroVerse/SeroRef/docs.yml?branch=main&label=Docs&logo=readthedocs)](https://seroverse.github.io/SeroRef/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-0d9488?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-0d9488.svg)](https://www.gnu.org/licenses/gpl-3.0)

[**Documentation**](https://seroverse.github.io/SeroRef/) · [**Architecture**](https://seroverse.github.io/SeroRef/architecture/) · [**Notebooks**](https://seroverse.github.io/SeroRef/notebooks/) · [**API Reference**](https://seroverse.github.io/SeroRef/reference/) · [**Issues**](https://github.com/SeroVerse/SeroRef/issues)

</div>

---

## Overview

**SeroRef** is the genomic antigen engine and **Strain–Assembly–Locus** reference database in the 4-domain **SeroVerse** diamond architecture:

```text
          seroverse (core SoA, graphs, HTTP/2 clients)
              /                                  \
             v                                    v
  seroref (genes, loci, SeroRefRecords)    kbank (sugars, glycans, GlycanRecords)
             \                                    /
              v                                  v
            serogen (geno-glycomic join & biosynthesis)
```

Built on zero-copy Structure-of-Arrays (`numpy`) containers from `seroverse.core` and SIMD-parallel JIT kernels (`numba`), **SeroRef** transforms raw bacterial or prophage genome assemblies into segmented polysaccharide synthesis loci, orthogroup pan-graphs, and curated `SeroRefRecords` (`src/seroref/data/seroref.npz`) in tens of milliseconds — **with zero external CLI dependencies**.

### Core Capabilities

- ⚡ **Single-Pass 29-State Bacterial & Prophage ORF Calling (`seroref.genes.OrfPredictor`)**: Decodes a 4.66 Mb *E. coli* chromosome in **`~71 ms` (>65 Mb/s)** via a frameshift- and operon-overlap-aware (`TGATG`/`ATGA`) profile-HMM Viterbi decoder with sliding 4-kb local GC adaptation and Shine-Dalgarno RBS scoring.
- 🧠 **Polysaccharide-Specific Multi-Scale 1D-CNN Role Classification (`seroref.genes.GeneRoleClassifier`)**: Classifies variable-length amino acid sequences across **30 canonical `GeneRole`s** (`FLIPPASE`, `POLYMERASE`, `ABC_PERMEASE`, `ABC_ATPASE`, `INITIATING_GT`, `GLYCOSYLTRANSFERASE`, modifiers, and precursor enzymes) plus a 1-bit bacterial vs. prophage `source` head using a compiled Numba AdamW multi-scale 1D-CNN with emergent `[conv_max, conv_avg]` dual pooling.
- 🧩 **De Novo Fragmented & MGE-Tolerant Locus Discovery (`seroref.loci.LocusPredictor` & `LocusSegmenter`)**: Segments `CPS_WZY`, `CPS_ABC`, `CPS_SYNTHASE`, `LPS_WZY`, `LPS_ABC`, `EPS_PEL`, `EPS_PGA`, `EPS_ALGINATE`, `EPS_BCS`, and `TEICHOIC_ACID` gene clusters *de novo* using Archetype-Constrained Exact Partitioning Dynamic Programming (`LocusSegmenter`), bridges `IS` transposase/integrase insertions, and stitches loci split across draft contigs (`stitch_contig_splits=True`).
- 🕸️ **Syntenic Graph Alignment & Pan-Locus Matching (`seroref.loci.LocusMatcher`)**: Bit-packed 32-bit `GeneNode` and `SyntenyEdge` (`IntergenicEdge`) graphs with `.as_pan_graph()` consensus compression, hybrid MCL clustering, and bitflag variant classification (`MatchType`).
- 🗄️ **Four-Tier De Novo Reference Database (`seroref.db.SeroRefRecords`)**: Zero-pickle SoA archive (`src/seroref/data/seroref.npz`) curated across **ENA NCTC3000 (`PRJEB6403`)**, **DSMZ StrainInfo v2 / BacDive v2**, **NCBI Landmark BGC Sweeps**, and **NCBI Accession, PubMed `elink` & Strain Assembly Resolution**, splitting multi-antigen formulas (`O1:K1:H7`) into one `SeroRefRecord` per biological surface polysaccharide locus.

---

## Installation

```bash
uv add seroref
# or with pip:
pip install seroref
```

---

## Quickstart

```python
from seroverse.core.genome import GenomeAssembly
from seroref.db.records import load_seroref_records
from seroref.db.species import BacterialSpecies
from seroref.loci.matcher import LocusMatcher
from seroref.loci.predictor import LocusPredictor

# 1. Load the curated SeroRef reference database (src/seroref/data/seroref.npz)
db = load_seroref_records()
ecoli_cps = db.where_species(BacterialSpecies.ESCHERICHIA_COLI).where_antigen_type(b"CPS")
print(f"Loaded {len(db):,} SeroRefRecords across {len(db.strains):,} strains ({len(ecoli_cps)} E. coli CPS loci)")

# 2. Predict surface polysaccharide loci de novo on a genome assembly
assembly = GenomeAssembly.from_fasta("genome.fasta")
predictor = LocusPredictor(stitch_contig_splits=True)
loci = predictor.predict(assembly)

for locus in loci:
    print(
        f"{locus.archetype.name}: {locus.start}..{locus.end} "
        f"({len(locus.genes)} genes, completeness={locus.completeness:.2f})"
    )

# 3. Match predicted loci against the reference database and inspect structural variant flags
matcher = LocusMatcher(db.loci, cluster_mode="mcl", min_identity=80.0, min_coverage=0.80)
matches = matcher.match(loci)
best = matches[0]
print(f"Matched locus: {best.target_name} (score={best.score:.3f}, flags={best.match_type!r})")
```

---

## Development

```bash
# Install all development, test, lint, typing, and docs groups
uv sync --all-groups

# Run linter, formatter, and static type checker
uv run ruff check --fix && uv run ruff format
uv run ty check src/ tests/

# Run test suite
uv run pytest -n 0

# Run the 4-tier SeroRef curation pipeline & slide deck
uv run python scripts/curate_seroref.py

# Build and serve documentation locally with Zensical
uv run zensical serve
```

---

## License

Released under the [GNU General Public License v3.0 (GPLv3)](LICENSE).
