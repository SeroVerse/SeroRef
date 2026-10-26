"""Tests for `seroref.genes` role ontology, `OrfPredictor`, `Genes`, and `GeneRoleClassifier`."""

import numpy as np
from seroverse.core.collections import RaggedArrayContainer, ScalarView
from seroverse.core.genome import GenomeAssembly
from seroverse.core.seq.alphabet import GeneticCode
from seroverse.core.seq.seq import Sequence, Sequences

from seroref.genes.cnn import GeneRoleClassifier
from seroref.genes.models import Gene, Genes
from seroref.genes.orf.predictor import OrfPredictor
from seroref.genes.roles import GeneRole, RoleCategory, RoleSpec, RoleSpecs

EXPECTED_MIN_ORFS: dict[bytes, int] = {
    b"PAO1": 5_200,
    b"SGH10_chromosome": 4_600,
    b"STMC125": 4_400,
}


def test_role_ontology_hierarchy() -> None:
    """`RoleCategory`, `GeneRole`, `RoleSpec`, and `RoleSpecs` must follow the 3-tier ontology design."""
    assert ScalarView in RoleSpec.__mro__
    assert RaggedArrayContainer in RoleSpecs.__mro__

    assert RoleCategory.resolve("Transport") == RoleCategory.TRANSPORT
    assert RoleCategory.resolve(b"PolymerAssembly") == RoleCategory.POLYMER_ASSEMBLY
    assert RoleCategory.TRANSPORT.label == b"Transport"

    specs = GeneRole.specs()
    assert isinstance(specs, RoleSpecs)
    assert len(specs) == len(GeneRole) == 30
    assert isinstance(specs[0], RoleSpec)
    assert specs[0].name == b"FLIPPASE"
    assert b"wzx" in specs[0].aliases
    assert specs.feature_matrix.shape == (len(GeneRole), len(RoleCategory) + 1)
    assert specs.biosynthetic_mask.shape == (len(GeneRole),)
    assert int(specs.biosynthetic_mask.sum()) == 27
    assert len(RoleSpecs.concat([specs[:5], specs[5:10]])) == 10
    assert len(RoleSpecs.empty()) == 0

    assert GeneRole.resolve("wzx") == GeneRole.FLIPPASE
    assert GeneRole.resolve(b"wzy") == GeneRole.POLYMERASE


def test_orf_predictor_on_genome_assembly(
    genome_assembly: GenomeAssembly,
    orf_predictor: OrfPredictor,
) -> None:
    """`OrfPredictor.predict` on each parsed `GenomeAssembly` fixture must return valid `Genes` containers."""
    genes = orf_predictor.predict(genome_assembly)

    assert isinstance(genes, Genes)
    assert len(genes) >= EXPECTED_MIN_ORFS[genome_assembly.id]

    contig_len = int(genome_assembly.lengths[0])
    assert np.all(genes.intervals.starts >= 0)
    assert np.all(genes.intervals.ends > genes.intervals.starts)
    assert np.all(genes.intervals.ends <= contig_len)
    assert np.all((genes.intervals.strands == 1) | (genes.intervals.strands == -1))
    assert np.all(genes.seqs.lengths >= 60)
    assert np.all(genes.seqs.lengths % 3 == 0)

    subset = genes[:10]
    assert isinstance(subset, Genes)
    assert len(subset) == 10
    first_gene = subset[0]
    assert isinstance(first_gene, Gene)
    assert first_gene.interval == subset.intervals[0]
    assert first_gene.seq == subset.seqs[0]


def test_orf_predictor_mid_orf_contig_split_and_frameshifts(
    sgh10_assembly: GenomeAssembly,
) -> None:
    """`OrfPredictor` must flag contig-edge partial ORFs and stitch 1-bp frameshifts / stop codons."""
    assert GeneticCode.from_table(11) is GeneticCode.bacterial
    assert GeneticCode.from_table(b"bacterial") is GeneticCode.bacterial
    cds = Sequence(b"cod", b"ATGAAATAGAAATGA")
    assert bytes(cds.translate(code=GeneticCode.bacterial, to_stop=False).seq) == b"MK*K*"

    predictor = OrfPredictor()
    sub_seq = bytes(sgh10_assembly[0].seq[:30_000])
    intact_genes = predictor.predict(sub_seq)
    fwd_long = np.nonzero((intact_genes.intervals.strands == 1) & (intact_genes.intervals.lengths >= 1200))[0]
    assert len(fwd_long) > 0
    g_idx = int(fwd_long[0])
    g_start = int(intact_genes.intervals.starts[g_idx])
    g_end = int(intact_genes.intervals.ends[g_idx])
    cut_pos = (g_start + g_end) // 2

    split_genes = predictor.predict(Sequences.from_bytes([sub_seq[:cut_pos], sub_seq[cut_pos:]]))
    c0_genes = split_genes[split_genes.contig_indices == 0]
    c1_genes = split_genes[split_genes.contig_indices == 1]
    assert bool(c0_genes[-1].is_partial) is True
    assert bool(c1_genes[0].is_partial) is True

    mid = cut_pos - ((cut_pos - g_start) % 3)
    del_seq = b"%s%s" % (sub_seq[:mid], sub_seq[mid + 1 :])
    mut_genes = predictor.predict(del_seq)
    matches = np.nonzero(
        (np.abs(mut_genes.intervals.starts - g_start) <= 30) & (np.abs(mut_genes.intervals.ends - g_end) <= 30)
    )[0]
    assert len(matches) == 1
    assert bool(mut_genes.has_frameshift[int(matches[0])]) is True


def test_gene_role_classifier_inference(sgh10_assembly: GenomeAssembly, orf_predictor: OrfPredictor) -> None:
    """`GeneRoleClassifier` must produce normalized probability distributions over `GeneRole`."""
    genes = orf_predictor.predict(sgh10_assembly[0].seq[:25_000])
    clf = GeneRoleClassifier.pretrained()
    probs = clf.predict_proba(genes.translations[:16])
    assert probs.shape == (16, clf.num_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
