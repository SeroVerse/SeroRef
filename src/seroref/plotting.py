# from collections.abc import Sequence
# from typing import Any
#
# import numpy as np
# from scipy.sparse import csr_array
# from scipy.stats import qmc
# import altair as alt
# from seroverse.core.graph.layout import LayoutEngine
# from seroverse.core.graph.models import GraphBatch, GraphModel
#
# from seroref.genes.roles import GeneRole
# from seroref.loci.models import GeneNode, IntergenicOrientation, Loci, Locus
#
#
# class OKLCh:
#     """
#     Generates a perceptually uniform hex palette from an IntEnum
#     using OKLCh -> OKLab -> Linear sRGB -> sRGB (via NumPy & SciPy).
#     """
#     def __init__(
#         self,
#         n: int,
#         lightness: float = 0.68,
#         chroma: float = 0.14,
#         hue_offset: float = 0.08,
#         use_halton: bool = False
#     ) -> None:
#         self.n = n
#         if use_halton:
#             # Halton quasi-random sequence ensures adjacent enum items are visually distinct
#             sampler = qmc.Halton(d=1, scramble=False)
#             fractions = (sampler.random(n=n).flatten() + hue_offset) % 1.0
#         else:
#             # Evenly spaced around the color wheel
#             fractions = (np.linspace(0, 1, n, endpoint=False) + hue_offset) % 1.0
#
#         h = fractions * 2 * np.pi
#         L = np.full(n, lightness)
#         C = np.full(n, chroma)
#         a = C * np.cos(h)
#         b = C * np.sin(h)
#         oklab = np.column_stack([L, a, b])
#         M1_inv = np.array([
#             [1.0, 0.3963377774, 0.2158037573],
#             [1.0, -0.1055613458, -0.0638541728],
#             [1.0, -0.0894841775, -1.2914855480]
#         ])
#         lms_nonlin = oklab @ M1_inv.T
#         lms = lms_nonlin ** 3
#         M2_inv = np.array([
#             [4.0767416621, -3.3077115913, 0.2309699292],
#             [-1.2684380046, 2.6097574011, -0.3413193965],
#             [-0.0041960863, -0.7034186147, 1.7076147010]
#         ])
#         rgb_lin = np.clip(lms @ M2_inv.T, 0.0, 1.0)
#         srgb = np.where(
#             rgb_lin <= 0.0031308,
#             12.92 * rgb_lin,
#             1.055 * (np.maximum(rgb_lin, 1e-10) ** (1.0 / 2.4)) - 0.055
#         )
#
#         rgb_int = np.round(np.clip(srgb, 0.0, 1.0) * 255).astype(int)
#         self.hex_colors = [f"#{r:02x}{g:02x}{b_val:02x}" for r, g, b_val in rgb_int]
#
#
# class LocusPlotter:
#     """Abstract base class for plotting `GraphModel` and `GraphBatch` objects via `LayoutEngine` and `altair`."""
#     # Directional SVG arrow paths in [-1, 1] x [-1, 1] unit box for Altair mark_point(shape=...)
#     ARROW_FWD_PATH = "M -0.95,-0.48 L 0.28,-0.48 L 0.28,-0.88 L 0.98,0 L 0.28,0.88 L 0.28,0.48 L -0.95,0.48 Z"
#     ARROW_REV_PATH = "M 0.95,-0.48 L -0.28,-0.48 L -0.28,-0.88 L -0.98,0 L -0.28,0.88 L -0.28,0.48 L 0.95,0.48 Z"
#
#     # Custom SNFG 5-pointed star (Pentoses) and 6-sided hexagon (Heptoses) SVG paths
#     STAR5_PATH = (
#         "M 0,-0.95 L 0.22,-0.30 L 0.90,-0.30 L 0.35,0.12 L 0.56,0.80 "
#         "L 0,0.38 L -0.56,0.80 L -0.35,0.12 L -0.90,-0.30 L -0.22,-0.30 Z"
#     )
#     HEXAGON_PATH = "M 0,-0.92 L 0.80,-0.46 L 0.80,0.46 L 0,0.92 L -0.80,0.46 L -0.80,-0.46 Z"
#
#     def __init__(self):
#         self.width: int = 760
#         self.height: int = 280
#         self.color_by: str = "role"
#         self.use_force_layout: bool = False
#         self.height: int = 195
#
#     def compute_layout(
#         self,
#         graph: GraphModel | GraphBatch[Any, Any] | csr_array | np.ndarray,
#         *,
#         pos: np.ndarray | None = None,
#         fixed: Sequence[int] | np.ndarray | None = None,
#     ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
#         """Compute 2D coordinates `(N, 2)` for `graph` using `self.layout_engine` (`LayoutEngine`)."""
#         return self.layout_engine.layout(graph, pos=pos, fixed=fixed)
#
#     def _compute_locus_coords(self, locus: Locus) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
#         """Compute 2D gene coordinates via `LayoutEngine` with 5'->3' syntenic initialization and pinned ends."""
#         n = len(locus.nodes)
#         if n == 0:
#             return np.empty((0, 2), dtype=np.float64)
#         if n == 1:
#             return np.zeros((1, 2), dtype=np.float64)
#
#         init_pos = np.zeros((n, 2), dtype=np.float64)
#         if len(locus.genes) == n and len(locus.genes.intervals.starts) == n:
#             mids = 0.5 * (
#                 locus.genes.intervals.starts.astype(np.float64) + locus.genes.intervals.ends.astype(np.float64)
#             )
#             span = float(np.max(mids) - np.min(mids))
#             if span > 1.0:
#                 init_pos[:, 0] = 2.0 * (mids - float(np.min(mids))) / span - 1.0
#             else:
#                 init_pos[:, 0] = np.linspace(-1.0, 1.0, n)
#         else:
#             init_pos[:, 0] = np.linspace(-1.0, 1.0, n)
#
#         if not self.use_force_layout:
#             # Pin all genes on the 5'->3' genomic axis (0 iterations of FR displacement)
#             fixed_all = np.ones(n, dtype=np.bool_)
#             return self.compute_layout(locus, pos=init_pos, fixed=fixed_all)
#
#         # Warm-start LayoutEngine from 5'->3' axis, pinning 5' and 3' flanking anchor genes
#         init_pos[:, 1] = np.where(np.arange(n) % 2 == 0, 0.06, -0.06)
#         init_pos[0, 1] = 0.0
#         init_pos[-1, 1] = 0.0
#         return self.compute_layout(locus, pos=init_pos, fixed=[0, n - 1])
#
#     def extract_elements(
#         self,
#         locus: Locus,
#         *,
#         coords: np.ndarray | None = None,
#         assignments: Sequence[GeneGlycanAssignment] = (),
#         record_label: str = "",
#     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
#         """Extract strand-aware gene arrow nodes and intergenic synteny edges for `locus`."""
#         n = len(locus.nodes)
#         if n == 0:
#             return [], []
#
#         pos = coords if coords is not None else self._compute_locus_coords(locus)
#         assign_by_gene: dict[int, list[GeneGlycanAssignment]] = {}
#         for a in assignments:
#             if a.gene_index >= 0:
#                 assign_by_gene.setdefault(int(a.gene_index), []).append(a)
#
#         has_genes = len(locus.genes) == n
#         if has_genes and len(locus.genes.intervals.strands) == n:
#             strands_arr = locus.genes.intervals.strands
#         else:
#             strands_arr = np.ones(n, dtype=np.int8)
#             if int(locus.strand) < 0:
#                 strands_arr[0] = -1
#             for edge in locus.edges:
#                 u_e, v_e = int(edge.source), int(edge.target)
#                 if 0 <= u_e < n and 0 <= v_e < n and v_e == u_e + 1:
#                     orient = edge.edge_type.orientation
#                     if orient == IntergenicOrientation.CODIRECTIONAL:
#                         strands_arr[v_e] = strands_arr[u_e]
#                     elif orient == IntergenicOrientation.CONVERGENT:
#                         strands_arr[v_e] = -1
#                     elif orient == IntergenicOrientation.DIVERGENT:
#                         strands_arr[v_e] = 1
#
#         node_rows: list[dict[str, Any]] = []
#         for idx, g_node in enumerate(locus.nodes):
#             role_name = g_node.role.name or "UNKNOWN"
#             role_abbrev = g_node.role.abbreviation
#             cid = int(g_node.cluster_id)
#             strand_val = int(strands_arr[idx])
#             if has_genes:
#                 start_bp = int(locus.genes.intervals.starts[idx])
#                 end_bp = int(locus.genes.intervals.ends[idx])
#                 length_bp = max(1, end_bp - start_bp)
#             else:
#                 start_bp, end_bp, length_bp = idx * 1000, (idx + 1) * 1000, 1000
#
#             g_assigns = assign_by_gene.get(idx, [])
#             if g_assigns:
#                 rxn_str = ", ".join(a.component_label for a in g_assigns)
#                 ev_str = g_assigns[0].evidence.name or "ALIGNED"
#                 score_val = float(max(a.score for a in g_assigns))
#                 gene_slot = f"G{idx}"
#             else:
#                 rxn_str = "—"
#                 ev_str = "UNASSIGNED"
#                 score_val = 0.0
#                 gene_slot = f"U{idx}"
#
#             node_rows.append(
#                 {
#                     "record_label": record_label,
#                     "gene_index": idx,
#                     "gene_slot": gene_slot,
#                     "x": round(float(pos[idx, 0]), 4),
#                     "y": round(float(pos[idx, 1]), 4),
#                     "role": role_name,
#                     "role_abbrev": role_abbrev,
#                     "cluster_id": f"CID-{cid:04d}" if cid > 0 else "Unclustered",
#                     "strand": "+1 (→)" if strand_val >= 0 else "-1 (←)",
#                     "strand_dir": "forward" if strand_val >= 0 else "reverse",
#                     "arrow_label": f"{role_abbrev} →" if strand_val >= 0 else f"← {role_abbrev}",
#                     "coords_bp": f"{start_bp:,}..{end_bp:,} bp ({length_bp:,} bp)",
#                     "length_bp": length_bp,
#                     "assigned_reaction": rxn_str,
#                     "evidence": ev_str,
#                     "alignment_score": round(score_val, 3),
#                     "is_assigned": bool(g_assigns),
#                 }
#             )
#
#         edge_rows: list[dict[str, Any]] = []
#         for edge in locus.edges:
#             u, v = int(edge.source), int(edge.target)
#             if not (0 <= u < n and 0 <= v < n):
#                 continue
#             orient = edge.edge_type.orientation
#             orient_str = {
#                 IntergenicOrientation.CODIRECTIONAL: "Codirectional (→ →)",
#                 IntergenicOrientation.CONVERGENT: "Convergent (→ ←)",
#                 IntergenicOrientation.DIVERGENT: "Divergent (← →)",
#             }.get(orient, "Syntenic")
#             edge_rows.append(
#                 {
#                     "record_label": record_label,
#                     "x": round(float(pos[u, 0]), 4),
#                     "y": round(float(pos[u, 1]), 4),
#                     "x2": round(float(pos[v, 0]), 4),
#                     "y2": round(float(pos[v, 1]), 4),
#                     "orientation": orient_str,
#                 }
#             )
#         return node_rows, edge_rows
#
#     def plot(
#         self,
#         locus: Locus,
#         *,
#         assignments: Sequence[GeneGlycanAssignment] = (),
#         title: str | None = None,
#         gene_selection: Any = None,
#     ) -> alt.Chart | alt.LayerChart:
#         """Plot a single `Locus` with strand-aware directional arrows coloured by role or cluster."""
#         node_rows, edge_rows = self.extract_elements(locus, assignments=assignments)
#         if not node_rows:
#             return alt.Chart(alt.Data(values=[])).mark_point()
#
#         x_scale = alt.Scale(domain=[-1.15, 1.15])
#         y_dom = [-0.45, 0.45] if not self.use_force_layout else [-1.20, 1.20]
#         y_scale = alt.Scale(domain=y_dom)
#
#         backbone_rule = (
#             alt.Chart(alt.Data(values=edge_rows))
#             .mark_rule(color="#64748b", strokeWidth=2.2, opacity=0.45)
#             .encode(
#                 x=alt.X("x:Q", scale=x_scale, axis=None),
#                 y=alt.Y("y:Q", scale=y_scale, axis=None),
#                 x2="x2:Q",
#                 y2="y2:Q",
#                 tooltip=[alt.Tooltip("orientation:N", title="Intergenic Synteny")],
#             )
#         )
#
#         if self.color_by == "role":
#             present_roles = list(dict.fromkeys(str(r["role"]) for r in node_rows))
#             present_colors = [GENE_ROLE_COLORS.get(r, "#94a3b8") for r in present_roles]
#             color_enc = alt.Color(
#                 "role:N",
#                 scale=alt.Scale(domain=present_roles, range=present_colors),
#                 title="Gene Role",
#             )
#         else:
#             color_enc = alt.Color(
#                 "cluster_id:N",
#                 scale=alt.Scale(scheme="category20"),
#                 title="Homology Cluster",
#             )
#
#         opacity_enc = (
#             alt.condition(gene_selection, alt.value(1.0), alt.value(0.28))
#             if gene_selection is not None
#             else alt.value(0.95)
#         )
#
#         arrows_layer = (
#             alt.Chart(alt.Data(values=node_rows))
#             .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.4)
#             .encode(
#                 x=alt.X("x:Q", scale=x_scale, axis=None),
#                 y=alt.Y("y:Q", scale=y_scale, axis=None),
#                 shape=alt.Shape(
#                     "strand_dir:N",
#                     scale=alt.Scale(
#                         domain=["forward", "reverse"],
#                         range=[ARROW_FWD_PATH, ARROW_REV_PATH],
#                     ),
#                     legend=None,
#                 ),
#                 size=alt.Size(
#                     "length_bp:Q",
#                     scale=alt.Scale(range=[420, 920]),
#                     legend=None,
#                 ),
#                 color=color_enc,
#                 opacity=opacity_enc,
#                 tooltip=[
#                     alt.Tooltip("gene_index:Q", title="ORF Index"),
#                     alt.Tooltip("role_abbrev:N", title="Gene Role"),
#                     alt.Tooltip("role:N", title="Role Class"),
#                     alt.Tooltip("cluster_id:N", title="Homology Cluster"),
#                     alt.Tooltip("strand:N", title="Strand"),
#                     alt.Tooltip("coords_bp:N", title="Interval"),
#                 ],
#             )
#         )
#
#         if gene_selection is not None:
#             arrows_layer = arrows_layer.add_params(gene_selection)
#
#         labels_layer = (
#             alt.Chart(alt.Data(values=node_rows))
#             .mark_text(dy=-22, fontSize=10, fontWeight="bold", color="#334155")
#             .encode(
#                 x=alt.X("x:Q", scale=x_scale, axis=None),
#                 y=alt.Y("y:Q", scale=y_scale, axis=None),
#                 text="arrow_label:N",
#             )
#         )
#
#         rxn_labels_layer = (
#             alt.Chart(alt.Data(values=[r for r in node_rows if r["is_assigned"]]))
#             .mark_text(dy=24, fontSize=9, color="#059669", fontWeight="bold")
#             .encode(
#                 x=alt.X("x:Q", scale=x_scale, axis=None),
#                 y=alt.Y("y:Q", scale=y_scale, axis=None),
#                 text="assigned_reaction:N",
#             )
#         )
#
#         chart_title = title or (
#             f"Biosynthetic Gene Cluster ({locus.archetype.name}, {len(node_rows)} ORFs, "
#             f"Completeness={locus.completeness:.2f})"
#         )
#         return (backbone_rule + arrows_layer + labels_layer + rxn_labels_layer).properties(
#             width=self.width, height=self.height, title=chart_title
#         )
#
#     def plot_pan_graph(
#         self,
#         loci: Loci,
#         *,
#         min_node_count: int = 3,
#         min_edge_support: int = 2,
#         title: str = "Pan-Locus Synteny Network (SeroGen LayoutEngine Fruchterman-Reingold)",
#     ) -> alt.Chart | alt.LayerChart:
#         """Plot a 2D Pan-Locus homology-cluster synteny network across a `Loci` batch via `LayoutEngine`."""
#         pan_adj, pan_nodes, pan_counts = loci.as_pan_graph(node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"))
#         pan_csr = pan_adj.tocsr().copy()
#         pan_csr.setdiag(0)
#         pan_csr.data[pan_csr.data < min_edge_support] = 0
#         pan_csr.eliminate_zeros()
#
#         has_edge = (np.diff(pan_csr.indptr) > 0) | (np.diff(pan_csr.tocsc().indptr) > 0)
#         keep_mask = (pan_counts >= min_node_count) & (GeneNode.CLUSTER_ID.extract_array(pan_nodes) > 0) & has_edge
#         keep_indices = np.nonzero(keep_mask)[0]
#         if len(keep_indices) == 0:
#             return alt.Chart(alt.Data(values=[])).mark_point()
#
#         sub_csr = pan_csr[keep_indices, :][:, keep_indices].tocsr()
#         sub_nodes = pan_nodes[keep_indices]
#         sub_counts = pan_counts[keep_indices]
#
#         layout_csr = sub_csr.copy().astype(np.float64)
#         if layout_csr.nnz > 0:
#             layout_csr.data = 1.0 + 0.30 * np.log1p(layout_csr.data)
#
#         pan_engine = LayoutEngine(
#             mode="fr",
#             dim=2,
#             iterations=200,
#             gravity=1.20,
#             scale=1.0,
#             seed=42,
#         )
#         coords = pan_engine.layout(layout_csr)
#
#         n_sub = len(sub_nodes)
#         max_edge_per_node = np.zeros(n_sub, dtype=np.int32)
#         edge_rows: list[dict[str, Any]] = []
#         for u in range(n_sub):
#             u_val = int(sub_nodes[u])
#             u_role = GeneRole(int(GeneNode.ROLE_ID.extract(u_val)))
#             u_cid = int(GeneNode.CLUSTER_ID.extract(u_val))
#             for k_idx in range(int(sub_csr.indptr[u]), int(sub_csr.indptr[u + 1])):
#                 v = int(sub_csr.indices[k_idx])
#                 w = int(sub_csr.data[k_idx])
#                 if u == v or w < min_edge_support:
#                     continue
#                 max_edge_per_node[u] = max(max_edge_per_node[u], w)
#                 max_edge_per_node[v] = max(max_edge_per_node[v], w)
#                 v_val = int(sub_nodes[v])
#                 v_role = GeneRole(int(GeneNode.ROLE_ID.extract(v_val)))
#                 v_cid = int(GeneNode.CLUSTER_ID.extract(v_val))
#                 edge_rows.append(
#                     {
#                         "x": round(float(coords[u, 0]), 4),
#                         "y": round(float(coords[u, 1]), 4),
#                         "x2": round(float(coords[v, 0]), 4),
#                         "y2": round(float(coords[v, 1]), 4),
#                         "weight": w,
#                         "transition": f"{u_role.abbreviation} (C{u_cid}) → {v_role.abbreviation} (C{v_cid})",
#                     }
#                 )
#
#         node_rows: list[dict[str, Any]] = []
#         top_indices = set(np.argsort(sub_counts)[-28:].tolist())
#         for i in range(n_sub):
#             n_val = int(sub_nodes[i])
#             role_obj = GeneRole(int(GeneNode.ROLE_ID.extract(n_val)))
#             cid = int(GeneNode.CLUSTER_ID.extract(n_val))
#             cnt = int(sub_counts[i])
#             role_name = role_obj.name or "UNKNOWN"
#             node_rows.append(
#                 {
#                     "x": round(float(coords[i, 0]), 4),
#                     "y": round(float(coords[i, 1]), 4),
#                     "role": role_name,
#                     "role_abbrev": role_obj.abbreviation,
#                     "category": role_obj.category.value,
#                     "cluster_id": f"CID-{cid:04d}",
#                     "locus_count": cnt,
#                     "max_edge": max(int(max_edge_per_node[i]), min_edge_support),
#                     "label": f"{role_obj.abbreviation} (C{cid})" if i in top_indices else "",
#                 }
#             )
#
#         max_w = max((int(r["weight"]) for r in edge_rows), default=min_edge_support + 2)
#         edge_slider = alt.binding_range(
#             min=min_edge_support,
#             max=max(min_edge_support + 1, min(25, max_w)),
#             step=1,
#             name="Min Shared Loci per Synteny Edge: ",
#         )
#         edge_param = alt.param(name="locus_pan_min_edge", value=min_edge_support, bind=edge_slider)
#         cat_sel = alt.selection_point(fields=["category"], bind="legend")
#
#         edges_chart = (
#             alt.Chart(alt.Data(values=edge_rows))
#             .mark_rule(strokeCap="round", color="#64748b")
#             .transform_filter(alt.datum.weight >= edge_param)
#             .encode(
#                 x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
#                 y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
#                 x2="x2:Q",
#                 y2="y2:Q",
#                 strokeWidth=alt.StrokeWidth(
#                     "weight:Q",
#                     scale=alt.Scale(type="sqrt", range=[0.8, 5.0]),
#                     title="Shared Loci",
#                 ),
#                 opacity=alt.Opacity("weight:Q", scale=alt.Scale(range=[0.20, 0.78]), legend=None),
#                 tooltip=[
#                     alt.Tooltip("transition:N", title="Syntenic Transition"),
#                     alt.Tooltip("weight:Q", title="Shared Loci"),
#                 ],
#             )
#         )
#
#         nodes_base = alt.Chart(alt.Data(values=node_rows)).transform_filter(alt.datum.max_edge >= edge_param)
#
#         nodes_chart = (
#             nodes_base.mark_point(filled=True, shape=ARROW_FWD_PATH, stroke="#0f172a", strokeWidth=1.0)
#             .encode(
#                 x=alt.X("x:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
#                 y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.15, 1.15]), axis=None),
#                 size=alt.Size("locus_count:Q", scale=alt.Scale(range=[80, 620]), title="Locus Count"),
#                 color=alt.Color("category:N", scale=alt.Scale(scheme="tableau10"), title="Functional Category"),
#                 opacity=alt.condition(cat_sel, alt.value(0.92), alt.value(0.15)),
#                 tooltip=[
#                     alt.Tooltip("cluster_id:N", title="Homology Cluster"),
#                     alt.Tooltip("role_abbrev:N", title="Gene Role"),
#                     alt.Tooltip("role:N", title="Role Class"),
#                     alt.Tooltip("category:N", title="Category"),
#                     alt.Tooltip("locus_count:Q", title="Locus Count"),
#                 ],
#             )
#             .add_params(edge_param, cat_sel)
#         )
#
#         text_chart = (
#             nodes_base.transform_filter(alt.datum.label != "")
#             .mark_text(dy=-12, fontSize=9.5, fontWeight="bold", color="#1e293b")
#             .encode(
#                 x="x:Q",
#                 y="y:Q",
#                 text="label:N",
#             )
#         )
#
#         return (edges_chart + nodes_chart + text_chart).properties(
#             width=self.width,
#             height=440,
#             title=title,
#         )
