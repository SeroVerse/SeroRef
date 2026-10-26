# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair>=5.5.0",
#     "cassetter>=0.14.0",
#     "h2>=4.4.1",
#     "httpx2>=2.13.0",
#     "marimo>=0.23.1",
#     "numba>=0.65.0",
#     "numpy>=2.4.4",
#     "seroverse[all]",
#     "seroref",
# ]
#
# [tool.uv.sources]
# seroverse = { path = "../../SeroVerse", editable = true }
# seroref = { path = "..", editable = true }
# ///

"""Unified SeroRef Strain–Assembly–Locus Reference Database Curation Marimo Slide Deck."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="SeroRef — De Novo Biosynthetic Locus & Reference Strain Curation",
    layout_file="layouts/curate_seroref.slides.json",
)


@app.cell
def _():
    import asyncio
    import os
    import re
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import altair as alt
    import marimo as mo
    import numpy as np
    from cassetter import Cassetter, RawRequest, RawResponse, SkipRecording
    from seroverse.client.dsmz import BacDiveClient, StrainInfoClient, StrainPassport
    from seroverse.client.ena import ENAClient
    from seroverse.client.ncbi import NCBIClient

    from seroref.db.models import ReferenceStrain, SeroRefRecord
    from seroref.db.records import DEFAULT_SEROREF_DB_PATH, ReferenceStrains, SeroRefRecords
    from seroref.db.species import BacterialSpecies
    from seroref.genes.roles import GeneRole
    from seroref.loci.archetypes import ArchetypeCategory, LocusArchetype
    from seroref.loci.predictor import LocusPredictor

    alt.data_transformers.disable_max_rows()

    DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
    CASSETTE_DIR = DEFAULT_CACHE_DIR / "cassettes"
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)

    @dataclass(frozen=True, slots=True)
    class NCTCReferenceStrain:
        """Harvested ENA NCTC3000 (`PRJEB6403`) reference assembly metadata."""

        sample_accession: str
        organism: str = ""
        tax_id: int = 0
        strain: str = ""
        serotype: str = ""
        assembly_accession: str = ""

    _ACCESSION_RANGE_RE = re.compile(r"^([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$")

    def expand_accession_spec(spec: tuple[bytes, bytes] | tuple[str, str] | bytes | str) -> list[str]:
        """Expand an accession range `(b'AB811596', b'AB812084')` or `'AB811596:AB812084'` into individual IDs."""
        if isinstance(spec, tuple):
            a_str = spec[0].decode("ascii") if isinstance(spec[0], (bytes, bytearray)) else str(spec[0])
            b_str = spec[1].decode("ascii") if isinstance(spec[1], (bytes, bytearray)) else str(spec[1])
            spec_str = f"{a_str.strip()}:{b_str.strip()}"
        elif isinstance(spec, (bytes, bytearray)):
            spec_str = spec.decode("ascii").strip()
        else:
            spec_str = spec.strip()
        m = _ACCESSION_RANGE_RE.match(spec_str)
        if m is None:
            return [spec_str] if spec_str else []
        p1, s1, p2, s2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if p1.upper() != p2.upper():
            return [spec_str]
        start_n, end_n = int(s1), int(s2)
        if end_n < start_n or (end_n - start_n) > 2000:
            return [spec_str]
        width = len(s1)
        return [f"{p1.upper()}{num:0{width}d}" for num in range(start_n, end_n + 1)]

    async def harvest_ena_nctc3000(
        cache_path: Path,
        *,
        force_refresh: bool = False,
    ) -> list[NCTCReferenceStrain]:
        """Harvest ENA NCTC3000 (`PRJEB6403`) PacBio reference assemblies and strain/serovar metadata."""
        if not force_refresh and cache_path.is_file():
            try:
                return ENAClient.load_msgpack(cache_path, list[NCTCReferenceStrain])
            except Exception:
                pass
        async with ENAClient() as ena:
            samples = await ena.search_samples('study_accession="PRJEB6403"', limit=4000)
            assemblies = await ena.search_assemblies('study_accession="PRJEB6403"', limit=4000)
        asm_by_sample = {a.sample_accession: a.accession for a in assemblies if a.sample_accession and a.accession}
        refs: list[NCTCReferenceStrain] = []
        for s in samples:
            tid = int(s.tax_id) if str(s.tax_id).strip().isdigit() else 0
            sp = BacterialSpecies.resolve(s.scientific_name)
            if sp == BacterialSpecies.UNKNOWN and tid > 0:
                sp = BacterialSpecies.from_taxid(tid)
            if sp == BacterialSpecies.UNKNOWN:
                continue
            refs.append(
                NCTCReferenceStrain(
                    sample_accession=s.accession,
                    organism=sp.scientific_name_str,
                    tax_id=tid or sp.ncbi_taxid,
                    strain=s.strain,
                    serotype=s.serovar,
                    assembly_accession=asm_by_sample.get(s.accession, ""),
                )
            )
        ENAClient.save_msgpack(cache_path, refs)
        return refs

    async def harvest_straininfo_passports(
        cache_path: Path,
        *,
        force_refresh: bool = False,
    ) -> dict[bytes, list[StrainPassport]]:
        """Harvest DSMZ StrainInfo v2 and BacDive v2 `StrainPassport` equivalence cliques per `BacterialSpecies`."""
        if not force_refresh and cache_path.is_file():
            try:
                raw = StrainInfoClient.load_msgpack(cache_path, dict[str, list[StrainPassport]])
                return {BacterialSpecies.resolve(k).scientific_name: v for k, v in raw.items() if v}
            except Exception:
                pass
        out: dict[bytes, list[StrainPassport]] = {}
        async with StrainInfoClient() as si, BacDiveClient() as bd:
            for sp in BacterialSpecies:
                if sp == BacterialSpecies.UNKNOWN:
                    continue
                passports = await si.passports_for_taxon(sp.scientific_name_str, max_strains=150, bacdive=bd)
                if passports:
                    out[sp.scientific_name] = passports
        StrainInfoClient.save_msgpack(cache_path, {k.decode("utf-8"): v for k, v in out.items()})
        return out

    async def harvest_landmark_loci(
        existing: SeroRefRecords,
        *,
        max_fetch: int = 60,
    ) -> list[SeroRefRecord]:
        """Sweep uncached landmark BGC accessions declared on `BacterialSpecies.landmark_accessions`."""
        known_accs = {a.decode("utf-8").split(".")[0].upper() for a in existing.strains.assembly_ids if a}
        to_fetch: list[tuple[BacterialSpecies, str]] = []
        for sp in BacterialSpecies:
            if sp == BacterialSpecies.UNKNOWN:
                continue
            for spec in sp.landmark_accessions:
                for acc in expand_accession_spec(spec):
                    stem = acc.split(".")[0].upper()
                    if stem not in known_accs:
                        to_fetch.append((sp, acc))
                        known_accs.add(stem)
                        if len(to_fetch) >= max_fetch:
                            break
                if len(to_fetch) >= max_fetch:
                    break
            if len(to_fetch) >= max_fetch:
                break

        if not to_fetch:
            return []

        predictor = LocusPredictor.pretrained()
        new_records: list[SeroRefRecord] = []
        async with NCBIClient() as ncbi:
            accs = [acc for _, acc in to_fetch]
            sp_by_stem = {acc.split(".")[0].upper(): sp for sp, acc in to_fetch}
            summaries = await ncbi.esummary_nuccore(accs)
            seqs_by_acc = await ncbi.fetch_sequences_by_accessions(accs)
            for acc, seqs in seqs_by_acc.items():
                if len(seqs) == 0:
                    continue
                stem = acc.split(".")[0].upper()
                sp = sp_by_stem.get(stem, BacterialSpecies.UNKNOWN)
                summary = summaries.get(acc) or summaries.get(stem)
                title = getattr(summary, "title", "") if summary else ""
                strain_str = getattr(summary, "strain", "") if summary else ""
                taxid = int(getattr(summary, "taxid", 0) or 0) if summary else sp.ncbi_taxid
                loci = predictor.predict(seqs)
                ref_strain = ReferenceStrain.from_raw(
                    sp,
                    strain_str,
                    raw_serotype=title,
                    assembly_id=acc,
                    ncbi_taxid=taxid,
                )
                new_records.extend(ref_strain.bind_loci(loci, raw_serotype=title))
        return new_records

    def enrich_records_with_metadata(
        records: SeroRefRecords,
        nctc_refs: list[NCTCReferenceStrain],
        passports_by_org: dict[bytes, list[StrainPassport]],
    ) -> SeroRefRecords:
        """Enrich `SeroRefRecords` with ENA NCTC3000 and DSMZ StrainInfo `StrainPassport` equivalence cliques."""
        nctc_by_asm: dict[bytes, NCTCReferenceStrain] = {}
        for ref in nctc_refs:
            if ref.assembly_accession:
                stem = ref.assembly_accession.split(".")[0].upper().encode("utf-8")
                nctc_by_asm[stem] = ref
                nctc_by_asm[ref.assembly_accession.encode("utf-8")] = ref

        passport_by_org_des: dict[tuple[bytes, bytes], StrainPassport] = {}
        passport_by_seq: dict[bytes, StrainPassport] = {}
        for org_b, plist in passports_by_org.items():
            for p in plist:
                for d in p.designations:
                    d_norm = re.sub(r"[\s\-_/]+", "", d.strip().upper()).encode("utf-8")
                    if d_norm:
                        passport_by_org_des.setdefault((org_b, d_norm), p)
                for seq_acc in (*p.sequence_accessions, *p.genome_accessions):
                    stem = seq_acc.split(".")[0].upper().encode("utf-8")
                    if stem:
                        passport_by_seq.setdefault(stem, p)

        enriched_list: list[SeroRefRecord] = []
        for rec in records:
            sr = rec.strain_record
            asm_stem = sr.assembly_id.split(b".")[0].upper() if sr.assembly_id else b""
            raw_sero = rec.serotype

            # 1. Enrich from ENA NCTC3000 assembly match if present
            nctc = nctc_by_asm.get(sr.assembly_id) or nctc_by_asm.get(asm_stem)
            if nctc is not None:
                nctc_sr = ReferenceStrain.from_raw(
                    sr.species,
                    nctc.strain,
                    raw_serotype=nctc.serotype,
                    assembly_id=sr.assembly_id,
                    ncbi_taxid=nctc.tax_id,
                )
                sr = sr.merge(nctc_sr)
                if not raw_sero and nctc.serotype:
                    raw_sero = nctc.serotype.encode("utf-8")

            # 2. Enrich from DSMZ StrainInfo / BacDive passport match
            matched_passport: Any = passport_by_seq.get(asm_stem) if asm_stem else None
            if matched_passport is None:
                for cand in (sr.name, *sr.synonyms):
                    c_norm = re.sub(rb"[\s\-_/]+", b"", cand.strip().upper())
                    if c_norm:
                        matched_passport = passport_by_org_des.get((sr.organism, c_norm))
                        if matched_passport is not None:
                            break
            if matched_passport is not None:
                sr = sr.with_passport(matched_passport, raw_serotype=raw_sero)

            enriched_list.append(
                SeroRefRecord(
                    strain_record=sr,
                    serotype=raw_sero,
                    locus=rec.locus,
                )
            )

        return SeroRefRecords.from_list(enriched_list, deduplicate=True)

    def _skip_efetch_requests(request: RawRequest) -> RawRequest:
        if "efetch.fcgi" in request.uri or "esummary.fcgi" in request.uri or "/download" in request.uri:
            raise SkipRecording
        return request

    def _skip_error_responses(response: RawResponse) -> RawResponse:
        if response.status >= 400:
            raise SkipRecording
        return response

    RECORDER = Cassetter(
        cassette_library_dir=CASSETTE_DIR,
        record_mode=os.environ.get("SEROREF_RECORD_MODE", "new_episodes"),
        match_on=["method", "uri", "body"],
        filter_query_parameters=["api_key"],
        ignore_hosts=[
            "ftp.ncbi.nlm.nih.gov",
            "api.straininfo.dsmz.de",
            "api.bacdive.dsmz.de",
            "www.ebi.ac.uk",
        ],
        before_record_request=_skip_efetch_requests,
        before_record_response=_skip_error_responses,
    )

    return (
        ArchetypeCategory,
        BacterialSpecies,
        DEFAULT_CACHE_DIR,
        DEFAULT_SEROREF_DB_PATH,
        GeneRole,
        LocusArchetype,
        NCTCReferenceStrain,
        RECORDER,
        ReferenceStrain,
        ReferenceStrains,
        SeroRefRecord,
        SeroRefRecords,
        StrainPassport,
        alt,
        asyncio,
        enrich_records_with_metadata,
        harvest_ena_nctc3000,
        harvest_landmark_loci,
        harvest_straininfo_passports,
        mo,
        np,
        os,
        time,
    )


@app.cell
async def _(
    BacterialSpecies,
    DEFAULT_CACHE_DIR,
    DEFAULT_SEROREF_DB_PATH,
    GeneRole,
    LocusArchetype,
    SeroRefRecords,
    alt,
    enrich_records_with_metadata,
    harvest_ena_nctc3000,
    harvest_landmark_loci,
    harvest_straininfo_passports,
    np,
    os,
    time,
):
    t0 = time.perf_counter()
    ena_cache_file = DEFAULT_CACHE_DIR / "ena_nctc3000.msgpack"
    si_cache_file = DEFAULT_CACHE_DIR / "straininfo_passports.msgpack"
    force_rebuild = os.environ.get("SEROREF_REBUILD", "0") == "1"

    base_records = (
        SeroRefRecords.load(DEFAULT_SEROREF_DB_PATH) if DEFAULT_SEROREF_DB_PATH.is_file() else SeroRefRecords.empty()
    )
    nctc_refs = await harvest_ena_nctc3000(ena_cache_file, force_refresh=force_rebuild)
    passports_by_org = await harvest_straininfo_passports(si_cache_file, force_refresh=force_rebuild)

    if force_rebuild:
        max_nuccore = int(os.environ.get("SEROREF_MAX_NUCCORE", "60"))
        new_landmark_records = await harvest_landmark_loci(base_records, max_fetch=max_nuccore)
        if new_landmark_records:
            base_records = SeroRefRecords.from_list([*base_records, *new_landmark_records], deduplicate=True)

    seroref_records = enrich_records_with_metadata(base_records, nctc_refs, passports_by_org)
    curation_elapsed_s = time.perf_counter() - t0

    npz_path = seroref_records.save(DEFAULT_SEROREF_DB_PATH)
    _reloaded = SeroRefRecords.load(npz_path)
    npz_parity = bool(
        len(_reloaded) == len(seroref_records)
        and len(_reloaded.strains) == len(seroref_records.strains)
        and np.array_equal(_reloaded.strain_indices, seroref_records.strain_indices)
        and np.array_equal(_reloaded.species_ids, seroref_records.species_ids)
        and np.array_equal(_reloaded.serotypes, seroref_records.serotypes)
        and np.array_equal(_reloaded.assembly_ids, seroref_records.assembly_ids)
        and np.array_equal(_reloaded.antigen_types, seroref_records.antigen_types)
        and np.array_equal(_reloaded.loci.names, seroref_records.loci.names)
        and np.array_equal(_reloaded.loci.node_types, seroref_records.loci.node_types)
    )
    npz_kb = npz_path.stat().st_size / 1024.0
    species_specs = BacterialSpecies.specs()
    strain_locus_csr = seroref_records.strain_locus_csr
    pan_csr, pan_nodes, _ = seroref_records.loci.as_pan_graph()
    pan_components, _ = pan_csr.connected_components()

    # 1. LocusArchetype Distribution Chart
    _u_arch, _a_counts = np.unique(seroref_records.archetypes, return_counts=True)
    _a_order = np.argsort(-_a_counts)
    _arch_rows = []
    for _idx in _a_order:
        _arch = LocusArchetype(int(_u_arch[_idx]))
        _arch_rows.append(
            {
                "archetype": _arch.name,
                "category": _arch.category.label.decode("ascii"),
                "count": int(_a_counts[_idx]),
            }
        )

    archetype_chart = (
        alt.Chart(alt.Data(values=_arch_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "archetype:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="De Novo Predicted LocusArchetype",
            ),
            x=alt.X("count:Q", title="Curated SeroRefRecords"),
            color=alt.Color(
                "category:N",
                title="Archetype Category",
                scale=alt.Scale(scheme="tealblues"),
            ),
            tooltip=[
                alt.Tooltip("archetype:N", title="LocusArchetype"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Records", format=","),
            ],
        )
        .properties(title="Curated SeroRefRecords by De Novo LocusArchetype", width=360, height=320)
    )

    # 2. Top 15 Bacterial Species by Surface Antigen Class
    _orgs, _org_counts = np.unique(seroref_records.organisms, return_counts=True)
    _org_order = np.argsort(-_org_counts)[:15]
    _species_rows = []
    _ag_arr = seroref_records.antigen_types
    for _idx in _org_order:
        _org_b = _orgs[_idx]
        _org_name = _org_b.decode("utf-8", errors="replace")
        _tot_c = int(_org_counts[_idx])
        _mask = seroref_records.organisms == _org_b
        _sub_ags, _sub_cnts = np.unique(_ag_arr[_mask], return_counts=True)
        for _ag_b, _c in zip(_sub_ags, _sub_cnts, strict=False):
            _species_rows.append(
                {
                    "organism": _org_name,
                    "antigen_type": bytes(_ag_b).rstrip(b"\x00").decode("utf-8", errors="replace") or "Other",
                    "records": int(_c),
                    "total_count": _tot_c,
                }
            )

    species_chart = (
        alt.Chart(alt.Data(values=_species_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "organism:N",
                sort=alt.EncodingSortField(field="total_count", op="max", order="descending"),
                title="Bacterial Species",
            ),
            x=alt.X("records:Q", title="Curated SeroRefRecords"),
            color=alt.Color(
                "antigen_type:N",
                title="Surface Antigen Class",
                scale=alt.Scale(
                    domain=["O-antigen", "CPS", "EPS", "WTA"],
                    range=["#0d9488", "#0284c7", "#d97706", "#7c3aed"],
                ),
            ),
            tooltip=[
                alt.Tooltip("organism:N", title="Organism"),
                alt.Tooltip("antigen_type:N", title="Antigen Class"),
                alt.Tooltip("records:Q", title="Loci", format=","),
                alt.Tooltip("total_count:Q", title="Species Total", format=","),
            ],
        )
        .properties(title="Top 15 Bacterial Species by Surface Antigen Class", width=380, height=320)
    )

    # 3. Canonical GeneRole Distribution across SeroRef Loci
    _node_types = seroref_records.loci.node_types
    _role_ids = _node_types & 0x1F
    _u_roles, _r_counts = np.unique(_role_ids, return_counts=True)
    _r_order = np.argsort(-_r_counts)[:18]
    _role_rows = []
    for _idx in _r_order:
        _rid = int(_u_roles[_idx])
        if 0 <= _rid < len(GeneRole):
            _role = GeneRole(_rid)
            _role_rows.append(
                {
                    "abbreviation": _role.abbreviation.decode("ascii"),
                    "role_name": _role.name,
                    "category": _role.category.label.decode("ascii"),
                    "count": int(_r_counts[_idx]),
                }
            )

    roles_chart = (
        alt.Chart(alt.Data(values=_role_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "abbreviation:N",
                sort=alt.EncodingSortField(field="count", order="descending"),
                title="Canonical GeneRole (Abbrev)",
            ),
            x=alt.X("count:Q", title="Predicted BGC Gene Nodes"),
            color=alt.Color(
                "category:N",
                title="Functional Category",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("abbreviation:N", title="Abbrev"),
                alt.Tooltip("role_name:N", title="GeneRole"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Gene Nodes", format=","),
            ],
        )
        .properties(title="Top 18 Biosynthetic GeneRoles across Curated Loci", width=360, height=340)
    )

    # 4. Locus Gene Count vs Completeness Distribution
    _gene_counts = seroref_records.loci.node_lengths
    _completeness = seroref_records.loci.completeness * 100.0
    _hist_rows = []
    for _gc, _comp, _ag_b in zip(_gene_counts, _completeness, _ag_arr, strict=False):
        _hist_rows.append(
            {
                "gene_count": int(_gc),
                "completeness": float(_comp),
                "antigen_type": bytes(_ag_b).rstrip(b"\x00").decode("utf-8", errors="replace") or "Other",
            }
        )

    locus_size_chart = (
        alt.Chart(alt.Data(values=_hist_rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("gene_count:Q", bin=alt.Bin(maxbins=20), title="ORFs per Segmented BGC Locus"),
            y=alt.Y("count():Q", title="Curated SeroRefRecords"),
            color=alt.Color(
                "antigen_type:N",
                title="Surface Antigen Class",
                scale=alt.Scale(
                    domain=["O-antigen", "CPS", "EPS", "WTA"],
                    range=["#0d9488", "#0284c7", "#d97706", "#7c3aed"],
                ),
            ),
            tooltip=[
                alt.Tooltip("antigen_type:N", title="Antigen Class"),
                alt.Tooltip("count():Q", title="Loci", format=","),
            ],
        )
        .properties(title="BGC ORF Count Distribution across Curated Loci", width=380, height=340)
    )

    # 5. Top 25 Curated SeroRefRecords Table
    _scores = seroref_records.loci.scores
    _top_order = np.argsort(-_scores)[:25]
    top_record_rows = []
    for _idx in _top_order:
        _rec = seroref_records[int(_idx)]
        _loc = _rec.locus
        top_record_rows.append(
            {
                "organism": _rec.organism.decode("utf-8", errors="replace"),
                "antigen": _rec.antigen_type.decode("utf-8", errors="replace"),
                "serotype": _rec.serotype.decode("utf-8", errors="replace") or "—",
                "strain": _rec.strain.decode("utf-8", errors="replace") or "—",
                "assembly": _rec.assembly_id.decode("utf-8", errors="replace"),
                "archetype": _rec.archetype.name,
                "genes": len(_loc.nodes) if _loc is not None else 0,
                "completeness": float(_loc.completeness) * 100.0 if _loc is not None else 0.0,
                "score": float(_loc.score) if _loc is not None else 0.0,
                "taxid": int(_rec.ncbi_taxid),
                "pmids": len(_rec.pmids),
            }
        )

    total_passports = sum(len(v) for v in passports_by_org.values())
    typed_nctc_count = sum(1 for r in nctc_refs if r.serotype)

    return (
        archetype_chart,
        curation_elapsed_s,
        locus_size_chart,
        nctc_refs,
        npz_kb,
        npz_parity,
        npz_path,
        pan_components,
        pan_csr,
        pan_nodes,
        passports_by_org,
        roles_chart,
        seroref_records,
        species_chart,
        species_specs,
        strain_locus_csr,
        top_record_rows,
        total_passports,
        typed_nctc_count,
    )


@app.cell
def _(
    curation_elapsed_s,
    mo,
    nctc_refs,
    np,
    npz_kb,
    npz_parity,
    pan_components,
    pan_csr,
    pan_nodes,
    seroref_records,
    species_specs,
    strain_locus_csr,
    total_passports,
    typed_nctc_count,
):
    # Slide 1: Hero Title, Executive Summary & KPI Cards
    _n_total = len(seroref_records)
    _n_strains = len(seroref_records.strains)
    _n_orgs = len(np.unique(seroref_records.species_ids))
    _tot_nodes = len(seroref_records.loci.node_types)
    _tot_edges = len(seroref_records.loci.edge_types)
    _multi_loci_strains = int(np.sum(np.asarray(strain_locus_csr.sum(axis=1)) > 1))

    mo.vstack(
        [
            mo.md(
                """
                # `SeroRef` — De Novo Biosynthetic Locus & Reference Strain Curation
                Harvesting, resolving, and segmenting bacterial surface polysaccharide biosynthetic gene clusters (*de novo* via
                [`OrfPredictor`][seroref.genes.orf.predictor.OrfPredictor] + [`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier] + [`LocusPredictor`][seroref.loci.predictor.LocusPredictor])
                using data-oriented [`BacterialSpecies`][seroref.db.species.BacterialSpecies] specifications, deduplicated [`ReferenceStrains`][seroref.db.records.ReferenceStrains],
                pure-NumPy [`CSR`][seroverse.core.graph.csr.CSR] incidence/pan-locus graphs, and relational [`SeroRefRecords`][seroref.db.records.SeroRefRecords].
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Curated SeroRefRecords",
                        value=f"{_n_total:,}",
                        caption=f"{_n_orgs:,} Species | {_n_strains:,} Strains ({_multi_loci_strains:,} Multi-Locus)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Pan-Locus CSR Graph",
                        value=f"{len(pan_nodes):,} Nodes",
                        caption=f"{pan_csr.nnz:,} CSR Edges | {pan_components:,} Components ({_tot_nodes:,} ORFs)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Species Ontology",
                        value=f"{len(species_specs) - 1:,} Species",
                        caption="SpeciesSpecs SoA + SerotypeRules",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Reference Strain Cliques",
                        value=f"{total_passports:,} Passports",
                        caption=f"{len(nctc_refs):,} NCTC3000 ({typed_nctc_count:,} Typed)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Relational SoA Archive",
                        value=f"{npz_kb:.1f} KB",
                        caption=f"src/seroref/data/seroref.npz | Parity: {'PASS' if npz_parity else 'FAIL'}",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.callout(
                mo.md(
                    f"**Exported Relational Reference Database (`src/seroref/data/seroref.npz`)**: "
                    f"`{_n_total:,}` de novo segmented surface polysaccharide loci (`{_tot_nodes:,}` classified ORFs) "
                    f"linked to `{_n_strains:,}` deduplicated [`ReferenceStrains`][seroref.db.records.ReferenceStrains] via `strain_indices` (`{npz_parity}`)."
                ),
                kind="success",
            ),
        ],
        gap=1.2,
    )
    return


@app.cell
def _(alt, archetype_chart, mo, species_chart):
    # Slide 2: LocusArchetype Distribution & Top 15 Species
    mo.vstack(
        [
            mo.md(
                """
                ## 1. De Novo `LocusArchetype` Repertoire & Taxonomic Coverage
                Every reference assembly is segmented *de novo* by [`LocusPredictor`][seroref.loci.predictor.LocusPredictor], splitting
                multi-antigen formulas (`O1:K1:H7`) via [`BacterialSpecies.parse_serotype`][seroref.db.species.BacterialSpecies.parse_serotype] and [`ReferenceStrain.bind_loci`][seroref.db.models.ReferenceStrain.bind_loci].
                """
            ),
            alt.hconcat(archetype_chart, species_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(alt, locus_size_chart, mo, roles_chart):
    # Slide 3: Canonical GeneRole Repertoire & Locus Size Distribution
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Canonical `GeneRole` Repertoire & BGC Architecture
                Distribution of the 30 canonical biosynthetic gene roles predicted by the Multi-Scale 1D-CNN
                [`GeneRoleClassifier`][seroref.genes.cnn.GeneRoleClassifier] and ORF counts across segmented gene clusters.
                """
            ),
            alt.hconcat(roles_chart, locus_size_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    mo,
    nctc_refs,
    np,
    passports_by_org,
    seroref_records,
    species_specs,
    total_passports,
    typed_nctc_count,
):
    # Slide 4: Relational Provenance & Cross-Reference Enrichment
    _strains = seroref_records.strains
    _n_with_strain = int(np.sum(_strains.names != b""))
    _n_with_taxid = int(np.sum(_strains.ncbi_taxids > 0))
    _n_with_pmids = int(np.sum(_strains.pmid_lengths > 0))
    _n_with_syns = int(np.sum(_strains.lengths > 0))
    _n_with_si = int(np.sum(_strains.straininfo_ids > 0))

    _tier_table = "\n".join(
        [
            "| Curation Layer | Primary Client / SoA Container | Local Artifact | Harvested Entities | Role in `SeroRefRecords` Compilation |",
            "| :--- | :--- | :--- | ---: | :--- |",
            f"| **1 — `SpeciesSpecs` Ontology** | `BacterialSpecies` + `SpeciesSpec` | `seroref.db.species` | **{len(species_specs) - 1:,}** species | Declarative `SerotypeRule` grammar, Kauffmann-White mapping, and `landmark_accessions` |",
            f"| **2 — ENA NCTC3000 (`PRJEB6403`)** | `ENAClient` + `ReferenceStrain` | `.cache/ena_nctc3000.msgpack` | **{len(nctc_refs):,}** assemblies ({typed_nctc_count:,} typed) | PacBio reference genomes & multi-antigen formula splitting (`O1:K1:H7`) |",
            f"| **3 — DSMZ StrainInfo & BacDive** | `StrainInfoClient` + `BacDiveClient` | `.cache/straininfo_passports.msgpack` | **{total_passports:,}** passports ({len(passports_by_org):,} species) | Culture collection equivalence cliques (`synonym_values`), `straininfo_id`, `bacdive_id`, `pmids` |",
            f"| **4 — Relational `SeroRefRecords`** | `ReferenceStrains` + `Loci` | `src/seroref/data/seroref.npz` | **{len(seroref_records):,}** loci ({len(_strains):,} strains) | Normalized foreign-key `strain_indices` linking `N` BGC loci to `M` unique `ReferenceStrains` |",
        ]
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 3. Relational `ReferenceStrains` Table & Cross-Database Provenance
                Deduplicating reference strain isolates (`M` rows in [`ReferenceStrains`][seroref.db.records.ReferenceStrains]) and linking `N` de novo BGC loci via `strain_indices`.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="TaxID Annotated Strains",
                        value=f"{_n_with_taxid:,}",
                        caption=f"{_n_with_taxid / max(1, len(_strains)) * 100:.1f}% of ReferenceStrains",
                        bordered=True,
                    ),
                    mo.stat(
                        label="PubMed Linked Strains",
                        value=f"{_n_with_pmids:,}",
                        caption=f"{len(_strains.pmid_values):,} total PMID links",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Named Strain Isolates",
                        value=f"{_n_with_strain:,}",
                        caption="True strain isolates (never serotype copy)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Synonym Cliques",
                        value=f"{_n_with_syns:,}",
                        caption=f"{len(_strains.synonym_values):,} culture collection aliases",
                        bordered=True,
                    ),
                    mo.stat(
                        label="StrainInfo Passports",
                        value=f"{_n_with_si:,}",
                        caption=f"From {total_passports:,} DSMZ passports",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.md(_tier_table),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo, top_record_rows):
    # Slide 5: Reference Table of Top Curated SeroRefRecords
    _rows_md = "\n".join(
        f"| *{r['organism']}* | {r['antigen']} | `{r['serotype']}` | `{r['strain']}` | `{r['assembly']}` | `{r['archetype']}` | {r['genes']} | {r['completeness']:.1f}% | {r['taxid'] or '—'} | {r['pmids']} | **{r['score']:.2f}** |"
        for r in top_record_rows
    )
    _table_md = "\n".join(
        [
            "| Organism | Antigen Class | Serotype / `locus.name` | Reference Strain | NCBI Accession | `LocusArchetype` | BGC ORFs | Completeness | TaxID | PMIDs | Score |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            _rows_md,
        ]
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 4. Reference Table: Top 25 De Novo Curated `SeroRefRecords`
                High-confidence biosynthetic gene clusters in `src/seroref/data/seroref.npz` ranked by [`ArchetypeClassifier`][seroref.loci.classifier.ArchetypeClassifier] log-likelihood score.
                """
            ),
            mo.md(_table_md),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
