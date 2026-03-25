#!/usr/bin/env python3
"""
==============================================================================
Enrich matched panels with normalized citation, productivity & funding metrics
==============================================================================

PREREQUISITE: Run build_citation_tables_v2.sql in BigQuery console first.

Normalization approach:
  Citations (old/na):
    For each paper p (pub_year=y, fields={f_i}), in each citing_year t:
      E(p, t) = mean_i( expected_annual_citations(f_i, y, t) )
      normalized(p, t) = raw_citations(p, t) / E(p, t)
    Sum across paper subsets (old = before venue_year, na = not venue_year).

  Productivity:
    For each paper p published in year t with fields {f_i}:
      E_prod = mean_i( expected_publications(f_i, t) )
      normalized_pub_contribution(p) = 1 / E_prod
    Sum across papers per author per year.

Columns added:
  - citations_old / cum_citations_old  (raw)
  - citations_na  / cum_citations_na   (raw)
  - normalized_citations_old  / cum_normalized_citations_old
  - normalized_citations_na   / cum_normalized_citations_na
  - normalized_productivity   / cum_normalized_productivity
  - normalized_publication_count_adj / cum_normalized_publication_count_adj

Usage:
    python enrich_citations_normalized.py \
        --input_dir ../../data/matches \
        --output_dir ../../data/matches/enriched_normalized
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

# =============================================================================
# Config
# =============================================================================

PROJECT = 'ccnr-success'
DATASET = 'venue_effect'
DS = f'{PROJECT}.{DATASET}'


# =============================================================================
# BigQuery helpers
# =============================================================================

def get_bq_client(project):
    return bigquery.Client(project=project)


def query_citation_edges(client, author_ids, chunk_size=1000):
    """
    For each author's papers, get individual citation edges:
      author_id, pub_id, pub_year, citing_pub_id, citing_year
    Covers ALL papers by the author.
    """
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT
            apy.author_id,
            apy.pub_id,
            apy.pub_year,
            e.citing_pub_id,
            e.citing_year
        FROM `{DS}.author_paper_years` apy
        JOIN `{DS}.author_paper_citation_edges` e
            ON apy.pub_id = e.pub_id
        WHERE apy.author_id IN ({ids_str})
        """

        print(f'    Chunk {i // chunk_size + 1} '
              f'({len(chunk)} authors)...', end=' ', flush=True)
        t0 = time.time()
        df = client.query(query).to_dataframe()
        print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def query_paper_fields(client, pub_ids, chunk_size=10000):
    """Get ANZSRC field codes per paper."""
    all_results = []
    pub_list = sorted(set(pub_ids))

    for i in range(0, len(pub_list), chunk_size):
        chunk = pub_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{pid}'" for pid in chunk)

        query = f"""
        SELECT pub_id, field_code
        FROM `{DS}.paper_field_codes`
        WHERE pub_id IN ({ids_str})
        """

        df = client.query(query).to_dataframe()
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def query_field_expected_citations_annual(client):
    """Load field expected citations with (field, pub_year, citing_year)."""
    query = f"""
    SELECT field_code, pub_year, citing_year, expected_annual_citations
    FROM `{DS}.field_expected_citations_annual`
    """
    print('    Loading field expected citations (annual)...', end=' ', flush=True)
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
    return df


def query_field_expected_publications(client):
    """Load field expected publications per author per year."""
    query = f"""
    SELECT field_code, pub_year, expected_publications
    FROM `{DS}.field_expected_publications`
    """
    print('    Loading field expected publications...', end=' ', flush=True)
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
    return df


def query_author_papers(client, author_ids, chunk_size=1000):
    """Get all papers for given authors (for productivity normalization)."""
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT DISTINCT author_id, pub_id, pub_year
        FROM `{DS}.author_paper_years`
        WHERE author_id IN ({ids_str})
        """

        df = client.query(query).to_dataframe()
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# Raw citation breakdown (COUNT DISTINCT citing_pub_id)
# =============================================================================

def compute_citation_breakdown(edges, matched_df):
    """
    For each author x citing_year, count DISTINCT citing papers that cite:
      - 'old': papers published strictly BEFORE venue_year
      - 'na':  papers published in any year EXCEPT venue_year
    """
    author_vy = matched_df.groupby('author_id')['venue_year'].first().to_dict()

    e = edges.copy()
    e['venue_year'] = e['author_id'].map(author_vy)
    e = e.dropna(subset=['venue_year'])
    e['venue_year'] = e['venue_year'].astype(int)

    e['is_old'] = (e['pub_year'] < e['venue_year']).astype(int)
    e['is_na']  = (e['pub_year'] != e['venue_year']).astype(int)

    # citations_old: distinct citers of old papers
    old_edges = e[e['is_old'] == 1]
    citations_old = old_edges.groupby(['author_id', 'citing_year'])['citing_pub_id'] \
        .nunique().reset_index()
    citations_old.columns = ['author_id', 'year', 'citations_old']

    # citations_na: distinct citers of non-venue-year papers
    na_edges = e[e['is_na'] == 1]
    citations_na = na_edges.groupby(['author_id', 'citing_year'])['citing_pub_id'] \
        .nunique().reset_index()
    citations_na.columns = ['author_id', 'year', 'citations_na']

    annual = citations_old.merge(citations_na, on=['author_id', 'year'], how='outer')
    annual['citations_old'] = annual['citations_old'].fillna(0).astype(int)
    annual['citations_na'] = annual['citations_na'].fillna(0).astype(int)

    return annual


# =============================================================================
# Normalized citation breakdown — paper-level CNCI
# =============================================================================

def compute_normalized_citation_breakdown(edges, matched_df, paper_fields,
                                          field_expected_annual,
                                          author_modal_field=None):
    """
    Paper-level CNCI normalization for annual citation flows.

    For each paper p (pub_year=y, fields={f_i}):
      For each citing_year t:
        raw_citations(p, t) = number of distinct citing papers in year t
        E(p, t) = mean_i(expected_annual_citations(f_i, y, t))
        normalized(p, t) = raw_citations(p, t) / E(p, t)

    If a paper has no field codes, falls back to the author's modal_field_code.

    Then aggregate by author x citing_year over the relevant paper subsets.
    """
    author_vy = matched_df.groupby('author_id')['venue_year'].first().to_dict()

    # --- Step 1: Raw citations per paper per citing_year ---
    paper_annual_raw = edges.groupby(['pub_id', 'pub_year', 'citing_year']) \
        ['citing_pub_id'].nunique().reset_index()
    paper_annual_raw.columns = ['pub_id', 'pub_year', 'citing_year', 'raw_citations']

    # --- Step 2: Build expected citations lookup ---
    fe_dict = field_expected_annual.set_index(
        ['field_code', 'pub_year', 'citing_year']
    )['expected_annual_citations'].to_dict()

    pub_fields_dict = paper_fields.groupby('pub_id')['field_code'].apply(list).to_dict()

    # Build paper -> author mapping for modal field fallback
    paper_author = edges[['pub_id', 'author_id']].drop_duplicates()
    paper_to_author = paper_author.groupby('pub_id')['author_id'].first().to_dict()

    # author_modal_field: author_id -> modal_field_code (from matched_df)
    if author_modal_field is None:
        author_modal_field = {}

    # --- Step 3: Compute E(p, t) for each unique (pub_id, pub_year, citing_year) ---
    unique_combos = paper_annual_raw[['pub_id', 'pub_year', 'citing_year']] \
        .drop_duplicates()
    print(f'      Computing expected values for {len(unique_combos):,} '
          f'(paper, citing_year) combinations...')

    n_fallback = 0
    n_no_field = 0
    expected_vals = {}
    for _, row in unique_combos.iterrows():
        pub_id = row['pub_id']
        pub_year = int(row['pub_year'])
        citing_year = int(row['citing_year'])

        fields = pub_fields_dict.get(pub_id, [])

        # Fallback: use author's modal field if paper has no field codes
        if not fields:
            aid = paper_to_author.get(pub_id)
            modal = author_modal_field.get(aid) if aid else None
            if modal:
                fields = [modal]
                n_fallback += 1
            else:
                n_no_field += 1
                expected_vals[(pub_id, pub_year, citing_year)] = None
                continue

        expectations = [
            fe_dict.get((f, pub_year, citing_year))
            for f in fields
        ]
        expectations = [ev for ev in expectations if ev is not None and ev > 0]

        if expectations:
            expected_vals[(pub_id, pub_year, citing_year)] = np.mean(expectations)
        else:
            expected_vals[(pub_id, pub_year, citing_year)] = None

    if n_fallback > 0 or n_no_field > 0:
        print(f'      Citations: {n_fallback:,} combos used modal field fallback, '
              f'{n_no_field:,} had no field at all')

    paper_annual_raw['expected'] = paper_annual_raw.apply(
        lambda r: expected_vals.get(
            (r['pub_id'], int(r['pub_year']), int(r['citing_year']))
        ), axis=1
    )

    # Normalized = raw / expected
    valid = paper_annual_raw['expected'].notna() & (paper_annual_raw['expected'] > 0)
    paper_annual_raw['normalized_citations'] = np.where(
        valid,
        paper_annual_raw['raw_citations'] / paper_annual_raw['expected'],
        np.nan
    )

    paper_norm = paper_annual_raw.dropna(subset=['normalized_citations']).copy()

    n_dropped = len(paper_annual_raw) - len(paper_norm)
    if n_dropped > 0:
        pct = 100 * n_dropped / len(paper_annual_raw)
        print(f'      Dropped {n_dropped:,} rows ({pct:.1f}%) with no field expectation')

    # --- Step 4: Map papers back to authors and tag old/na ---
    author_paper = edges[['author_id', 'pub_id', 'pub_year']].drop_duplicates()
    paper_norm = paper_norm.merge(
        author_paper, on=['pub_id', 'pub_year'], how='inner'
    )

    paper_norm['venue_year'] = paper_norm['author_id'].map(author_vy)
    paper_norm = paper_norm.dropna(subset=['venue_year'])
    paper_norm['venue_year'] = paper_norm['venue_year'].astype(int)

    paper_norm['is_old'] = (paper_norm['pub_year'] < paper_norm['venue_year']).astype(int)
    paper_norm['is_na']  = (paper_norm['pub_year'] != paper_norm['venue_year']).astype(int)

    # --- Step 5: Aggregate by author x citing_year ---
    norm_old = paper_norm[paper_norm['is_old'] == 1].groupby(
        ['author_id', 'citing_year']
    )['normalized_citations'].sum().reset_index()
    norm_old.columns = ['author_id', 'year', 'normalized_citations_old']

    norm_na = paper_norm[paper_norm['is_na'] == 1].groupby(
        ['author_id', 'citing_year']
    )['normalized_citations'].sum().reset_index()
    norm_na.columns = ['author_id', 'year', 'normalized_citations_na']

    annual = norm_old.merge(norm_na, on=['author_id', 'year'], how='outer')
    annual['normalized_citations_old'] = annual['normalized_citations_old'].fillna(0)
    annual['normalized_citations_na'] = annual['normalized_citations_na'].fillna(0)

    return annual


# =============================================================================
# Normalized productivity — paper-level
# =============================================================================

def compute_normalized_productivity(author_papers, matched_df, pub_fields_dict,
                                    fe_pub_dict, author_modal_field=None):
    """
    For each paper p published by the author in year t with fields {f_i}:
      E_prod = mean_i(expected_publications(f_i, t))
      normalized_pub_contribution(p) = 1 / E_prod

    If a paper has no field codes, falls back to the author's modal_field_code.

    Sum across papers per author per year.

    Returns TWO DataFrames:
      - annual_prod: normalized_productivity (all papers)
      - annual_prod_adj: normalized_publication_count_adj (excluding focal paper)
    """
    if author_modal_field is None:
        author_modal_field = {}

    # Get focal paper IDs if available
    focal_papers = set()
    if 'focal_pub_id' in matched_df.columns:
        focal_papers = set(matched_df['focal_pub_id'].dropna().unique())

    n_fallback = 0
    n_no_field = 0
    records_all = []
    records_adj = []

    for _, row in author_papers.iterrows():
        pub_id = row['pub_id']
        pub_year = int(row['pub_year'])
        author_id = row['author_id']

        fields = pub_fields_dict.get(pub_id, [])

        # Fallback: use author's modal field
        if not fields:
            modal = author_modal_field.get(author_id)
            if modal:
                fields = [modal]
                n_fallback += 1
            else:
                n_no_field += 1

        if not fields:
            norm_pub = 1.0
        else:
            expectations = [
                fe_pub_dict.get((f, pub_year))
                for f in fields
            ]
            expectations = [ev for ev in expectations if ev is not None and ev > 0]
            norm_pub = 1.0 / np.mean(expectations) if expectations else 1.0

        rec = {'author_id': author_id, 'year': pub_year, 'norm_pub': norm_pub}
        records_all.append(rec)

        # For adj: exclude focal paper
        if pub_id not in focal_papers:
            records_adj.append(rec.copy())

    # All papers
    if n_fallback > 0 or n_no_field > 0:
        print(f'      Productivity: {n_fallback:,} papers used modal field fallback, '
              f'{n_no_field:,} had no field at all')
    annual_prod = pd.DataFrame()
    if records_all:
        df_all = pd.DataFrame(records_all)
        annual_prod = df_all.groupby(['author_id', 'year'])['norm_pub'] \
            .sum().reset_index()
        annual_prod.columns = ['author_id', 'year', 'normalized_productivity']

    # Adjusted (excluding focal)
    annual_prod_adj = pd.DataFrame()
    if records_adj:
        df_adj = pd.DataFrame(records_adj)
        annual_prod_adj = df_adj.groupby(['author_id', 'year'])['norm_pub'] \
            .sum().reset_index()
        annual_prod_adj.columns = ['author_id', 'year',
                                   'normalized_publication_count_adj']

    return annual_prod, annual_prod_adj


# =============================================================================
# Enrich one file
# =============================================================================

def enrich_one_file(input_path, output_path, client,
                    field_expected_annual=None,
                    field_expected_pubs=None,
                    skip_normalized=False):

    print(f'\n{"="*60}')
    print(f'Enriching: {input_path}')
    print(f'{"="*60}')

    df = pd.read_csv(input_path, sep=';')
    print(f'  Loaded: {len(df):,} rows, {df["author_id"].nunique():,} authors')

    author_ids = list(df['author_id'].unique())

    # ── Query citation edges ──
    print(f'  Querying citation edges for {len(author_ids):,} authors...')
    edges = query_citation_edges(client, author_ids)
    if edges.empty:
        print('  WARNING: No citation data returned. Skipping.')
        return

    print(f'  Total citation edges: {len(edges):,}')

    # ── Query paper fields (needed for all normalizations) ──
    paper_fields = pd.DataFrame()
    pub_fields_dict = {}
    if not skip_normalized:
        pub_ids_from_edges = list(edges['pub_id'].unique())
        print(f'  Querying field codes for {len(pub_ids_from_edges):,} papers...')
        paper_fields = query_paper_fields(client, pub_ids_from_edges)
        print(f'    {len(paper_fields):,} paper-field rows')
        pub_fields_dict = paper_fields.groupby('pub_id')['field_code'] \
            .apply(list).to_dict()

    # ── Raw citation breakdown ──
    print('  Computing raw citation breakdown (old / na)...')
    annual_raw = compute_citation_breakdown(edges, df)
    print(f'    {len(annual_raw):,} author-year rows')

    # ── Normalized citation breakdown ──
    annual_norm = None
    # Build author -> modal_field_code mapping for fallback
    author_modal_field = {}
    if 'modal_field_code' in df.columns:
        amf = df.groupby('author_id')['modal_field_code'].first()
        author_modal_field = amf.dropna().to_dict()
        print(f'  Modal field fallback available for {len(author_modal_field):,} authors')

    if not skip_normalized and field_expected_annual is not None:
        print('  Computing normalized citation breakdown...')
        annual_norm = compute_normalized_citation_breakdown(
            edges, df, paper_fields, field_expected_annual,
            author_modal_field=author_modal_field)
        print(f'    {len(annual_norm):,} author-year rows')

    # ── Normalized productivity ──
    annual_prod = pd.DataFrame()
    annual_prod_adj = pd.DataFrame()
    if not skip_normalized and field_expected_pubs is not None:
        print('  Computing normalized productivity...')
        author_papers = query_author_papers(client, author_ids)
        print(f'    {len(author_papers):,} author-paper rows')

        # Get field codes for papers not in edges (zero-citation papers)
        edge_pub_ids = set(pub_ids_from_edges)
        extra_pub_ids = set(author_papers['pub_id'].unique()) - edge_pub_ids
        if extra_pub_ids:
            print(f'    Querying field codes for {len(extra_pub_ids):,} '
                  f'additional papers (no citations)...')
            extra_fields = query_paper_fields(client, list(extra_pub_ids))
            if not extra_fields.empty:
                paper_fields = pd.concat([paper_fields, extra_fields],
                                         ignore_index=True)
                pub_fields_dict = paper_fields.groupby('pub_id')['field_code'] \
                    .apply(list).to_dict()

        fe_pub_dict = field_expected_pubs.set_index(
            ['field_code', 'pub_year']
        )['expected_publications'].to_dict()

        annual_prod, annual_prod_adj = compute_normalized_productivity(
            author_papers, df, pub_fields_dict, fe_pub_dict,
            author_modal_field=author_modal_field)

        if not annual_prod.empty:
            print(f'    {len(annual_prod):,} author-year rows (productivity)')
        if not annual_prod_adj.empty:
            print(f'    {len(annual_prod_adj):,} author-year rows (productivity adj)')

    # ── Cumulative on full year range ──
    print('  Computing cumulative values...')

    # Raw
    annual_raw = annual_raw.sort_values(['author_id', 'year'])
    for col in ['citations_old', 'citations_na']:
        annual_raw[f'cum_{col}'] = annual_raw.groupby('author_id')[col].cumsum()

    # Normalized citations
    if annual_norm is not None and not annual_norm.empty:
        annual_norm = annual_norm.sort_values(['author_id', 'year'])
        for col in ['normalized_citations_old', 'normalized_citations_na']:
            annual_norm[f'cum_{col}'] = annual_norm.groupby('author_id')[col].cumsum()

    # Normalized productivity
    if not annual_prod.empty:
        annual_prod = annual_prod.sort_values(['author_id', 'year'])
        annual_prod['cum_normalized_productivity'] = \
            annual_prod.groupby('author_id')['normalized_productivity'].cumsum()

    # Normalized productivity adj
    if not annual_prod_adj.empty:
        annual_prod_adj = annual_prod_adj.sort_values(['author_id', 'year'])
        annual_prod_adj['cum_normalized_publication_count_adj'] = \
            annual_prod_adj.groupby('author_id')[
                'normalized_publication_count_adj'].cumsum()

    # ── Merge onto panel ──
    print('  Merging onto matched panel...')
    df['year'] = df['year'].astype(int)

    # Helper: drop columns from df if they already exist (avoid _x/_y conflicts)
    def drop_existing(df, cols):
        to_drop = [c for c in cols if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
        return df

    # Raw citations
    annual_raw['year'] = annual_raw['year'].astype(int)
    raw_new_cols = ['citations_old', 'citations_na',
                    'cum_citations_old', 'cum_citations_na']
    df = drop_existing(df, raw_new_cols)
    merge_cols_raw = ['author_id', 'year'] + raw_new_cols
    df = df.merge(annual_raw[merge_cols_raw],
                  on=['author_id', 'year'], how='left')
    for col in raw_new_cols:
        df[col] = df[col].fillna(0).astype(int)

    # Normalized citations
    if annual_norm is not None and not annual_norm.empty:
        annual_norm['year'] = annual_norm['year'].astype(int)
        norm_cit_cols = ['normalized_citations_old', 'normalized_citations_na',
                         'cum_normalized_citations_old', 'cum_normalized_citations_na']
        df = drop_existing(df, norm_cit_cols)
        merge_cols = ['author_id', 'year'] + norm_cit_cols
        df = df.merge(annual_norm[merge_cols],
                      on=['author_id', 'year'], how='left')
        for col in norm_cit_cols:
            df[col] = df[col].fillna(0)

    # Normalized productivity
    if not annual_prod.empty:
        annual_prod['year'] = annual_prod['year'].astype(int)
        prod_cols = ['normalized_productivity', 'cum_normalized_productivity']
        df = drop_existing(df, prod_cols)
        merge_cols = ['author_id', 'year'] + prod_cols
        df = df.merge(annual_prod[merge_cols],
                      on=['author_id', 'year'], how='left')
        for col in prod_cols:
            df[col] = df[col].fillna(0)

    # Normalized productivity adj
    if not annual_prod_adj.empty:
        annual_prod_adj['year'] = annual_prod_adj['year'].astype(int)
        prod_adj_cols = ['normalized_publication_count_adj',
                         'cum_normalized_publication_count_adj']
        df = drop_existing(df, prod_adj_cols)
        merge_cols = ['author_id', 'year'] + prod_adj_cols
        df = df.merge(annual_prod_adj[merge_cols],
                      on=['author_id', 'year'], how='left')
        for col in prod_adj_cols:
            df[col] = df[col].fillna(0)

    # ── Save ──
    df.to_csv(output_path, index=False, sep=';')
    new_cols = [c for c in df.columns
                if any(x in c for x in ['citations_old', 'citations_na',
                                         'normalized_productivity',
                                         'normalized_publication_count_adj'])]
    print(f'\n  Saved: {output_path}')
    print(f'    {df["author_id"].nunique():,} authors, {len(df):,} rows')
    print(f'    New/relevant columns: {new_cols}')

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Enrich matched panels with normalized citation, '
                    'productivity & funding metrics')

    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: input_dir/enriched_normalized)')
    ap.add_argument('--project', type=str, default='ccnr-success')
    ap.add_argument('--skip_normalized', action='store_true',
                    help='Only compute raw citations_old/na, skip all normalization')
    ap.add_argument('--suffix', type=str, default='_enriched')
    ap.add_argument('--chunk_size', type=int, default=1000)

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    client = get_bq_client(args.project)

    # Load normalization reference tables
    field_expected_annual = None
    field_expected_pubs = None

    if not args.skip_normalized:
        print('\nLoading normalization reference tables...')
        field_expected_annual = query_field_expected_citations_annual(client)
        field_expected_pubs = query_field_expected_publications(client)

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else \
                input_path.parent / 'enriched_normalized'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (input_path.stem + args.suffix + '.csv')
        enrich_one_file(input_path, output_path, client,
                        field_expected_annual=field_expected_annual,
                        field_expected_pubs=field_expected_pubs,
                        skip_normalized=args.skip_normalized)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else \
            input_dir / 'enriched_normalized'
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob('merged_*.csv'))
        print(f'\nFound {len(files)} matched files in {input_dir}')
        print(f'Output directory: {out_dir}')

        for fpath in files:
            if args.suffix in fpath.stem:
                continue
            output_path = out_dir / (fpath.stem + args.suffix + '.csv')
            enrich_one_file(fpath, output_path, client,
                            field_expected_annual=field_expected_annual,
                            field_expected_pubs=field_expected_pubs,
                            skip_normalized=args.skip_normalized)


if __name__ == '__main__':
    main()