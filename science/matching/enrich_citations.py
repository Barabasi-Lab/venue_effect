#!/usr/bin/env python3
"""
==============================================================================
Enrich matched panels with pre-venue & non-venue citation metrics
==============================================================================

PREREQUISITE: Run build_citation_tables.sql in BigQuery console first.
This creates tables in ccnr-success.venue_effect:
  - author_paper_years:             author_id, pub_id, pub_year
  - author_paper_citation_edges:    pub_id, citing_pub_id, citing_year
  - paper_field_codes:              pub_id, field_code
  - field_expected_citations:       field_code, pub_year, expected_citations

KEY: Uses COUNT(DISTINCT citing_pub_id) per author per year — matching
the original Step B method (COUNT(DISTINCT cit.id)).

"old" = citations to papers published strictly BEFORE venue_year
        (ALL such papers in the author's career, not just the panel window)
"na"  = citations to papers NOT published IN venue_year

Columns added:
  - citations_old / cum_citations_old
  - citations_na  / cum_citations_na
  - normalized_citations_old / cum_normalized_citations_old
  - normalized_citations_na  / cum_normalized_citations_na

Usage:
    python enrich_citations.py \
        --input ../../data/matches/merged_physics_Nature.csv \
        --skip_normalized

    python enrich_citations.py \
        --input_dir ../../data/matches
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
# BigQuery queries
# =============================================================================

def get_bq_client(project):
    return bigquery.Client(project=project)


def query_citation_edges(client, author_ids, chunk_size=1000):
    """
    For each author's papers, get individual citation edges:
      author_id, pub_id, pub_year, citing_pub_id, citing_year

    NOT aggregated — we need citing_pub_id for deduplication.
    Covers ALL papers by the author (not just the panel window).
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
    """Get ANZSRC field codes per paper from pre-built table."""
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


def query_field_expected_citations(client):
    """Load field expected citations from pre-built table."""
    query = f"""
    SELECT field_code, pub_year, expected_citations
    FROM `{DS}.field_expected_citations`
    """
    print('    Loading field expected citations...', end=' ', flush=True)
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
    return df


# =============================================================================
# Compute citation breakdown (raw) — COUNT(DISTINCT citing_pub_id)
# =============================================================================

def compute_citation_breakdown(edges, matched_df):
    """
    For each author x citing_year, count DISTINCT citing papers that cite:
      - 'old' papers: published strictly BEFORE venue_year (ALL such papers)
      - 'na' papers:  published in any year EXCEPT venue_year

    Uses the same deduplication as Step B: COUNT(DISTINCT cit.id).
    """
    author_vy = matched_df.groupby('author_id')['venue_year'].first().to_dict()

    e = edges.copy()
    e['venue_year'] = e['author_id'].map(author_vy)
    e = e.dropna(subset=['venue_year'])
    e['venue_year'] = e['venue_year'].astype(int)

    # Tag each edge
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

    # Merge
    annual = citations_old.merge(citations_na, on=['author_id', 'year'], how='outer')
    annual['citations_old'] = annual['citations_old'].fillna(0).astype(int)
    annual['citations_na'] = annual['citations_na'].fillna(0).astype(int)

    return annual


# =============================================================================
# Compute normalized citation breakdown (CNCI)
# =============================================================================

def compute_normalized_breakdown(edges, matched_df, paper_fields, field_expected):
    """
    CNCI-normalized version.
    Each citation edge is weighted by the cited paper's CNCI factor.
    Per distinct citing paper, we take the mean weight across cited papers.
    """
    author_vy = matched_df.groupby('author_id')['venue_year'].first().to_dict()

    e = edges.copy()
    e['venue_year'] = e['author_id'].map(author_vy)
    e = e.dropna(subset=['venue_year'])
    e['venue_year'] = e['venue_year'].astype(int)

    e['is_old'] = (e['pub_year'] < e['venue_year']).astype(int)
    e['is_na']  = (e['pub_year'] != e['venue_year']).astype(int)

    # Paper -> field codes
    pub_fields = paper_fields.groupby('pub_id')['field_code'].apply(list).to_dict()

    # (field_code, pub_year) -> expected_citations
    fe = field_expected.set_index(['field_code', 'pub_year'])['expected_citations'].to_dict()

    # Pre-compute CNCI factor per (pub_id, pub_year)
    paper_cnci_factor = {}
    for pub_id, pub_year in e[['pub_id', 'pub_year']].drop_duplicates().values:
        fields = pub_fields.get(pub_id, [])
        if not fields:
            paper_cnci_factor[(pub_id, int(pub_year))] = 1.0
            continue
        expectations = [fe.get((f, int(pub_year)), None) for f in fields]
        expectations = [e_val for e_val in expectations if e_val and e_val > 0]
        if expectations:
            paper_cnci_factor[(pub_id, int(pub_year))] = 1.0 / np.mean(expectations)
        else:
            paper_cnci_factor[(pub_id, int(pub_year))] = 1.0

    e['cnci_factor'] = e.apply(
        lambda r: paper_cnci_factor.get((r['pub_id'], int(r['pub_year'])), 1.0), axis=1)

    # For old papers: per (author, citing_year, citing_pub_id), mean weight
    old_e = e[e['is_old'] == 1].copy()
    old_weighted = old_e.groupby(['author_id', 'citing_year', 'citing_pub_id']) \
        ['cnci_factor'].mean().reset_index()
    norm_old = old_weighted.groupby(['author_id', 'citing_year'])['cnci_factor'] \
        .sum().reset_index()
    norm_old.columns = ['author_id', 'year', 'normalized_citations_old']

    # For na papers
    na_e = e[e['is_na'] == 1].copy()
    na_weighted = na_e.groupby(['author_id', 'citing_year', 'citing_pub_id']) \
        ['cnci_factor'].mean().reset_index()
    norm_na = na_weighted.groupby(['author_id', 'citing_year'])['cnci_factor'] \
        .sum().reset_index()
    norm_na.columns = ['author_id', 'year', 'normalized_citations_na']

    annual = norm_old.merge(norm_na, on=['author_id', 'year'], how='outer')
    annual['normalized_citations_old'] = annual['normalized_citations_old'].fillna(0)
    annual['normalized_citations_na'] = annual['normalized_citations_na'].fillna(0)

    return annual


# =============================================================================
# Cumulative helper
# =============================================================================

def add_cumulative(df, cols, group_col='author_id', sort_col='year'):
    df = df.sort_values([group_col, sort_col])
    for col in cols:
        df[f'cum_{col}'] = df.groupby(group_col)[col].cumsum()
    return df


# =============================================================================
# Enrich one file
# =============================================================================

def enrich_one_file(input_path, output_path, client,
                    field_expected=None, skip_normalized=False):

    print(f'\n{"="*60}')
    print(f'Enriching: {input_path}')
    print(f'{"="*60}')

    df = pd.read_csv(input_path, sep=';')
    print(f'  Loaded: {len(df):,} rows, {df["author_id"].nunique():,} authors')

    author_ids = list(df['author_id'].unique())

    # ── Query citation edges from pre-built tables ──
    print(f'  Querying citation edges for {len(author_ids):,} authors...')
    edges = query_citation_edges(client, author_ids)
    if edges.empty:
        print('  WARNING: No citation data returned. Skipping.')
        return

    print(f'  Total citation edges: {len(edges):,}')

    # ── Raw breakdown (COUNT DISTINCT citing_pub_id) ──
    print('  Computing raw citation breakdown (old / na)...')
    annual_raw = compute_citation_breakdown(edges, df)
    print(f'    {len(annual_raw):,} author-year rows')

    # ── Normalized breakdown ──
    annual_norm = None
    if not skip_normalized and field_expected is not None:
        print('  Computing normalized citation breakdown...')
        pub_ids = list(edges['pub_id'].unique())
        print(f'    Querying field codes for {len(pub_ids):,} papers...')
        paper_fields = query_paper_fields(client, pub_ids)
        print(f'    {len(paper_fields):,} paper-field rows')

        annual_norm = compute_normalized_breakdown(
            edges, df, paper_fields, field_expected)
        print(f'    {len(annual_norm):,} author-year rows')

    # ── Compute cumulative on FULL year range (not just panel window) ──
    # annual_raw has ALL years (e.g. 1998-2023), not just the panel's [-5,+10].
    # We must cumsum over the full range so that cum_citations_old at to_year=-5
    # includes all citations from before the panel window.
    print('  Computing cumulative on full year range...')

    cum_cols_raw = ['citations_old', 'citations_na']
    annual_raw = annual_raw.sort_values(['author_id', 'year'])
    for col in cum_cols_raw:
        annual_raw[f'cum_{col}'] = annual_raw.groupby('author_id')[col].cumsum()

    annual_norm_cum = None
    if annual_norm is not None and not annual_norm.empty:
        cum_cols_norm = ['normalized_citations_old', 'normalized_citations_na']
        annual_norm = annual_norm.sort_values(['author_id', 'year'])
        for col in cum_cols_norm:
            annual_norm[f'cum_{col}'] = annual_norm.groupby('author_id')[col].cumsum()
        annual_norm_cum = annual_norm

    # ── Merge onto panel (only panel years survive the left join) ──
    print('  Merging onto matched panel...')
    df['year'] = df['year'].astype(int)
    annual_raw['year'] = annual_raw['year'].astype(int)

    # Merge raw annual + cumulative
    merge_cols_raw = ['author_id', 'year', 'citations_old', 'citations_na',
                      'cum_citations_old', 'cum_citations_na']
    df = df.merge(annual_raw[merge_cols_raw], on=['author_id', 'year'], how='left')
    df['citations_old'] = df['citations_old'].fillna(0).astype(int)
    df['citations_na'] = df['citations_na'].fillna(0).astype(int)
    df['cum_citations_old'] = df['cum_citations_old'].fillna(0).astype(int)
    df['cum_citations_na'] = df['cum_citations_na'].fillna(0).astype(int)

    # Merge normalized annual + cumulative
    if annual_norm_cum is not None and not annual_norm_cum.empty:
        annual_norm_cum['year'] = annual_norm_cum['year'].astype(int)
        merge_cols_norm = ['author_id', 'year',
                           'normalized_citations_old', 'normalized_citations_na',
                           'cum_normalized_citations_old', 'cum_normalized_citations_na']
        df = df.merge(annual_norm_cum[merge_cols_norm], on=['author_id', 'year'], how='left')
        df['normalized_citations_old'] = df['normalized_citations_old'].fillna(0)
        df['normalized_citations_na'] = df['normalized_citations_na'].fillna(0)
        df['cum_normalized_citations_old'] = df['cum_normalized_citations_old'].fillna(0)
        df['cum_normalized_citations_na'] = df['cum_normalized_citations_na'].fillna(0)

    # ── Save ──
    df.to_csv(output_path, index=False, sep=';')
    new_cols = [c for c in df.columns if 'old' in c or '_na' in c]
    print(f'\n  Saved: {output_path}')
    print(f'    {df["author_id"].nunique():,} authors, {len(df):,} rows')
    print(f'    New columns: {new_cols}')

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Enrich matched panels with pre-venue citation metrics')

    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: input_dir/enriched_citations)')
    ap.add_argument('--project', type=str, default='ccnr-success')
    ap.add_argument('--skip_normalized', action='store_true')
    ap.add_argument('--suffix', type=str, default='_enriched')
    ap.add_argument('--chunk_size', type=int, default=1000)

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    client = get_bq_client(args.project)

    field_expected = None
    if not args.skip_normalized:
        print('\nLoading field expected citations...')
        field_expected = query_field_expected_citations(client)

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else \
                input_path.parent / 'enriched_citations'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (input_path.stem + args.suffix + '.csv')
        enrich_one_file(input_path, output_path, client,
                        field_expected=field_expected,
                        skip_normalized=args.skip_normalized)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else \
            input_dir / 'enriched_citations'
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob('merged_*.csv'))
        print(f'\nFound {len(files)} matched files in {input_dir}')
        print(f'Output directory: {out_dir}')

        for fpath in files:
            if args.suffix in fpath.stem:
                continue
            output_path = out_dir / (fpath.stem + args.suffix + '.csv')
            enrich_one_file(fpath, output_path, client,
                            field_expected=field_expected,
                            skip_normalized=args.skip_normalized)


if __name__ == '__main__':
    main()