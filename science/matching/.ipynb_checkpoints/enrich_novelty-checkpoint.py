#!/usr/bin/env python3
"""
==============================================================================
Enrich matched panels with novelty metrics (Uzzi et al. 2013 / Tian et al. 2025)
==============================================================================

PREREQUISITE: Run build_novelty_tables.sql in BigQuery console first.
This creates tables in ccnr-success.venue_effect:
  - author_paper_novelty:          author_id, pub_id, pub_year,
                                   Atyp_10pct_Z, Atyp_Median_Z, Atyp_Pairs
  - year_novelty_thresholds:       pub_year, median_atyp_median_z, ...
  - field_year_novelty_thresholds: field_code, pub_year, median_atyp_median_z, ...
  - author_modal_field:            author_id, modal_field_code

Novelty definition (Tian et al. 2025 PNAS, following Uzzi et al. 2013):
  A paper is "novel" if BOTH:
    (a) Atyp_Median_Z > year-level median of Atyp_Median_Z  (high conventionality)
    (b) Atyp_10pct_Z <= 0                                   (high tail novelty)

Columns added:
  - n_papers_total          : total papers by author in that year
  - n_papers_with_novelty   : papers with SciSciNet novelty scores
  - n_novel_papers           : papers meeting the novelty definition (global threshold)
  - pct_novel                : n_novel_papers / n_papers_with_novelty (global)
  - n_novel_papers_field     : papers meeting the novelty definition (field threshold)
  - pct_novel_field          : n_novel_papers_field / n_papers_with_novelty
  - mean_atyp_median_z       : mean Atyp_Median_Z across papers that year
  - mean_atyp_10pct_z        : mean Atyp_10pct_Z across papers that year
  - cum_n_novel_papers       : cumulative novel papers (global)
  - cum_n_papers_with_novelty: cumulative papers with scores
  - cum_pct_novel            : cumulative % novel (global)
  - cum_n_novel_papers_field : cumulative novel papers (field)
  - cum_pct_novel_field      : cumulative % novel (field)

"old" variants (pre-venue papers only):
  - n_novel_old / pct_novel_old / cum_n_novel_old / cum_pct_novel_old
  - n_novel_old_field / pct_novel_old_field

Usage:
    python enrich_novelty.py \\
        --input ../../data/matches/merged_physics_Nature.csv

    python enrich_novelty.py \\
        --input_dir ../../data/matches

    python enrich_novelty.py \\
        --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \\
        --output_dir ../../data/matches/enriched_novelty
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


def query_author_paper_novelty(client, author_ids, chunk_size=1000):
    """
    For each author's papers, get SciSciNet novelty scores.
    Returns: author_id, pub_id, pub_year, Atyp_10pct_Z, Atyp_Median_Z, Atyp_Pairs
    Covers ALL papers by the author (not just the panel window).
    """
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT
            author_id,
            pub_id,
            pub_year,
            Atyp_10pct_Z,
            Atyp_Median_Z,
            Atyp_Pairs
        FROM `{DS}.author_paper_novelty`
        WHERE author_id IN ({ids_str})
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


def query_author_paper_years(client, author_ids, chunk_size=1000):
    """
    Get ALL papers per author (with or without novelty scores) to compute
    total paper counts per year. This allows us to compute the denominator
    correctly: n_papers_with_novelty vs n_papers_total.
    """
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT author_id, pub_id, pub_year
        FROM `{DS}.author_paper_years`
        WHERE author_id IN ({ids_str})
        """

        df = client.query(query).to_dataframe()
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def query_year_thresholds(client):
    """Load global year-level novelty thresholds."""
    query = f"""
    SELECT pub_year, median_atyp_median_z, n_papers
    FROM `{DS}.year_novelty_thresholds`
    ORDER BY pub_year
    """
    print('    Loading global year thresholds...', end=' ', flush=True)
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
    return df


def query_field_year_thresholds(client):
    """Load field-specific year-level novelty thresholds."""
    query = f"""
    SELECT field_code, pub_year, median_atyp_median_z, n_papers
    FROM `{DS}.field_year_novelty_thresholds`
    ORDER BY field_code, pub_year
    """
    print('    Loading field-year thresholds...', end=' ', flush=True)
    t0 = time.time()
    df = client.query(query).to_dataframe()
    print(f'{len(df):,} rows ({time.time() - t0:.1f}s)')
    return df


def query_author_modal_fields(client, author_ids, chunk_size=5000):
    """Get each author's modal ANZSRC field code."""
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT author_id, modal_field_code
        FROM `{DS}.author_modal_field`
        WHERE author_id IN ({ids_str})
        """

        df = client.query(query).to_dataframe()
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def query_author_venue_pubs(client, author_ids, chunk_size=5000):
    """
    Get each author's venue (focal) paper pub_id from the pre-built
    author_venue_pubs table. Used to precisely exclude venue papers
    from novelty calculations.
    """
    all_results = []
    author_list = sorted(set(author_ids))

    for i in range(0, len(author_list), chunk_size):
        chunk = author_list[i:i + chunk_size]
        ids_str = ', '.join(f"'{aid}'" for aid in chunk)

        query = f"""
        SELECT author_id, venue_pub_id, venue_year, discipline
        FROM `{DS}.author_venue_pubs`
        WHERE author_id IN ({ids_str})
        """

        df = client.query(query).to_dataframe()
        all_results.append(df)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)

def compute_novelty_metrics(novelty_df, all_papers_df, matched_df,
                            year_thresholds, field_year_thresholds,
                            author_modal_fields,
                            venue_pub_ids=None,
                            min_coverage=0.0):
    """
    For each author × year, compute:
      - Paper counts (total, with novelty data)
      - Novel paper counts (global and field-specific thresholds)
      - Percentage novel
      - Mean z-scores
      - "old" variants for pre-venue papers

    IMPORTANT: Excludes the venue (focal) paper from all calculations,
    consistent with the citations pipeline excluding focal paper citations.

    Novelty definition (Tian et al. 2025):
      Novel = 1 if Atyp_Median_Z > year_median AND Atyp_10pct_Z <= 0

    Args:
        venue_pub_ids: set of pub_ids to exclude (venue/focal papers).
                       Queried from author_venue_pubs table.
        min_coverage: minimum ratio of n_papers_with_novelty / n_papers_total
                      required to compute pct_novel. Below this, pct_novel = NaN.
                      Default 0.0 (no filter). Set to e.g. 0.3 for stricter filtering.
    """

    # ── Prepare lookups ──
    author_vy = matched_df.groupby('author_id')['venue_year'].first().to_dict()

    # Venue paper exclusion set
    if venue_pub_ids is None:
        venue_pub_ids = set()

    print(f'    Venue papers to exclude: {len(venue_pub_ids):,}')

    # Global threshold: year -> median_atyp_median_z
    global_thresh = year_thresholds.set_index('pub_year')['median_atyp_median_z'].to_dict()

    # Field threshold: (field_code, pub_year) -> median_atyp_median_z
    field_thresh = field_year_thresholds.set_index(
        ['field_code', 'pub_year'])['median_atyp_median_z'].to_dict()

    # Author -> modal field
    author_field = author_modal_fields.set_index('author_id')['modal_field_code'].to_dict()

    # ── Tag novelty scores with thresholds ──
    n = novelty_df.copy()
    n['pub_year'] = n['pub_year'].astype(int)

    # Global novelty flag
    n['global_threshold'] = n['pub_year'].map(global_thresh)
    n['is_novel_global'] = (
        (n['Atyp_Median_Z'] > n['global_threshold']) &
        (n['Atyp_10pct_Z'] <= 0)
    ).astype(int)

    # Field-specific novelty flag (using author's modal field)
    n['modal_field'] = n['author_id'].map(author_field)
    n['field_threshold'] = n.apply(
        lambda r: field_thresh.get((r['modal_field'], r['pub_year']), np.nan)
        if pd.notna(r.get('modal_field')) else np.nan,
        axis=1
    )
    n['is_novel_field'] = (
        (n['Atyp_Median_Z'] > n['field_threshold']) &
        (n['Atyp_10pct_Z'] <= 0) &
        (n['field_threshold'].notna())
    ).astype(int)

    # Tag pre-venue papers and venue paper exclusion
    n['venue_year'] = n['author_id'].map(author_vy)
    n['is_old'] = (n['pub_year'] < n['venue_year']).astype(int)

    # ── Exclude venue (focal) paper ──
    if venue_pub_ids:
        # Precise exclusion: remove only the specific venue paper(s)
        n['is_venue_paper'] = n['pub_id'].isin(venue_pub_ids).astype(int)
    else:
        # No venue pub IDs available — exclude ALL papers from venue_year
        # as a conservative fallback
        n['is_venue_paper'] = (n['pub_year'] == n['venue_year']).astype(int)

    n_before = len(n)
    n = n[n['is_venue_paper'] == 0]
    print(f'    Excluded {n_before - len(n):,} venue paper rows '
          f'({(n_before - len(n)) / max(n_before, 1) * 100:.1f}%)')

    # Also exclude venue papers from total paper counts
    if venue_pub_ids:
        all_papers_excl = all_papers_df[~all_papers_df['pub_id'].isin(venue_pub_ids)].copy()
    else:
        all_papers_excl = all_papers_df.copy()
        all_papers_excl['venue_year'] = all_papers_excl['author_id'].map(author_vy)
        all_papers_excl = all_papers_excl[
            all_papers_excl['pub_year'].astype(int) != all_papers_excl['venue_year']
        ]

    # ── Aggregate per author-year: ALL non-venue papers (novelty-scored) ──
    annual_novelty = n.groupby(['author_id', 'pub_year']).agg(
        n_papers_with_novelty=('pub_id', 'nunique'),
        n_novel_papers=('is_novel_global', 'sum'),
        n_novel_papers_field=('is_novel_field', 'sum'),
        mean_atyp_median_z=('Atyp_Median_Z', 'mean'),
        mean_atyp_10pct_z=('Atyp_10pct_Z', 'mean'),
    ).reset_index()
    annual_novelty.rename(columns={'pub_year': 'year'}, inplace=True)

    # Percentage novel (with minimum coverage filter)
    annual_novelty['pct_novel'] = np.where(
        annual_novelty['n_papers_with_novelty'] > 0,
        annual_novelty['n_novel_papers'] / annual_novelty['n_papers_with_novelty'],
        np.nan  # NaN when no scored papers, not 0
    )
    annual_novelty['pct_novel_field'] = np.where(
        annual_novelty['n_papers_with_novelty'] > 0,
        annual_novelty['n_novel_papers_field'] / annual_novelty['n_papers_with_novelty'],
        np.nan
    )

    # ── Aggregate per author-year: OLD papers only (pre-venue) ──
    old_n = n[n['is_old'] == 1]
    annual_old = old_n.groupby(['author_id', 'pub_year']).agg(
        n_papers_with_novelty_old=('pub_id', 'nunique'),
        n_novel_old=('is_novel_global', 'sum'),
        n_novel_old_field=('is_novel_field', 'sum'),
    ).reset_index()
    annual_old.rename(columns={'pub_year': 'year'}, inplace=True)

    annual_old['pct_novel_old'] = np.where(
        annual_old['n_papers_with_novelty_old'] > 0,
        annual_old['n_novel_old'] / annual_old['n_papers_with_novelty_old'],
        np.nan
    )
    annual_old['pct_novel_old_field'] = np.where(
        annual_old['n_papers_with_novelty_old'] > 0,
        annual_old['n_novel_old_field'] / annual_old['n_papers_with_novelty_old'],
        np.nan
    )

    # ── Total paper counts (excluding venue paper, consistent with above) ──
    all_p = all_papers_excl.copy()
    all_p['pub_year'] = all_p['pub_year'].astype(int)

    annual_total = all_p.groupby(['author_id', 'pub_year']).agg(
        n_papers_total=('pub_id', 'nunique')
    ).reset_index()
    annual_total.rename(columns={'pub_year': 'year'}, inplace=True)

    # ── Merge everything ──
    annual = annual_total.merge(annual_novelty, on=['author_id', 'year'], how='left')
    annual = annual.merge(annual_old, on=['author_id', 'year'], how='left')

    # Fill NaN with 0 for INTEGER count columns only
    fill_cols = [
        'n_papers_with_novelty', 'n_novel_papers', 'n_novel_papers_field',
        'n_papers_with_novelty_old', 'n_novel_old', 'n_novel_old_field',
    ]
    for col in fill_cols:
        if col in annual.columns:
            annual[col] = annual[col].fillna(0).astype(int)

    # Apply minimum coverage filter: set pct_novel to NaN where
    # too few papers have novelty scores for a reliable estimate
    if min_coverage > 0:
        low_cov = (
            annual['n_papers_with_novelty'] /
            annual['n_papers_total'].clip(lower=1)
        ) < min_coverage
        for col in ['pct_novel', 'pct_novel_field']:
            if col in annual.columns:
                annual.loc[low_cov, col] = np.nan
        n_filtered = low_cov.sum()
        print(f'    Coverage filter ({min_coverage:.0%}): set {n_filtered:,} '
              f'author-year pct_novel values to NaN')

    # NOTE: pct columns keep NaN where no data exists (not filled with 0)
    # This is important: NaN means "no data", 0 means "zero novel papers"
    z_cols = ['mean_atyp_median_z', 'mean_atyp_10pct_z']
    # z-score columns also keep NaN — meaningful to know when no data exists

    # ── Compute cumulative metrics on full year range ──
    annual = annual.sort_values(['author_id', 'year'])

    cum_int_cols = [
        'n_papers_total', 'n_papers_with_novelty',
        'n_novel_papers', 'n_novel_papers_field',
        'n_novel_old', 'n_novel_old_field',
        'n_papers_with_novelty_old',
    ]
    for col in cum_int_cols:
        if col in annual.columns:
            annual[f'cum_{col}'] = annual.groupby('author_id')[col].cumsum()

    # Cumulative pct_novel = cum_n_novel_papers / cum_n_papers_with_novelty
    annual['cum_pct_novel'] = np.where(
        annual['cum_n_papers_with_novelty'] > 0,
        annual['cum_n_novel_papers'] / annual['cum_n_papers_with_novelty'],
        0.0
    )
    annual['cum_pct_novel_field'] = np.where(
        annual['cum_n_papers_with_novelty'] > 0,
        annual['cum_n_novel_papers_field'] / annual['cum_n_papers_with_novelty'],
        0.0
    )

    if 'cum_n_papers_with_novelty_old' in annual.columns:
        annual['cum_pct_novel_old'] = np.where(
            annual['cum_n_papers_with_novelty_old'] > 0,
            annual['cum_n_novel_old'] / annual['cum_n_papers_with_novelty_old'],
            0.0
        )
        annual['cum_pct_novel_old_field'] = np.where(
            annual['cum_n_papers_with_novelty_old'] > 0,
            annual['cum_n_novel_old_field'] / annual['cum_n_papers_with_novelty_old'],
            0.0
        )

    return annual


# =============================================================================
# Enrich one file
# =============================================================================

def enrich_one_file(input_path, output_path, client,
                    year_thresholds, field_year_thresholds,
                    min_coverage=0.0):

    print(f'\n{"="*60}')
    print(f'Enriching novelty: {input_path}')
    print(f'{"="*60}')

    df = pd.read_csv(input_path, sep=';')
    print(f'  Loaded: {len(df):,} rows, {df["author_id"].nunique():,} authors')

    author_ids = list(df['author_id'].unique())

    # ── Query novelty scores from pre-built table ──
    print(f'  Querying novelty scores for {len(author_ids):,} authors...')
    novelty_df = query_author_paper_novelty(client, author_ids)
    if novelty_df.empty:
        print('  WARNING: No novelty data returned. Skipping.')
        return

    print(f'  Total author-paper-novelty rows: {len(novelty_df):,}')

    # ── Query ALL papers per author (for total counts) ──
    print(f'  Querying all papers for {len(author_ids):,} authors...')
    all_papers_df = query_author_paper_years(client, author_ids)
    print(f'  Total author-paper rows: {len(all_papers_df):,}')

    # ── Query modal fields ──
    print(f'  Querying modal fields...')
    author_modal = query_author_modal_fields(client, author_ids)
    print(f'  Authors with modal field: {len(author_modal):,}')

    # ── Query venue (focal) paper IDs for exclusion ──
    print(f'  Querying venue paper IDs for exclusion...')
    venue_pubs_df = query_author_venue_pubs(client, author_ids)
    if not venue_pubs_df.empty:
        venue_pub_ids = set(venue_pubs_df['venue_pub_id'].unique())
        print(f'  Found {len(venue_pub_ids):,} venue papers to exclude')
    else:
        venue_pub_ids = set()
        print('  WARNING: No venue papers found — fallback to year-based exclusion')

    # Coverage diagnostics
    authors_with_novelty = novelty_df['author_id'].nunique()
    papers_with_novelty = novelty_df['pub_id'].nunique()
    total_papers = all_papers_df['pub_id'].nunique()
    print(f'  Coverage: {authors_with_novelty:,}/{len(author_ids):,} authors '
          f'({authors_with_novelty/len(author_ids)*100:.1f}%)')
    print(f'  Coverage: {papers_with_novelty:,}/{total_papers:,} papers '
          f'({papers_with_novelty/total_papers*100:.1f}%)')

    # ── Compute novelty metrics ──
    print('  Computing novelty metrics...')
    annual = compute_novelty_metrics(
        novelty_df, all_papers_df, df,
        year_thresholds, field_year_thresholds,
        author_modal,
        venue_pub_ids=venue_pub_ids,
        min_coverage=min_coverage
    )
    print(f'    {len(annual):,} author-year rows')

    # ── Merge onto panel ──
    print('  Merging onto matched panel...')
    df['year'] = df['year'].astype(int)
    annual['year'] = annual['year'].astype(int)

    # Select columns to merge (avoid duplicating if already present)
    merge_cols = ['author_id', 'year'] + [
        c for c in annual.columns
        if c not in ('author_id', 'year') and c not in df.columns
    ]

    df = df.merge(annual[merge_cols], on=['author_id', 'year'], how='left')

    # Fill count columns with 0 (no papers = zero count)
    int_fill = [c for c in df.columns if c.startswith(('n_novel', 'n_papers', 'cum_n_'))]
    for col in int_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # NOTE: pct_novel columns are NOT filled with 0.
    # NaN means "no novelty-scored papers in this author-year"
    # 0.0 means "papers exist but none are novel"
    # This distinction matters for the DiD — NaN rows should be
    # excluded from the estimation, not treated as zero novelty.

    # ── Save ──
    df.to_csv(output_path, index=False, sep=';')
    new_cols = [c for c in df.columns if 'novel' in c.lower() or 'atyp' in c.lower()]
    print(f'\n  Saved: {output_path}')
    print(f'    {df["author_id"].nunique():,} authors, {len(df):,} rows')
    print(f'    New columns ({len(new_cols)}): {new_cols[:10]}...')

    # Quick sanity check
    panel_years = df[df['n_papers_with_novelty'] > 0]
    if len(panel_years) > 0:
        avg_pct = panel_years['pct_novel'].mean()
        print(f'    Avg pct_novel (where data exists): {avg_pct:.3f} '
              f'({avg_pct*100:.1f}%)')

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Enrich matched panels with novelty metrics (Uzzi/Tian)')

    ap.add_argument('--input', type=str, default=None,
                    help='Single input CSV (semicolon-delimited)')
    ap.add_argument('--output', type=str, default=None,
                    help='Output path (default: auto-generated)')
    ap.add_argument('--input_dir', type=str, default=None,
                    help='Directory of matched panel CSVs')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: input_dir/enriched_novelty)')
    ap.add_argument('--project', type=str, default='ccnr-success')
    ap.add_argument('--suffix', type=str, default='_novelty')
    ap.add_argument('--chunk_size', type=int, default=1000)
    ap.add_argument('--min_coverage', type=float, default=0.0,
                    help='Min ratio of novelty-scored papers to total papers. '
                         'Below this, pct_novel = NaN. Default 0.0 (no filter).')

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    client = get_bq_client(args.project)

    # Load thresholds once (used for all files)
    print('\nLoading novelty thresholds...')
    year_thresholds = query_year_thresholds(client)
    field_year_thresholds = query_field_year_thresholds(client)

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            # Place enriched_novelty/ as sibling to enriched_citations/
            # e.g. data/matches/enriched_citations/foo.csv
            #   -> data/matches/enriched_novelty/foo_novelty.csv
            if args.output_dir:
                out_dir = Path(args.output_dir)
            else:
                out_dir = input_path.parent.parent / 'enriched_novelty'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (input_path.stem + args.suffix + '.csv')
        enrich_one_file(input_path, output_path, client,
                        year_thresholds, field_year_thresholds,
                        min_coverage=args.min_coverage)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        # Place enriched_novelty/ as sibling to the input directory
        # e.g. data/matches/enriched_citations/
        #   -> data/matches/enriched_novelty/
        out_dir = Path(args.output_dir) if args.output_dir else \
            input_dir.parent / 'enriched_novelty'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Look for enriched citation files first, fall back to merged_*
        files = sorted(input_dir.glob('*_enriched.csv'))
        if not files:
            files = sorted(input_dir.glob('merged_*.csv'))
        print(f'\nFound {len(files)} files in {input_dir}')
        print(f'Output directory: {out_dir}')

        for fpath in files:
            if args.suffix in fpath.stem:
                continue  # skip already-enriched novelty files
            output_path = out_dir / (fpath.stem + args.suffix + '.csv')
            enrich_one_file(fpath, output_path, client,
                            year_thresholds, field_year_thresholds,
                            min_coverage=args.min_coverage)


if __name__ == '__main__':
    main()