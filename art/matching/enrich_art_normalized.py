#!/usr/bin/env python3
"""
==============================================================================
Enrich matched art panels with normalized exhibition metrics
==============================================================================

ANALOGY TO SCIENCE NORMALIZATION:
  In science, we normalize citations by field and year because citation
  norms differ across fields and time periods.

  In art, the analogous confound is temporal inflation in the exhibition
  ecosystem. The number of galleries, biennials, and art fairs has exploded
  since the 1990s. An artist getting 2 solo exhibitions in 1985 is very
  different from getting 2 in 2015.

NORMALIZATION:
  For each exhibition of type X (S/G/B/F) that an artist has in year t:
    E(X, t) = avg exhibitions of type X per active artist in year t
    normalized_contribution = 1 / E(X, t)

  Sum across exhibitions per artist per year → normalized annual count.
  Cumulative = running sum over all years.

  For S_titles (title reuse): same logic. Each title-reuse match gets
  weighted by 1/E(S, t), since the opportunity to have a title-reusing
  solo exhibition scales with the number of solo exhibitions available.

COLUMNS ADDED:
  - normalized_S / cum_normalized_S   (solo exhibitions)
  - normalized_G / cum_normalized_G   (group exhibitions)
  - normalized_B / cum_normalized_B   (biennials)
  - normalized_F / cum_normalized_F   (art fairs)
  - normalized_s_titles / cum_normalized_s_titles  (title reuse)

PREREQUISITE:
  The input panel must already have columns like:
    S, G, B, F (annual counts) or cum_S, cum_G, etc.
  and optionally s_titles / S_titles from enrich_titles.py.

  The artfacts exhibition dataset is needed to compute expected values.

Usage:
    python enrich_art_normalized.py \
        --input_dir ../../data/matches \
        --output_dir ../../data/matches/enriched_normalized \
        --artfacts ../../data/artist_info/artfacts_artists_all.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Config
# =============================================================================

ARTFACTS_PATH_DEFAULT = '../../data/artist_info/artfacts_artists_all.csv'

# Exhibition types to normalize
EXHIBITION_TYPES = ['S', 'G', 'B', 'F']


# =============================================================================
# I/O helpers
# =============================================================================

def load_df(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    with open(path, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    return pd.read_csv(path, sep=sep)


def detect_csv_sep(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return ','
    with open(path, 'r') as f:
        first_line = f.readline()
    return ';' if ';' in first_line else ','


# =============================================================================
# Build expected exhibitions table from artfacts
# =============================================================================

def build_expected_exhibitions(artfacts_df):
    """
    For each (exhibition_type, year), compute the average number of
    exhibitions per active artist.

    An artist is "active" in year t if they have at least one exhibition
    (of any type) in year t.

    Returns dict: (type, year) -> expected_exhibitions_per_artist
    """
    print('  Building expected exhibitions table...')
    t0 = time.time()

    # Active artists per year (any exhibition type)
    active_per_year = artfacts_df.groupby('end_year')['artist_id'] \
        .nunique().to_dict()

    # Count exhibitions per type per year
    type_year_counts = artfacts_df.groupby(['type', 'end_year']).size() \
        .reset_index(name='n_exhibitions')

    # Expected = n_exhibitions / n_active_artists
    expected = {}
    for _, row in type_year_counts.iterrows():
        etype = row['type']
        year = int(row['end_year'])
        n_exh = row['n_exhibitions']
        n_active = active_per_year.get(year, 0)
        if n_active > 0:
            expected[(etype, year)] = n_exh / n_active

    print(f'    {len(expected):,} (type, year) entries ({time.time() - t0:.1f}s)')

    # Print summary
    for etype in EXHIBITION_TYPES:
        vals = [v for (t, y), v in expected.items() if t == etype]
        if vals:
            print(f'    Type {etype}: mean E={np.mean(vals):.3f}, '
                  f'range [{np.min(vals):.3f}, {np.max(vals):.3f}]')

    return expected


# =============================================================================
# Compute normalized exhibition counts
# =============================================================================

def compute_normalized_exhibitions(artfacts_df, matched_df, expected,
                                   artfacts_index=None):
    """
    For each artist x year, compute normalized counts for each exhibition type.

    Each exhibition of type X in year t contributes 1/E(X, t).
    Sum per artist per year.

    Returns DataFrame with columns:
      artist_id, year, normalized_S, normalized_G, normalized_B, normalized_F
    """
    artist_ids = matched_df['artist_id'].unique()

    if artfacts_index is None:
        artfacts_index = {aid: grp for aid, grp in artfacts_df.groupby('artist_id')}

    records = []
    n_no_expected = 0

    for artist_id in artist_ids:
        aex = artfacts_index.get(str(artist_id))
        if aex is None or aex.empty:
            continue

        for etype in EXHIBITION_TYPES:
            type_exh = aex[aex['type'] == etype]
            if type_exh.empty:
                continue

            for year, year_group in type_exh.groupby('end_year'):
                year = int(year)
                e_val = expected.get((etype, year))
                if e_val is None or e_val <= 0:
                    n_no_expected += 1
                    continue

                n_exh = len(year_group)
                norm_val = n_exh / e_val

                records.append({
                    'artist_id': str(artist_id),
                    'year': year,
                    f'normalized_{etype}': norm_val,
                })

    if n_no_expected > 0:
        print(f'    Skipped {n_no_expected:,} (type, year) combos with no expected value')

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Pivot: each record has only one normalized_X column filled.
    # Group by (artist_id, year) and sum across types.
    norm_cols = [f'normalized_{t}' for t in EXHIBITION_TYPES]
    for col in norm_cols:
        if col not in df.columns:
            df[col] = 0.0

    annual = df.groupby(['artist_id', 'year'])[norm_cols].sum().reset_index()

    return annual


def compute_normalized_s_titles(matched_df, expected):
    """
    Normalize existing s_titles column using E(S, year).

    Each title-reuse match in year t gets weighted by 1/E(S, t),
    since the opportunity scales with solo exhibition availability.

    Returns DataFrame: artist_id, year, normalized_s_titles
    """
    if 's_titles' not in matched_df.columns:
        print('    s_titles column not found — skipping title reuse normalization')
        return pd.DataFrame()

    records = []
    for _, row in matched_df.iterrows():
        s_titles_val = row.get('s_titles', 0)
        if pd.isna(s_titles_val) or s_titles_val == 0:
            continue

        year = int(row['year'])
        e_val = expected.get(('S', year))
        if e_val is None or e_val <= 0:
            continue

        records.append({
            'artist_id': str(row['artist_id']),
            'year': year,
            'normalized_s_titles': s_titles_val / e_val,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    annual = df.groupby(['artist_id', 'year'])['normalized_s_titles'] \
        .sum().reset_index()

    return annual


# =============================================================================
# Enrich one file
# =============================================================================

def enrich_one_file(input_path, output_path, artfacts_df, expected,
                    artfacts_index=None):

    print(f'\n{"="*60}')
    print(f'Enriching: {input_path}')
    print(f'{"="*60}')

    input_path = Path(input_path)
    output_path = Path(output_path)
    sep = detect_csv_sep(input_path)

    df = load_df(input_path)
    print(f'  Loaded: {len(df):,} rows, {df["artist_id"].nunique():,} artists')

    df['artist_id'] = df['artist_id'].astype(str)
    df['year'] = df['year'].astype(int)

    # ── Normalized exhibition counts ──
    print('  Computing normalized exhibition counts...')
    annual_exh = compute_normalized_exhibitions(
        artfacts_df, df, expected, artfacts_index=artfacts_index)

    if not annual_exh.empty:
        print(f'    {len(annual_exh):,} artist-year rows')

        # Cumulative
        annual_exh = annual_exh.sort_values(['artist_id', 'year'])
        for etype in EXHIBITION_TYPES:
            col = f'normalized_{etype}'
            if col in annual_exh.columns:
                annual_exh[f'cum_normalized_{etype}'] = \
                    annual_exh.groupby('artist_id')[col].cumsum()

    # ── Normalized title reuse ──
    print('  Computing normalized title reuse...')
    annual_titles = compute_normalized_s_titles(df, expected)

    if not annual_titles.empty:
        print(f'    {len(annual_titles):,} artist-year rows')
        annual_titles = annual_titles.sort_values(['artist_id', 'year'])
        annual_titles['cum_normalized_s_titles'] = \
            annual_titles.groupby('artist_id')['normalized_s_titles'].cumsum()

    # ── Merge onto panel ──
    print('  Merging onto matched panel...')

    # Exhibition counts
    if not annual_exh.empty:
        annual_exh['year'] = annual_exh['year'].astype(int)
        annual_exh['artist_id'] = annual_exh['artist_id'].astype(str)

        merge_cols = ['artist_id', 'year']
        for etype in EXHIBITION_TYPES:
            merge_cols.extend([f'normalized_{etype}', f'cum_normalized_{etype}'])
        merge_cols = [c for c in merge_cols if c in annual_exh.columns]

        df = df.merge(annual_exh[merge_cols], on=['artist_id', 'year'], how='left')

        for etype in EXHIBITION_TYPES:
            for prefix in ['normalized_', 'cum_normalized_']:
                col = f'{prefix}{etype}'
                if col in df.columns:
                    df[col] = df[col].fillna(0)

    # Title reuse
    if not annual_titles.empty:
        annual_titles['year'] = annual_titles['year'].astype(int)
        annual_titles['artist_id'] = annual_titles['artist_id'].astype(str)

        merge_cols = ['artist_id', 'year',
                      'normalized_s_titles', 'cum_normalized_s_titles']
        df = df.merge(annual_titles[merge_cols],
                      on=['artist_id', 'year'], how='left')
        df['normalized_s_titles'] = df['normalized_s_titles'].fillna(0)
        df['cum_normalized_s_titles'] = df['cum_normalized_s_titles'].fillna(0)

    # ── Save ──
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, sep=sep)

    pq_path = output_path.with_suffix('.parquet')
    df.to_parquet(pq_path, index=False)

    new_cols = [c for c in df.columns if 'normalized_' in c and
                any(x in c for x in ['_S', '_G', '_B', '_F', '_s_titles'])]
    print(f'\n  Saved: {csv_path}')
    print(f'    {df["artist_id"].nunique():,} artists, {len(df):,} rows')
    print(f'    New columns: {new_cols}')

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Enrich matched art panels with normalized exhibition metrics')

    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Default: ../../data/matches/enriched_normalized')
    ap.add_argument('--artfacts', type=str, default=ARTFACTS_PATH_DEFAULT)
    ap.add_argument('--suffix', type=str, default='_enriched')

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    # ── Load artfacts ──
    artfacts_path = Path(args.artfacts)
    print(f'Loading artfacts: {artfacts_path}')
    t0 = time.time()

    if artfacts_path.suffix == '.parquet':
        artfacts_df = pd.read_parquet(artfacts_path)
    else:
        artfacts_df = pd.read_csv(artfacts_path)

    if 'end_year' in artfacts_df.columns:
        artfacts_df['end_year'] = pd.to_numeric(
            artfacts_df['end_year'], errors='coerce')
    elif 'end_date' in artfacts_df.columns:
        artfacts_df['end_year'] = pd.to_datetime(
            artfacts_df['end_date'], errors='coerce').dt.year

    artfacts_df = artfacts_df.dropna(subset=['end_year'])
    artfacts_df['end_year'] = artfacts_df['end_year'].astype(int)
    artfacts_df['artist_id'] = artfacts_df['artist_id'].astype(str)

    print(f'  {len(artfacts_df):,} exhibitions, '
          f'{artfacts_df["artist_id"].nunique():,} artists '
          f'({time.time() - t0:.1f}s)')

    # ── Build expected exhibitions ──
    expected = build_expected_exhibitions(artfacts_df)

    # ── Build artfacts index ──
    print('  Building artfacts index...')
    t1 = time.time()
    artfacts_index = {aid: grp for aid, grp in artfacts_df.groupby('artist_id')}
    print(f'  Index built ({time.time() - t1:.1f}s)')

    # ── Process files ──
    default_out = '../../data/matches/enriched_normalized'

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else Path(default_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (input_path.stem + args.suffix + '.csv')
        enrich_one_file(input_path, output_path, artfacts_df, expected,
                        artfacts_index=artfacts_index)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else Path(default_out)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            list(input_dir.glob('matched_*.csv')) +
            list(input_dir.glob('matched_*.parquet'))
        )
        # Deduplicate: prefer parquet
        seen_stems = {}
        for f in files:
            stem = f.stem
            if stem not in seen_stems or f.suffix == '.parquet':
                seen_stems[stem] = f
        files = sorted(seen_stems.values())

        print(f'\nFound {len(files)} matched files in {input_dir}')
        print(f'Output directory: {out_dir}')

        for fpath in files:
            if args.suffix in fpath.stem:
                continue
            output_path = out_dir / (fpath.stem + args.suffix + '.csv')
            enrich_one_file(fpath, output_path, artfacts_df, expected,
                            artfacts_index=artfacts_index)

    print('\n' + '=' * 60)
    print('ALL DONE')
    print('=' * 60)


if __name__ == '__main__':
    main()