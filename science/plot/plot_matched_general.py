#!/usr/bin/env python3
"""
Plot treated vs control trajectories — one figure per metric.
Errorbars (not shading) at each to_year. Writes a log of mean ± CI.

If `l2_field_modal_name` exists, exclude HEP-like fields:
  - Particle and High Energy Physics
  - Synchrotrons and Accelerators
  - Nuclear and Plasma Physics

Rows with NaN in l2_field_modal_name are kept.

Usage:
    python plot_matched.py --file ../../data/matches/merged_physics_Nature.csv --save
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['axes.grid.axis'] = 'y'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

HEP_BROAD_FIELDS = [
    'Particle and High Energy Physics',
    'Synchrotrons and Accelerators',
    'Nuclear and Plasma Physics',
]


def load_merged(path):
    df = pd.read_csv(path, sep=';')
    print(f'Loaded {path}: {len(df):,} rows, {df["author_id"].nunique()} authors')

    # Optional: exclude HEP-like L2 modal fields if the column exists.
    # Keep NaN values.
    if 'l2_field_modal_name' in df.columns:
        n_rows_before = len(df)
        n_authors_before = df['author_id'].nunique()

        df = df[
            df['l2_field_modal_name'].isna() |
            (~df['l2_field_modal_name'].isin(HEP_BROAD_FIELDS))
        ].copy()

        n_rows_after = len(df)
        n_authors_after = df['author_id'].nunique()

        print(
            'Filtered HEP-like L2 fields: '
            f'removed {n_rows_before - n_rows_after:,} rows, '
            f'{n_authors_before - n_authors_after:,} authors'
        )
    else:
        print('l2_field_modal_name not found; keeping all authors')

    return df


def split_treated_control(df):
    if 'matched_to' in df.columns:
        treated_ids = set(df[df['matched_to'].isna()]['author_id'].unique())
        control_ids = set(df[df['matched_to'].notna()]['author_id'].unique())
    else:
        treated_ids = set(df[df['is_venue'] == 1]['author_id'].unique())
        control_ids = set(df['author_id'].unique()) - treated_ids
    return treated_ids, control_ids


def compute_stats(df, col, author_ids):
    sub = df[df['author_id'].isin(author_ids)].copy()
    sub[col] = pd.to_numeric(sub[col], errors='coerce')
    rows = []
    for ty in sorted(sub['to_year'].unique()):
        vals = sub.loc[sub['to_year'] == ty, col].dropna()
        n = len(vals)
        if n == 0:
            rows.append({'to_year': ty, 'mean': np.nan, 'se': 0, 'n': 0})
            continue
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        rows.append({'to_year': ty, 'mean': m, 'se': se, 'n': n})
    out = pd.DataFrame(rows)
    out['ci_lo'] = out['mean'] - 1.96 * out['se']
    out['ci_hi'] = out['mean'] + 1.96 * out['se']
    return out


def plot_one(t_stats, c_stats, ylabel, out_path=None):
    # Fixed axes size: 2 x 2 inches for the plot area itself.
    ax_w, ax_h = 2, 2
    left_margin, bottom_margin = 0.75, 0.55
    right_margin, top_margin = 0.15, 0.15

    fw = left_margin + ax_w + right_margin
    fh = bottom_margin + ax_h + top_margin
    fig = plt.figure(figsize=(fw, fh), dpi=600)
    ax = fig.add_axes([left_margin / fw, bottom_margin / fh,
                       ax_w / fw, ax_h / fh])

    # Control first
    ax.errorbar(c_stats['to_year'], c_stats['mean'],
                yerr=1.96 * c_stats['se'],
                color='darkgray', fmt='none',
                capsize=3, capthick=0.96, elinewidth=0.96)
    ax.plot(c_stats['to_year'], c_stats['mean'],
            '-o', color='darkgray', markersize=4, linewidth=2,
            alpha=0.5, label='Control individuals')

    # Treated on top
    ax.errorbar(t_stats['to_year'], t_stats['mean'],
                yerr=1.96 * t_stats['se'],
                color='yellowgreen', fmt='none',
                capsize=3, capthick=0.96, elinewidth=0.96)
    ax.plot(t_stats['to_year'], t_stats['mean'],
            '-o', color='yellowgreen', markersize=4, linewidth=2,
            alpha=0.5, label='Venue access individuals')

    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time to publish')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-5.5, 10.5)

    if out_path:
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=600)
        fig.savefig(out_path.with_suffix('.pdf'))
        plt.close(fig)
        print(f'  Saved: {out_path}')
        print(f'  Saved: {out_path.with_suffix(".pdf")}')
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--output_dir', default='../../figure')
    ap.add_argument('--log_file', default='../../figure/plot_log.csv')
    args = ap.parse_args()

    df = load_merged(args.file)
    treated_ids, control_ids = split_treated_control(df)

    stem = Path(args.file).stem
    label = stem.replace('merged_', '')

    out_dir = Path(args.output_dir)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ('cum_citations_na', '# Citations'),
        ('cum_citations_old', '# Pre-venue Work Citations'),
        ('cum_publication_count_adj', '# Productivity'),
        ('cum_funding_count', '# Grants'),
    ]

    log_rows = []

    for col, ylabel in metrics:
        if col not in df.columns:
            print(f'  WARNING: {col} not found, skipping')
            continue

        t_stats = compute_stats(df, col, treated_ids)
        c_stats = compute_stats(df, col, control_ids)

        safe_col = col.replace('cum_', '')
        out_path = out_dir / f'{label}_{safe_col}.png' if args.save else None
        plot_one(t_stats, c_stats, ylabel, out_path)

        for _, r in t_stats.iterrows():
            log_rows.append({
                'file': label, 'metric': col, 'group': 'treated',
                'to_year': int(r['to_year']), 'mean': r['mean'],
                'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'], 'n': int(r['n'])
            })
        for _, r in c_stats.iterrows():
            log_rows.append({
                'file': label, 'metric': col, 'group': 'control',
                'to_year': int(r['to_year']), 'mean': r['mean'],
                'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'], 'n': int(r['n'])
            })

    log_path = Path(args.log_file)
    log_df = pd.DataFrame(log_rows)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        existing = existing[existing['file'] != label]
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f'  Log: {log_path} ({len(log_df)} rows)')


if __name__ == '__main__':
    main()