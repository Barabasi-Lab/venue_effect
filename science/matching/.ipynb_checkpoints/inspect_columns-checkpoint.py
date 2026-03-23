#!/usr/bin/env python3
"""
Quick column inspection for treated_df and control_df.
Run this first to decide which columns to use for matching.

Usage:
    python inspect_columns.py --field physics
"""

import argparse
from pathlib import Path
import pyarrow.dataset as ds
import pandas as pd

DATA_DIR = Path('../../data/matching_needed')

parser = argparse.ArgumentParser()
parser.add_argument('--field', default='physics')
args = parser.parse_args()

field = args.field
t = ds.dataset(str(DATA_DIR / f'treated_df_{field}.parquet'), format='parquet').to_table().to_pandas()
c = ds.dataset(str(DATA_DIR / f'control_df_{field}.parquet'), format='parquet').to_table().to_pandas()

print(f'=== {field.upper()} ===')
print(f'Treated: {t["author_id"].nunique():,} authors, {len(t):,} rows')
print(f'Control: {c["author_id"].nunique():,} authors, {len(c):,} rows')

print(f'\n--- Treated columns ({len(t.columns)}) ---')
for col in t.columns:
    print(f'  {col:<45s}  {t[col].dtype}  nulls={t[col].isna().sum():,}  e.g. {t[col].dropna().iloc[0] if t[col].notna().any() else "ALL NULL"}')

print(f'\n--- Control columns ({len(c.columns)}) ---')
for col in c.columns:
    print(f'  {col:<45s}  {c[col].dtype}  nulls={c[col].isna().sum():,}  e.g. {c[col].dropna().iloc[0] if c[col].notna().any() else "ALL NULL"}')

print(f'\n--- Columns only in treated ---')
print(set(t.columns) - set(c.columns))

print(f'\n--- Columns only in control ---')
print(set(c.columns) - set(t.columns))

print(f'\n--- Shared columns ---')
shared = sorted(set(t.columns) & set(c.columns))
print(shared)

# Sample a treated author to see their panel
sample_id = t.drop_duplicates('author_id').iloc[0]['author_id']
print(f'\n--- Sample treated author: {sample_id} ---')
print(t[t['author_id'] == sample_id].sort_values('year').to_string())
