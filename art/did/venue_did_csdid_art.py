#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Heterogeneous DiD Estimation for ART (Python)
==============================================================================

Adapted from venue_did_csdid.py (science version) for art biennials data.

Key differences from the science version:
  - Outcomes: solo exhibitions (s), group exhibitions (g), art fairs (f),
    biennale participation (b) — already cumulative in the panel
  - Treatment: biennale_treated (1 = participated in top biennial)
  - Gender: gender_est column ("female", "male", "mostly_female", "mostly_male")
            → filtered to only "female" and "male" for gender heterogeneity
  - Age filtering: remove posthumous artists (age_stage != "posthumous")
  - Career stage: venue_career_age based, split at 10 / 25 years
  - Region: pre-computed continent dummies (europe, asia, northamerica,
            southamerica, oceania, africa)
  - Panel structure: year_diff (relative time to biennale) as panel time

Data expectations (from matching pipeline):
  - Each biennial has its own matched CSV file under data/matches/:
      matched_venice_biennale.csv
      matched_documenta.csv
      matched_bienal_sao_paulo.csv
      matched_biennale_of_sydney.csv
      matched_whitney_biennial.csv
      matched_istanbul_biennial.csv
      matched_manifesta.csv
      matched_gwangju_biennale.csv
  - Key columns: artist_id, year_diff, biennale_treated,
    gender_est, age_stage, career_stage,
    europe, asia, northamerica, southamerica, oceania, africa,
    s, g, f, b (cumulative exhibition/fair counts per year),
    venue_year (calendar year of first biennale participation)
  - Treated artists: biennale_treated == 1
  - Control artists: biennale_treated == 0

Usage:
    # Single biennial file
    python venue_did_csdid_art.py \\
        --input ../../data/matches/matched_venice_biennale.csv

    # With heterogeneity analysis
    python venue_did_csdid_art.py \\
        --input ../../data/matches/matched_venice_biennale.csv \\
        --outcomes S G F \\
        --heterogeneity gender career_stage region

    # All biennials in directory
    python venue_did_csdid_art.py \\
        --input_dir ../../data/matches/ \\
        --outcomes S G F \\
        --heterogeneity gender career_stage region cohort

Requirements:
    pip install csdid pandas numpy scipy
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from csdid.att_gt import ATTgt
except ImportError:
    raise ImportError(
        "Install csdid: pip install csdid\n"
        "See: https://d2cml-ai.github.io/csdid/"
    )

warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================

# Default outcomes for art
# Columns are uppercase: S (solo), G (group), F (fair), B (biennale)
DEFAULT_OUTCOMES = [
    'S',   # solo exhibitions (cumulative)
    'G',   # group exhibitions (cumulative)
    'F',   # art fair participation (cumulative)
]

# --------------------------------------------------------------------------
# Covariates rationale:
#
# The matching already blocks on gender (exact), continent (overlap),
# birth year (±5yr), and CEM on cumulative S/G/B/F at year_diff=-1.
# So gender, region, and career-age proxies are balanced by design.
#
# The DiD covariates provide additional control for residual imbalance.
# We include:
#   gender_code    : integer-coded (0=female, 1=male)
#   career_code    : integer-coded (0=early, 1=mid, 2=late)
#   6 continent dummies (same as Stata specification)
#   biennale_decade: decade of biennale access — NOT matched on,
#                    provides additional control for cohort trends
#                    (same as venue_decade in the science version)
#
# For heterogeneity: drop the splitting variable + biennale_decade
# for region (small subgroups cause singularity with decade).
# --------------------------------------------------------------------------

# Continent dummy columns — renamed to remove spaces for csdid formula compatibility
CONTINENT_COLS_RAW = ['Europe', 'Asia', 'Africa', 'Oceania',
                      'North America', 'South America']
CONTINENT_RENAME = {
    'North America': 'NorthAmerica',
    'South America': 'SouthAmerica',
}
CONTINENT_COLS = ['Europe', 'Asia', 'Africa', 'Oceania',
                  'NorthAmerica', 'SouthAmerica']

COVARIATE_COLS = [
    'gender_code',          # integer-coded gender
    'career_code',          # integer-coded career stage
    'Europe',               # continent dummy
    'Asia',                 # continent dummy
    'Africa',               # continent dummy
    'Oceania',              # continent dummy
    'NorthAmerica',         # continent dummy
    'SouthAmerica',         # continent dummy
    'biennale_decade',      # decade of biennale (not matched on → extra control)
]

# Which covariates to DROP for each heterogeneity type
# - Always drop the splitting variable (constant within subgroup)
# - For region: also drop biennale_decade (near-constant in small geo subgroups)
COVARIATES_TO_DROP = {
    'gender':       ['gender_code'],
    'career_stage': ['career_code'],
    'region':       CONTINENT_COLS + ['biennale_decade'],
}

# Geography groups for region heterogeneity
# geo_group uses the RENAMED column names (no spaces)
GEO_GROUPS = [
    'Europe',
    'Asia',
    'Africa',
    'Oceania',
    'NorthAmerica',
    'SouthAmerica',
]

# Map from career_stage label to integer code
CAREER_CODE_MAP = {
    'early-career': 0,
    'mid-career': 1,
    'late-career': 2,
}

# Columns that are genuinely numeric where NaN means 0
NUMERIC_ZERO_FILL_COLS = [
    'S', 'G', 'F', 'B', 'B_adj',
    'biennale_decade',
]

CAREER_STAGE_DEFS = {
    'early-career':  (0, 10),   # career_age <= 10
    'mid-career':    (11, 25),  # 10 < career_age <= 25
    'late-career':   (26, 999), # career_age > 25
}


# =============================================================================
# Data preparation
# =============================================================================

def load_and_prepare(input_path):
    """
    Load matched panel CSV for art biennials and prepare for csdid.

    Each biennial has its own file (e.g., matched_venice_biennale.csv),
    so no venue_id filtering is needed.

    KEY DESIGN (same as science version):
    We use `year_diff` (relative time) shifted to positive integers
    as the panel time variable. year_diff ranges from -5 to +10.
    panel_time = year_diff + 6  (range 1..16)
    Treatment:  first_treat = 6 for treated, 0 for never-treated

    Args:
        input_path: Path to matched panel CSV

    Returns:
        df: prepared DataFrame
        meta: dict with metadata
    """
    print(f'\n  Loading: {input_path}')

    # Try comma first, then semicolon
    try:
        df = pd.read_csv(input_path, sep=',')
        if len(df.columns) <= 2:
            df = pd.read_csv(input_path, sep=';')
    except Exception:
        df = pd.read_csv(input_path, sep=';')

    print(f'    {len(df):,} rows, {df["artist_id"].nunique():,} artists')

    # --- Rename columns with spaces (breaks csdid formula parser) ---
    df = df.rename(columns=CONTINENT_RENAME)

    # --- Remove posthumous artists ---
    if 'age_stage' in df.columns:
        n_before = df['artist_id'].nunique()
        # Keep non-posthumous and NaN (missing age info)
        df = df[
            (df['age_stage'] != 'posthumous') |
            df['age_stage'].isna()
        ].copy()
        n_after = df['artist_id'].nunique()
        print(f'    Removed posthumous artists: '
              f'{n_before - n_after:,} artists dropped')

    # --- Map column names ---
    # Handle year_diff vs to_year naming
    if 'year_diff' in df.columns and 'to_year' not in df.columns:
        df['to_year'] = df['year_diff']
    elif 'to_year' in df.columns and 'year_diff' not in df.columns:
        df['year_diff'] = df['to_year']

    # Handle biennale_treated vs is_venue naming
    if 'biennale_treated' in df.columns and 'is_venue' not in df.columns:
        df['is_venue'] = df['biennale_treated']
    elif 'is_venue' in df.columns and 'biennale_treated' not in df.columns:
        df['biennale_treated'] = df['is_venue']

    # --- Ensure required columns ---
    required = ['artist_id', 'to_year']
    # Need either biennale_treated or is_venue
    if 'biennale_treated' not in df.columns and 'is_venue' not in df.columns:
        raise ValueError('Missing treatment column (biennale_treated or is_venue)')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    # --- Create unique unit IDs (keep ALL rows, no dedup) ---
    # A control artist matched to multiple treated artists appears multiple
    # times with different matched_to / biennale_year. Instead of dropping,
    # give each occurrence a unique ID so csdid treats them as separate units.
    # unit_key = artist_id + matched_to + biennale_year
    treat_col = 'biennale_treated' if 'biennale_treated' in df.columns else 'is_venue'
    treated_ids = set(df.loc[df[treat_col] == 1, 'artist_id'].unique())

    if 'matched_to' in df.columns and 'biennale_year' in df.columns:
        df['unit_key'] = (df['artist_id'].astype(str) + '_' +
                          df['matched_to'].astype(str) + '_' +
                          df['biennale_year'].astype(str))
    elif 'matched_to' in df.columns:
        df['unit_key'] = (df['artist_id'].astype(str) + '_' +
                          df['matched_to'].astype(str))
    else:
        df['unit_key'] = df['artist_id'].astype(str)

    unit_keys = df['unit_key'].unique()
    id_map = {uk: i + 1 for i, uk in enumerate(unit_keys)}
    df['artist_int_id'] = df['unit_key'].map(id_map)

    n_units = len(unit_keys)
    n_artists = df['artist_id'].nunique()
    if n_units != n_artists:
        print(f'    Unit keys: {n_artists:,} artists → {n_units:,} units '
              f'(controls reused across treated artists)')

    # --- Identify treated vs control ---
    df['is_treated'] = df['artist_id'].isin(treated_ids).astype(int)

    # --- Create panel_time from to_year ---
    TIME_SHIFT = 6  # to_year=0 → panel_time=6
    df['panel_time'] = df['to_year'].astype(int) + TIME_SHIFT

    # --- Create first_treat ---
    df['first_treat'] = 0
    df.loc[df['is_treated'] == 1, 'first_treat'] = TIME_SHIFT

    # --- Dedup: ensure unique (artist_int_id, panel_time) ---
    dup_check = df.groupby(['artist_int_id', 'panel_time']).size()
    n_dups = (dup_check > 1).sum()
    if n_dups > 0:
        print(f'    WARNING: {n_dups:,} duplicate (unit, time) pairs detected. '
              f'Keeping first occurrence.')
        df = df.drop_duplicates(subset=['artist_int_id', 'panel_time'], keep='first')

    # --- Fill NaN → 0 for numeric count columns ---
    for col in NUMERIC_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Career stage at biennale year ---
    # The data has 'career_age' column (career age at biennale year).
    # career_stage exists but only for treated artists.
    # We recompute for ALL artists using career_age with 10/25 split.
    if 'career_age' in df.columns:
        vca = pd.to_numeric(df['career_age'], errors='coerce')
    elif 'venue_career_age' in df.columns:
        vca = pd.to_numeric(df['venue_career_age'], errors='coerce')
    elif 'min_year' in df.columns and 'biennale_year' in df.columns:
        # min_year = first exhibition year for the artist
        vca = (pd.to_numeric(df['biennale_year'], errors='coerce') -
               pd.to_numeric(df['min_year'], errors='coerce'))
    else:
        vca = pd.Series(np.nan, index=df.index)

    df['venue_career_age'] = vca

    # Assign career stage (10 / 25 split) for ALL artists
    df['career_stage'] = np.nan
    df.loc[vca <= 10, 'career_stage'] = 'early-career'
    df.loc[(vca > 10) & (vca <= 25), 'career_stage'] = 'mid-career'
    df.loc[vca > 25, 'career_stage'] = 'late-career'
    df.loc[vca.isna() | (vca < 0), 'career_stage'] = np.nan

    # --- Biennale decade (not matched on → extra cohort-trend control) ---
    if 'biennale_decade' in df.columns:
        df['biennale_decade'] = pd.to_numeric(
            df['biennale_decade'], errors='coerce'
        ).fillna(0).astype(int)
    elif 'biennale_year' in df.columns:
        df['biennale_decade'] = (
            pd.to_numeric(df['biennale_year'], errors='coerce') // 10 * 10
        ).fillna(0).astype(int)
    else:
        df['biennale_decade'] = 0

    # --- Gender labels ---
    # gender_label: fuzzy (mostly_male→male, mostly_female→female) for covariates
    # gender_exact: strict (only "male"/"female") for heterogeneity analysis
    GENDER_FUZZY_MAP = {
        'male': 'male',
        'female': 'female',
        'mostly_male': 'male',
        'mostly_female': 'female',
    }
    GENDER_EXACT_MAP = {
        'male': 'male',
        'female': 'female',
    }
    if 'gender_est' in df.columns:
        df['gender_label'] = df['gender_est'].map(GENDER_FUZZY_MAP)
        df['gender_exact'] = df['gender_est'].map(GENDER_EXACT_MAP)
    elif 'Gender' in df.columns:
        df['gender_label'] = df['Gender'].map({0: 'male', 1: 'female'})
        df['gender_exact'] = df['gender_label']  # already exact
    elif 'gender' in df.columns:
        df['gender_label'] = df['gender'].map(GENDER_FUZZY_MAP)
        df['gender_exact'] = df['gender'].map(GENDER_EXACT_MAP)
    else:
        df['gender_label'] = np.nan
        df['gender_exact'] = np.nan

    # --- Continent dummies: ensure they exist and are numeric ---
    # The data has: 'Europe', 'Asia', 'Africa', 'Oceania',
    #               'North America', 'South America'
    for col in CONTINENT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0

    # --- geo_group: for SUBSETTING in region heterogeneity only ---
    # Pick the first non-zero continent; all-zero → NaN
    def _pick_geo(row):
        for col in CONTINENT_COLS:
            if row.get(col, 0) == 1:
                return col  # column name IS the label
        return np.nan
    df['geo_group'] = df.apply(_pick_geo, axis=1)

    # ─── Create integer-coded covariates ───
    # Gender code: 0=female, 1=male (like Stata's i.gender_est)
    df['gender_code'] = df['gender_label'].map({'female': 0, 'male': 1})
    # Fill NaN with 0 for regression (unknown gender treated as reference)
    df['gender_code'] = df['gender_code'].fillna(0).astype(int)

    # Career stage code: 0=early, 1=mid, 2=late (like Stata's i.career_cat)
    df['career_code'] = df['career_stage'].map(CAREER_CODE_MAP)
    df['career_code'] = df['career_code'].fillna(0).astype(int)

    # --- Summary ---
    n_treated = df[df['is_treated'] == 1]['artist_id'].nunique()
    n_control = df[df['is_treated'] == 0]['artist_id'].nunique()
    print(f'    Treated: {n_treated:,}, Control: {n_control:,}')
    print(f'    panel_time range: {df["panel_time"].min()} - {df["panel_time"].max()}')
    print(f'    first_treat: treated={TIME_SHIFT}, control=0')
    print(f'    to_year range: {df["to_year"].min()} to {df["to_year"].max()}')

    if 'biennale_year' in df.columns:
        vy = df.loc[df['is_treated'] == 1, 'biennale_year']
        if not vy.empty:
            print(f'    Biennale year range: {vy.min()} - {vy.max()}')

    # Report NaN rates for subgroup columns
    for col in ['gender_label', 'gender_exact', 'career_stage', 'geo_group']:
        if col in df.columns:
            author_vals = df.groupby('artist_id')[col].first()
            n_na = author_vals.isna().sum()
            n_tot = len(author_vals)
            print(f'    {col}: {n_tot - n_na:,} known, {n_na:,} NaN '
                  f'({100 * n_na / n_tot:.1f}%)')

    # Report covariates
    present_covs = [c for c in COVARIATE_COLS if c in df.columns]
    print(f'    Covariates present: {present_covs}')

    meta = {
        'n_treated': n_treated,
        'n_control': n_control,
        'input_file': str(input_path),
        'time_shift': TIME_SHIFT,
    }

    return df, meta


# =============================================================================
# Run csdid estimation (same logic as science version)
# =============================================================================

def build_xformla(outcome, covariates, available_cols):
    """Build covariate formula for csdid."""
    if covariates is None:
        covariates = COVARIATE_COLS
    covs = [c for c in covariates if c in available_cols and c != outcome]
    if not covs:
        return f"{outcome}~1"
    return f"{outcome}~" + "+".join(covs)


def run_csdid(df, outcome, est_method='reg', covariates=None,
              n_boot=999, label=''):
    """
    Run csdid ATTgt estimation. Uses panel=False for unbalanced data.
    Includes singular-matrix fallback.
    """
    prefix = f'[{label}] ' if label else ''
    print(f'\n  {prefix}Estimating ATT(g,t) for: {outcome}')
    print(f'    Method: {est_method}, Bootstrap: {n_boot}')

    if outcome not in df.columns:
        print(f'    WARNING: outcome "{outcome}" not in data. Skipping.')
        return None

    avail_covs = [c for c in (covariates or COVARIATE_COLS)
                  if c in df.columns and c != outcome]

    xformla = build_xformla(outcome, covariates, set(df.columns))
    print(f'    Covariates formula: {xformla}')

    # Select columns
    keep_cols = ['artist_int_id', 'panel_time', 'first_treat', outcome] + avail_covs
    keep_cols = list(dict.fromkeys(keep_cols))
    work = df[keep_cols].copy()

    # Fill NaN → 0 for numeric columns
    numeric_cols = [outcome] + avail_covs
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0)

    # Ensure types
    work['panel_time'] = work['panel_time'].astype(int)
    work['first_treat'] = work['first_treat'].astype(int)
    work['artist_int_id'] = work['artist_int_id'].astype(int)

    n_units = work['artist_int_id'].nunique()
    n_treated_units = work.loc[work['first_treat'] > 0, 'artist_int_id'].nunique()
    n_control_units = work.loc[work['first_treat'] == 0, 'artist_int_id'].nunique()
    print(f'    Working data: {len(work):,} rows, {n_units:,} units '
          f'(T={n_treated_units:,}, C={n_control_units:,})')

    # Verify no duplicate (unit, time) pairs — critical for panel=True
    dup_check = work.groupby(['artist_int_id', 'panel_time']).size()
    n_dups = (dup_check > 1).sum()
    if n_dups > 0:
        print(f'    *** ERROR: {n_dups:,} duplicate (unit, time) pairs! ***')
        # Show example
        dup_pairs = dup_check[dup_check > 1].head(3)
        print(f'    Examples: {dup_pairs.to_dict()}')
        raise ValueError(f'{n_dups} duplicate (artist_int_id, panel_time) pairs. '
                         f'Fix: ensure matched_to column is present for unit_key.')

    def _fit(formula):
        """Try both API patterns."""
        try:
            att_gt = ATTgt(
                yname=outcome,
                gname="first_treat",
                idname="artist_int_id",
                tname="panel_time",
                xformla=formula,
                data=work,
                control_group='nevertreated',
                panel=True,
                est_method=est_method,
            )
            return att_gt.fit()
        except TypeError:
            att_gt = ATTgt(
                yname=outcome,
                gname="first_treat",
                idname="artist_int_id",
                tname="panel_time",
                xformla=formula,
                data=work,
                control_group='nevertreated',
                panel=True,
            )
            return att_gt.fit(est_method=est_method)

    t0 = time.time()
    try:
        result = _fit(xformla)
        elapsed = time.time() - t0
        print(f'    Fitted in {elapsed:.1f}s')

        # Print csdid output for verification
        print('\n    --- csdid ATT(g,t) summary ---')
        try:
            result.summ_attgt()
        except Exception:
            pass

        print('\n    --- csdid aggte("dynamic") ---')
        try:
            result.aggte("dynamic")
        except Exception:
            pass

        print('\n    --- csdid aggte("simple") ---')
        try:
            result.aggte("simple")
        except Exception:
            pass
        print('    --- end csdid output ---\n')

        return result

    except np.linalg.LinAlgError:
        # Singular matrix — drop covariates in priority order.
        # Drop smallest continent groups first (most likely to be
        # all-zero / near-constant), then decade, then core covariates.
        DROP_ORDER = [
            'Oceania',          # smallest continent group
            'Africa',           # second smallest
            'SouthAmerica',     # third smallest
            'Asia',             # can be sparse in some venues
            'NorthAmerica',     # usually large but drop before Europe
            'Europe',           # largest — drop last among continents
            'biennale_decade',  # can cause singularity in small subgroups
            'career_code',      # core covariate
            'gender_code',      # core covariate
        ]

        remaining = list(avail_covs)
        for drop_col in DROP_ORDER:
            if drop_col in remaining:
                remaining.remove(drop_col)
                if remaining:
                    reduced_formula = f"{outcome}~" + "+".join(remaining)
                else:
                    reduced_formula = f"{outcome}~1"
                print(f'    WARNING: Singular matrix, dropped {drop_col}, '
                      f'trying: {reduced_formula}')
                try:
                    result = _fit(reduced_formula)
                    elapsed = time.time() - t0
                    print(f'    Fitted in {elapsed:.1f}s')
                    return result
                except (np.linalg.LinAlgError, Exception):
                    continue

        # Final fallback: unconditional
        print(f'    WARNING: All covariates failed, falling back to ~1')
        try:
            result = _fit(f"{outcome}~1")
            elapsed = time.time() - t0
            print(f'    Fitted (unconditional) in {elapsed:.1f}s')
            return result
        except Exception as e2:
            print(f'    ERROR even unconditional: {e2}')
            return None

    except Exception as e:
        print(f'    ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Extract and aggregate results (same as science version)
# =============================================================================

def extract_attgt(result):
    """Extract the group-time ATT(g,t) table from csdid result."""
    from scipy.stats import norm

    try:
        summ = result.summ_attgt()
        df = summ.summary2.copy()
        col_map = {
            'Group': 'group', 'Time': 'time',
            'ATT(g, t)': 'att', 'ATT(g,t)': 'att',
            'Std. Error': 'se', 'Post': 'post',
            '[95% Pointwise': 'ci_lower', 'Conf. Band]': 'ci_upper',
            '[95.0% Pointwise': 'ci_lower',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.loc[:, df.columns.str.strip() != '']

        for col in ['att', 'se', 'ci_lower', 'ci_upper', 'group', 'time']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'pvalue' not in df.columns and 'att' in df.columns and 'se' in df.columns:
            z = (df['att'] / df['se']).replace([np.inf, -np.inf], 0)
            df['pvalue'] = 2 * norm.sf(np.abs(z))
        return df
    except Exception as e:
        print(f'    WARNING: Could not extract ATT(g,t) via summ_attgt: {e}')

    try:
        df = pd.DataFrame({
            'group': result.group,
            'time': result.t,
            'att': result.att,
            'se': result.se,
        })
        df['ci_lower'] = df['att'] - 1.96 * df['se']
        df['ci_upper'] = df['att'] + 1.96 * df['se']
        z = (df['att'] / df['se']).replace([np.inf, -np.inf], 0)
        df['pvalue'] = 2 * norm.sf(np.abs(z))
        return df
    except Exception as e:
        print(f'    WARNING: Could not extract ATT(g,t): {e}')
        return pd.DataFrame()


def _extract_aggte_results(agg_obj):
    """Try to extract results from aggte return object attributes."""
    from scipy.stats import norm

    attrs_to_try = [
        ('att_egt', 'se_egt', 'egt'),       # dynamic
        ('att_eg', 'se_eg', 'eg'),           # group
        ('att', 'se', None),                 # simple
    ]

    for att_attr, se_attr, idx_attr in attrs_to_try:
        try:
            att_vals = getattr(agg_obj, att_attr, None)
            se_vals = getattr(agg_obj, se_attr, None)
            if att_vals is None or se_vals is None:
                continue

            att_arr = np.atleast_1d(att_vals)
            se_arr = np.atleast_1d(se_vals)

            if idx_attr:
                idx_vals = np.atleast_1d(getattr(agg_obj, idx_attr))
            else:
                idx_vals = np.arange(len(att_arr))

            rows = []
            for i in range(len(att_arr)):
                a = float(att_arr[i])
                s = float(se_arr[i])
                z = a / s if s > 0 else 0
                rows.append({
                    'index': int(idx_vals[i]) if idx_attr else i,
                    'att': a,
                    'se': s,
                    'ci_lower': a - 1.96 * s,
                    'ci_upper': a + 1.96 * s,
                    'pvalue': 2 * norm.sf(abs(z)),
                })

            if rows:
                return pd.DataFrame(rows)
        except Exception:
            continue

    return pd.DataFrame()


def _capture_aggte(result, agg_type):
    """Call result.aggte(agg_type) and capture stdout."""
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        agg = result.aggte(agg_type)
    finally:
        sys.stdout = old_stdout
    printed = buffer.getvalue()
    return printed, agg


def _parse_dynamic_output(printed_text):
    """Parse 'Dynamic Effects:' table from csdid printed output."""
    from scipy.stats import norm

    lines = printed_text.strip().split('\n')
    rows = []
    in_dynamic = False

    for line in lines:
        line = line.strip()
        if 'Dynamic Effects:' in line:
            in_dynamic = True
            continue
        if in_dynamic and line.startswith('---'):
            break
        if in_dynamic and line and (line[0].isdigit() or line[0] == '-'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Handle negative event times
                    idx = 0
                    if parts[0].lstrip('-').isdigit() and not parts[1].lstrip('-').replace('.', '').isdigit():
                        event_time = int(parts[0])
                        att = float(parts[1])
                        se = float(parts[2])
                        idx = 3
                    else:
                        # row_idx, event_time, att, se
                        event_time = int(parts[1])
                        att = float(parts[2])
                        se = float(parts[3])
                        idx = 4

                    ci_lower = float(parts[idx]) if len(parts) > idx else att - 1.96 * se
                    ci_upper = float(parts[idx + 1]) if len(parts) > idx + 1 else att + 1.96 * se
                    z = att / se if se > 0 else 0
                    pvalue = 2 * norm.sf(abs(z))
                    rows.append({
                        'event_time': event_time,
                        'att': att,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'pvalue': pvalue,
                    })
                except (ValueError, IndexError):
                    continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['to_year'] = df['event_time']
    return df.sort_values('event_time').reset_index(drop=True)


def _parse_group_output(printed_text):
    """Parse 'Group Effects:' table from csdid printed output."""
    from scipy.stats import norm

    lines = printed_text.strip().split('\n')
    rows = []
    in_group = False

    for line in lines:
        line = line.strip()
        if 'Group Effects:' in line:
            in_group = True
            continue
        if in_group and line.startswith('---'):
            break
        if in_group and line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    group = int(parts[1])
                    att = float(parts[2])
                    se = float(parts[3])
                    ci_lower = float(parts[4]) if len(parts) > 4 else att - 1.96 * se
                    ci_upper = float(parts[5]) if len(parts) > 5 else att + 1.96 * se
                    z = att / se if se > 0 else 0
                    pvalue = 2 * norm.sf(abs(z))
                    rows.append({
                        'group': group,
                        'att': att,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'pvalue': pvalue,
                    })
                except (ValueError, IndexError):
                    continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _parse_overall_output(printed_text):
    """Parse the overall ATT line from csdid printed output."""
    from scipy.stats import norm

    lines = printed_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and line[0] not in ('-', 'A', 'S', 'C', 'E', 'O', '\n', ' '):
            parts = line.replace('*', '').split()
            if len(parts) >= 4:
                try:
                    att = float(parts[0])
                    se = float(parts[1])
                    ci_lower = float(parts[2])
                    ci_upper = float(parts[3])
                    z = att / se if se > 0 else 0
                    pvalue = 2 * norm.sf(abs(z))
                    return pd.DataFrame([{
                        'att': att,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'pvalue': pvalue,
                    }])
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame()


def aggregate_dynamic(result):
    """Aggregate ATT(g,t) by event time e = time - group."""
    # Method 1: aggte object attributes
    try:
        printed, agg = _capture_aggte(result, "dynamic")
        extracted = _extract_aggte_results(agg)
        if not extracted.empty:
            extracted = extracted.rename(columns={'index': 'event_time'})
            extracted['to_year'] = extracted['event_time']
            return extracted.sort_values('event_time').reset_index(drop=True)
    except Exception:
        pass

    # Method 2: parse printed output
    try:
        printed, agg = _capture_aggte(result, "dynamic")
        df = _parse_dynamic_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") failed: {e}')

    # Method 3: manual fallback
    from scipy.stats import norm
    attgt = extract_attgt(result)
    if attgt.empty:
        return pd.DataFrame()

    attgt = attgt.copy()
    attgt['event_time'] = attgt['time'] - attgt['group']

    dynamic = attgt.groupby('event_time').agg(
        att=('att', 'mean'),
        se=('se', lambda x: np.sqrt(np.mean(x**2))),
        n_groups=('group', 'nunique'),
    ).reset_index()

    dynamic['ci_lower'] = dynamic['att'] - 1.96 * dynamic['se']
    dynamic['ci_upper'] = dynamic['att'] + 1.96 * dynamic['se']
    z = dynamic['att'] / dynamic['se'].replace(0, np.inf)
    dynamic['pvalue'] = 2 * norm.sf(np.abs(z))
    dynamic['to_year'] = dynamic['event_time']

    return dynamic.sort_values('event_time').reset_index(drop=True)


def aggregate_cohort(result):
    """Aggregate ATT(g,t) by treatment cohort (group)."""
    try:
        printed, agg = _capture_aggte(result, "group")
        extracted = _extract_aggte_results(agg)
        if not extracted.empty:
            extracted = extracted.rename(columns={'index': 'group'})
            return extracted
    except Exception:
        pass

    try:
        printed, agg = _capture_aggte(result, "group")
        df = _parse_group_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("group") failed: {e}')

    from scipy.stats import norm
    attgt = extract_attgt(result)
    if attgt.empty:
        return pd.DataFrame()

    post = attgt[attgt['time'] >= attgt['group']].copy()
    if post.empty:
        return pd.DataFrame()

    cohort = post.groupby('group').agg(
        att=('att', 'mean'),
        se=('se', lambda x: np.sqrt(np.mean(x**2))),
        n_periods=('time', 'count'),
    ).reset_index()

    cohort['ci_lower'] = cohort['att'] - 1.96 * cohort['se']
    cohort['ci_upper'] = cohort['att'] + 1.96 * cohort['se']
    z = cohort['att'] / cohort['se'].replace(0, np.inf)
    cohort['pvalue'] = 2 * norm.sf(np.abs(z))

    return cohort


def aggregate_overall(result):
    """Overall ATT (simple aggregation)."""
    try:
        printed, agg = _capture_aggte(result, "simple")
        extracted = _extract_aggte_results(agg)
        if not extracted.empty:
            return extracted
    except Exception:
        pass

    try:
        printed, agg = _capture_aggte(result, "simple")
        df = _parse_overall_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("simple") failed: {e}')

    from scipy.stats import norm
    attgt = extract_attgt(result)
    if attgt.empty:
        return pd.DataFrame()

    post = attgt[attgt['time'] >= attgt['group']].copy()
    if post.empty:
        return pd.DataFrame()

    overall_att = post['att'].mean()
    valid_se = post.loc[post['se'].notna() & (post['se'] > 0), 'se']
    if len(valid_se) > 0:
        overall_se = np.sqrt(np.mean(np.asarray(valid_se) ** 2))
    else:
        overall_se = np.nan

    z = overall_att / overall_se if pd.notna(overall_se) and overall_se > 0 else 0

    return pd.DataFrame([{
        'att': overall_att,
        'se': overall_se,
        'ci_lower': overall_att - 1.96 * overall_se if pd.notna(overall_se) else np.nan,
        'ci_upper': overall_att + 1.96 * overall_se if pd.notna(overall_se) else np.nan,
        'pvalue': 2 * norm.sf(abs(z)) if z != 0 else np.nan,
        'n_groups': post['group'].nunique(),
        'n_periods': len(post),
    }])


# =============================================================================
# Cohort analysis (Figure 3 equivalent for art)
# =============================================================================

def run_cohort_analysis(df, outcome, est_method='reg', covariates=None,
                        n_boot=999, year_bins=None):
    """
    Run csdid separately for each venue_year bin.
    Default bins: decades from data.
    """
    if year_bins is None and 'biennale_year' in df.columns:
        vy = df.loc[df['is_treated'] == 1, 'biennale_year'].dropna()
        if vy.empty:
            return pd.DataFrame()
        min_decade = (vy.min() // 10) * 10
        max_decade = (vy.max() // 10) * 10 + 10
        year_bins = list(range(int(min_decade), int(max_decade) + 1, 10))
    elif year_bins is None:
        return pd.DataFrame()

    results = []
    for i in range(len(year_bins) - 1):
        lo, hi = year_bins[i], year_bins[i + 1]
        label = f'{lo}-{hi-1}'

        treated_in_bin = set(
            df.loc[(df['is_treated'] == 1) &
                   (df['biennale_year'] >= lo) &
                   (df['biennale_year'] < hi), 'artist_id'].unique()
        )
        if len(treated_in_bin) == 0:
            print(f'    Cohort {label}: 0 treated, skipping')
            continue

        # Get controls — keep all controls for this subset
        ctrl_ids = set(df.loc[df['is_treated'] == 0, 'artist_id'].unique())
        keep_ids = treated_in_bin | ctrl_ids
        sub = df[df['artist_id'].isin(keep_ids)].copy()
        n_t = sub[sub['is_treated'] == 1]['artist_id'].nunique()
        n_c = sub[sub['is_treated'] == 0]['artist_id'].nunique()
        print(f'\n    Cohort {label}: T={n_t}, C={n_c}')

        if n_t == 0 or n_c == 0:
            continue

        res = run_csdid(sub, outcome, est_method=est_method,
                        covariates=covariates, n_boot=n_boot,
                        label=f'cohort={label}')
        if res is not None:
            dyn = aggregate_dynamic(res)
            if isinstance(dyn, pd.DataFrame) and not dyn.empty:
                dyn['cohort'] = label
                dyn['cohort_start'] = lo
                dyn['cohort_end'] = hi - 1
                results.append(dyn)

            ovr = aggregate_overall(res)
            if isinstance(ovr, pd.DataFrame) and not ovr.empty:
                ovr['cohort'] = label
                ovr['cohort_start'] = lo
                results.append(ovr)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# Heterogeneity analysis
# =============================================================================

def subset_by_attribute(df, col, value):
    """Subset panel to artists where `col == value` (author-level)."""
    author_attr = df.groupby('artist_id')[col].first()
    keep_authors = set(author_attr[author_attr == value].index)
    return df[df['artist_id'].isin(keep_authors)].copy()


def run_heterogeneity(df, outcome, subgroup_col, subgroup_values,
                      est_method='reg', covariates=None, n_boot=999):
    """Run csdid separately for each subgroup value."""
    results = {}
    for val in subgroup_values:
        sub = subset_by_attribute(df, subgroup_col, val)
        n_treated = sub[sub['is_treated'] == 1]['artist_id'].nunique()
        n_control = sub[sub['is_treated'] == 0]['artist_id'].nunique()
        print(f'\n  --- Subgroup: {subgroup_col}={val} '
              f'(T={n_treated:,}, C={n_control:,}) ---')

        if n_treated == 0 or n_control == 0:
            print(f'    SKIPPING: no treated or no control units')
            continue

        res = run_csdid(sub, outcome, est_method=est_method,
                        covariates=covariates, n_boot=n_boot,
                        label=f'{subgroup_col}={val}')
        if res is not None:
            results[val] = res

    return results


def run_geo_heterogeneity(df, outcome, est_method='reg',
                          covariates=None, n_boot=999):
    """Run csdid for each geography group."""
    results = {}
    for geo_label in GEO_GROUPS:
        sub = subset_by_attribute(df, 'geo_group', geo_label)
        n_treated = sub[sub['is_treated'] == 1]['artist_id'].nunique()
        n_control = sub[sub['is_treated'] == 0]['artist_id'].nunique()
        print(f'\n  --- Geo: {geo_label} '
              f'(T={n_treated:,}, C={n_control:,}) ---')

        if n_treated == 0 or n_control == 0:
            print(f'    SKIPPING: no treated or no control units')
            continue

        res = run_csdid(sub, outcome, est_method=est_method,
                        covariates=covariates, n_boot=n_boot,
                        label=f'geo={geo_label}')
        if res is not None:
            results[geo_label] = res

    return results


# =============================================================================
# Save results
# =============================================================================

def save_results(results_dict, output_dir, file_label):
    """Save all results to organized CSV files."""
    out_dir = Path(output_dir) / file_label
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in results_dict.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            fpath = out_dir / f'{key}.csv'
            data.to_csv(fpath, index=False)
            print(f'    Saved: {fpath.name}  ({len(data)} rows)')
        elif isinstance(data, dict):
            for subkey, subdata in data.items():
                if isinstance(subdata, pd.DataFrame) and not subdata.empty:
                    fpath = out_dir / f'{key}_{subkey}.csv'
                    subdata.to_csv(fpath, index=False)
                    print(f'    Saved: {fpath.name}  ({len(subdata)} rows)')

    print(f'\n  All results saved to: {out_dir}')


# =============================================================================
# Main pipeline
# =============================================================================

def process_one_file(input_path, output_dir, outcomes, heterogeneity_types,
                     est_method='reg', covariates=None, n_boot=999):
    """Full pipeline for one art matched panel file."""
    input_path = Path(input_path)
    file_label = input_path.stem  # e.g. "matched_venice_biennale"

    print(f'\n{"="*70}')
    print(f'  ART VENUE DiD ESTIMATION: {file_label}')
    print(f'{"="*70}')

    df, meta = load_and_prepare(input_path)

    all_results = {}

    for outcome in outcomes:
        if outcome not in df.columns:
            print(f'\n  WARNING: {outcome} not in data, skipping.')
            continue

        outcome_tag = outcome

        # ── 1. General effect ──
        print(f'\n{"─"*50}')
        print(f'  OUTCOME: {outcome}')
        print(f'{"─"*50}')

        result = run_csdid(df, outcome, est_method=est_method,
                           covariates=covariates, n_boot=n_boot,
                           label='general')

        if result is None:
            continue

        # Dynamic (event-study) aggregation
        dynamic = aggregate_dynamic(result)
        if isinstance(dynamic, pd.DataFrame) and not dynamic.empty:
            all_results[f'dynamic_{outcome_tag}'] = dynamic
            print(f'\n    Dynamic ATT (event-study):')
            for _, r in dynamic.iterrows():
                e = r.get('event_time', '?')
                a = float(r.get('att', 0))
                se_val = float(r.get('se', 0))
                pv = r.get('pvalue', np.nan)
                sig = '*' if se_val > 0 and abs(a) > 1.96 * se_val else ''
                print(f'      e={e:>3}: ATT={a:>10.4f} (SE={se_val:.4f}) p={pv:.4g}{sig}')

        # Overall ATT
        overall = aggregate_overall(result)
        if isinstance(overall, pd.DataFrame) and not overall.empty:
            all_results[f'overall_{outcome_tag}'] = overall
            try:
                att_val = float(overall.iloc[0].get('att', 0))
                pv_val = float(overall.iloc[0].get('pvalue', 0))
                print(f'\n    Overall ATT: {att_val:.4f} (p={pv_val:.4g})')
            except (ValueError, TypeError):
                pass

        # ── 2. Cohort analysis ──
        if 'cohort' in heterogeneity_types:
            print(f'\n  ── Cohort analysis: {outcome} ──')
            cohort_df = run_cohort_analysis(
                df, outcome, est_method=est_method,
                covariates=covariates, n_boot=n_boot
            )
            if isinstance(cohort_df, pd.DataFrame) and not cohort_df.empty:
                all_results[f'cohort_{outcome_tag}'] = cohort_df

        # ── 3. Heterogeneity ──
        for het_type in heterogeneity_types:
            if het_type == 'cohort':
                continue

            # Build covariates: drop splitting variable
            het_covs = covariates
            if het_covs is None:
                drop_cols = COVARIATES_TO_DROP.get(het_type, [])
                het_covs = [c for c in COVARIATE_COLS if c not in drop_cols]
                print(f'\n  Covariates for {het_type}: '
                      f'{het_covs} (dropped {drop_cols})')

            if het_type == 'gender':
                print(f'\n  ── Gender heterogeneity (exact only): {outcome} ──')
                het_results = run_heterogeneity(
                    df, outcome, 'gender_exact', ['female', 'male'],
                    est_method=est_method, covariates=het_covs,
                    n_boot=n_boot
                )
                het_prefix = 'gender'

            elif het_type == 'career_stage':
                print(f'\n  ── Career stage heterogeneity: {outcome} ──')
                het_results = run_heterogeneity(
                    df, outcome, 'career_stage',
                    ['early-career', 'mid-career', 'late-career'],
                    est_method=est_method, covariates=het_covs,
                    n_boot=n_boot
                )
                het_prefix = 'career'

            elif het_type == 'region':
                print(f'\n  ── Region heterogeneity: {outcome} ──')
                het_results = run_geo_heterogeneity(
                    df, outcome, est_method=est_method,
                    covariates=het_covs, n_boot=n_boot
                )
                het_prefix = 'region'

            else:
                continue

            # Save each subgroup's dynamic and overall
            for sval, sres in het_results.items():
                safe_sval = str(sval).replace(' ', '_')

                dyn = aggregate_dynamic(sres)
                if isinstance(dyn, pd.DataFrame) and not dyn.empty:
                    dyn['subgroup'] = sval
                    all_results[
                        f'dynamic_{outcome_tag}_{het_prefix}_{safe_sval}'
                    ] = dyn

                ovr = aggregate_overall(sres)
                if isinstance(ovr, pd.DataFrame) and not ovr.empty:
                    ovr['subgroup'] = sval
                    all_results[
                        f'overall_{outcome_tag}_{het_prefix}_{safe_sval}'
                    ] = ovr

    # ── Save all ──
    print(f'\n{"─"*50}')
    print(f'  SAVING RESULTS')
    print(f'{"─"*50}')
    save_results(all_results, output_dir, file_label)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Art Venue Effect — Heterogeneous DiD via csdid')

    ap.add_argument('--input', type=str, default=None,
                    help='Single matched CSV file '
                         '(e.g., matched_venice_biennale.csv)')
    ap.add_argument('--input_dir', type=str, default=None,
                    help='Directory of matched CSV files')
    ap.add_argument('--output_dir', type=str, default='../../data/did_art',
                    help='Output directory for results')

    ap.add_argument('--outcomes', nargs='+', default=DEFAULT_OUTCOMES,
                    help='Outcome variables to estimate')
    ap.add_argument('--heterogeneity', nargs='*', default=[],
                    choices=['gender', 'career_stage', 'region', 'cohort'],
                    help='Heterogeneity analyses to run')
    ap.add_argument('--est_method', default='reg',
                    choices=['reg', 'ipw', 'dr'],
                    help='Estimation method')
    ap.add_argument('--n_boot', type=int, default=999,
                    help='Bootstrap iterations')
    ap.add_argument('--covariates', nargs='*', default=None,
                    help='Override covariate columns. '
                         'Use "none" for unconditional.')

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    # Handle --covariates none
    cov = args.covariates
    if cov and len(cov) == 1 and cov[0].lower() == 'none':
        cov = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_one_file(
            args.input, output_dir, args.outcomes, args.heterogeneity,
            est_method=args.est_method, covariates=cov,
            n_boot=args.n_boot
        )
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob('matched_*.csv'))
        print(f'\nFound {len(files)} matched files in {input_dir}')

        for fpath in files:
            process_one_file(
                fpath, output_dir, args.outcomes, args.heterogeneity,
                est_method=args.est_method, covariates=cov,
                n_boot=args.n_boot
            )


if __name__ == '__main__':
    main()