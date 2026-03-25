#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Heterogeneous DiD for ART (Dynamic / Calendar-Year)
==============================================================================

Uses REAL CALENDAR YEAR in csdid (like the science dynamic version):

    tname = end_year (calendar year)
    gname = biennale_year for treated, 0 for never-treated controls

This directly estimates:
  1. Dynamic/event-study effects (aggte("dynamic"))
  2. Cohort effects by actual biennale edition year (aggte("group"))
  3. Heterogeneity by subgroup (gender, career_stage, region)

Supports two control group modes via --control_group flag:
  - "nevertreated" (default): only never-treated controls (first_treat=0)
  - "notyettreated": later-cohort treated artists also serve as controls

Supports enriched_titles data with s_titles / S_titles outcomes.

Usage:
    python venue_did_csdid_dynamic_art.py \\
        --input ../../data/matches/enriched_titles/matched_venice_biennale_enriched_titles.csv \\
        --outcomes S G s_titles S_titles

    python venue_did_csdid_dynamic_art.py \\
        --input_dir ../../data/matches/enriched_titles \\
        --outcomes S G s_titles S_titles \\
        --heterogeneity gender career_stage region

    # Not-yet-treated
    python venue_did_csdid_dynamic_art.py \\
        --input ../../data/matches/enriched_titles/matched_venice_biennale_enriched_titles.csv \\
        --outcomes S G F \\
        --control_group notyettreated \\
        --output_dir ../../data/did_art_notyettreated
"""

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from csdid.att_gt import ATTgt
except ImportError:
    raise ImportError("pip install csdid\nSee: https://d2cml-ai.github.io/csdid/")

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTCOMES = ['S', 'G', 'F']

CONTINENT_COLS_RAW = ['Europe', 'Asia', 'Africa', 'Oceania',
                      'North America', 'South America']
CONTINENT_RENAME = {'North America': 'NorthAmerica', 'South America': 'SouthAmerica'}
CONTINENT_COLS = ['Europe', 'Asia', 'Africa', 'Oceania', 'NorthAmerica', 'SouthAmerica']

COVARIATE_COLS = [
    'gender_code', 'career_code',
    'Europe', 'Asia', 'Africa', 'Oceania', 'NorthAmerica', 'SouthAmerica',
    'biennale_decade',
]

COVARIATES_TO_DROP = {
    'gender':       ['gender_code'],
    'career_stage': ['career_code'],
    'region':       CONTINENT_COLS + ['biennale_decade'],
}

GEO_GROUPS = ['Europe', 'Asia', 'Africa', 'Oceania', 'NorthAmerica', 'SouthAmerica']

CAREER_CODE_MAP = {'early-career': 0, 'mid-career': 1, 'late-career': 2}

GENDER_FUZZY_MAP = {
    'male': 'male', 'female': 'female',
    'mostly_male': 'male', 'mostly_female': 'female',
}

NUMERIC_ZERO_FILL_COLS = [
    'S', 'G', 'F', 'B', 'B_adj',
    's_titles', 'S_titles',
    'biennale_decade',
]


def sanitize(x):
    return str(x).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')


# =============================================================================
# Data preparation — calendar-year design
# =============================================================================

def load_and_prepare(input_path, min_biennale_year=1900, min_treated_per_cohort=1):
    print(f'\n  Loading: {input_path}')
    try:
        df = pd.read_csv(input_path, sep=',')
        if len(df.columns) <= 2:
            df = pd.read_csv(input_path, sep=';')
    except Exception:
        df = pd.read_csv(input_path, sep=';')

    print(f'    {len(df):,} rows, {df["artist_id"].nunique():,} artists')

    df = df.rename(columns=CONTINENT_RENAME)

    if 'age_stage' in df.columns:
        n_before = df['artist_id'].nunique()
        df = df[(df['age_stage'] != 'posthumous') | df['age_stage'].isna()].copy()
        n_after = df['artist_id'].nunique()
        print(f'    Removed posthumous: {n_before - n_after:,} artists')

    if 'year_diff' in df.columns and 'to_year' not in df.columns:
        df['to_year'] = df['year_diff']
    if 'biennale_treated' in df.columns and 'is_venue' not in df.columns:
        df['is_venue'] = df['biennale_treated']

    if 'end_year' not in df.columns and 'biennale_year' in df.columns and 'year_diff' in df.columns:
        df['end_year'] = pd.to_numeric(df['biennale_year'], errors='coerce') + \
                         pd.to_numeric(df['year_diff'], errors='coerce')

    required = ['artist_id', 'end_year', 'biennale_year']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

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

    df['is_treated'] = df['artist_id'].isin(treated_ids).astype(int)

    df['panel_time'] = pd.to_numeric(df['end_year'], errors='coerce')
    df['first_treat'] = 0
    df.loc[df['is_treated'] == 1, 'first_treat'] = pd.to_numeric(
        df.loc[df['is_treated'] == 1, 'biennale_year'], errors='coerce'
    )

    df = df[df['panel_time'].notna()].copy()
    df['panel_time'] = df['panel_time'].astype(int)
    df['first_treat'] = df['first_treat'].fillna(0).astype(int)

    dup_check = df.groupby(['artist_int_id', 'panel_time']).size()
    n_dups = (dup_check > 1).sum()
    if n_dups > 0:
        print(f'    WARNING: {n_dups:,} duplicate (unit, time) pairs. Keeping first.')
        df = df.drop_duplicates(subset=['artist_int_id', 'panel_time'], keep='first')

    bad = (df['is_treated'] == 1) & (df['first_treat'] <= 0)
    if bad.any():
        bad_ids = set(df.loc[bad, 'artist_id'].unique())
        df.loc[df['artist_id'].isin(bad_ids), 'is_treated'] = 0
        df.loc[df['artist_id'].isin(bad_ids), 'first_treat'] = 0
        print(f'    Fixed {len(bad_ids)} treated with invalid biennale_year')

    old_ids = set(df.loc[(df['first_treat'] > 0) & (df['first_treat'] < min_biennale_year),
                         'artist_id'].unique())
    if old_ids:
        df = df[~df['artist_id'].isin(old_ids)].copy()
        print(f'    Dropped {len(old_ids)} treated with biennale_year < {min_biennale_year}')

    cohort_sizes = df[df['first_treat'] > 0].groupby('first_treat')['artist_int_id'].nunique()
    small = set(cohort_sizes[cohort_sizes < min_treated_per_cohort].index)
    valid_cohorts = set(cohort_sizes.index) - small
    if small:
        small_unit_ids = set(df.loc[df['first_treat'].isin(small), 'artist_int_id'].unique())
        df = df[~df['artist_int_id'].isin(small_unit_ids)].copy()
        print(f'    Dropped {len(small)} small cohorts')

    if valid_cohorts:
        lo = min(valid_cohorts) - 6
        hi = max(valid_cohorts) + 11
        n_before = len(df)
        df = df[(df['panel_time'] >= lo) & (df['panel_time'] <= hi)].copy()
        if len(df) < n_before:
            print(f'    Trimmed to [{lo}, {hi}]: {n_before - len(df)} rows removed')

    for col in NUMERIC_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'career_age' in df.columns:
        vca = pd.to_numeric(df['career_age'], errors='coerce')
    elif 'min_year' in df.columns and 'biennale_year' in df.columns:
        vca = pd.to_numeric(df['biennale_year'], errors='coerce') - \
              pd.to_numeric(df['min_year'], errors='coerce')
    else:
        vca = pd.Series(np.nan, index=df.index)
    df['career_stage'] = np.nan
    df.loc[vca <= 10, 'career_stage'] = 'early-career'
    df.loc[(vca > 10) & (vca <= 25), 'career_stage'] = 'mid-career'
    df.loc[vca > 25, 'career_stage'] = 'late-career'

    if 'biennale_decade' in df.columns:
        df['biennale_decade'] = pd.to_numeric(df['biennale_decade'], errors='coerce').fillna(0).astype(int)
    elif 'biennale_year' in df.columns:
        df['biennale_decade'] = (pd.to_numeric(df['biennale_year'], errors='coerce') // 10 * 10).fillna(0).astype(int)
    else:
        df['biennale_decade'] = 0

    GENDER_EXACT_MAP = {'male': 'male', 'female': 'female'}
    if 'gender_est' in df.columns:
        df['gender_label'] = df['gender_est'].map(GENDER_FUZZY_MAP)
        df['gender_exact'] = df['gender_est'].map(GENDER_EXACT_MAP)
    else:
        df['gender_label'] = np.nan
        df['gender_exact'] = np.nan

    for col in CONTINENT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            df[col] = 0

    def _pick_geo(row):
        for col in CONTINENT_COLS:
            if row.get(col, 0) == 1:
                return col
        return np.nan
    df['geo_group'] = df.apply(_pick_geo, axis=1)

    df['gender_code'] = df['gender_label'].map({'female': 0, 'male': 1}).fillna(0).astype(int)
    df['career_code'] = df['career_stage'].map(CAREER_CODE_MAP).fillna(0).astype(int)

    n_t = df[df['is_treated'] == 1]['artist_id'].nunique()
    n_c = df[df['is_treated'] == 0]['artist_id'].nunique()
    n_t_units = df[df['is_treated'] == 1]['artist_int_id'].nunique()
    n_c_units = df[df['is_treated'] == 0]['artist_int_id'].nunique()
    ft = df.loc[df['first_treat'] > 0, 'first_treat']
    print(f'    Treated: {n_t:,} artists ({n_t_units:,} units), '
          f'Control: {n_c:,} artists ({n_c_units:,} units)')
    print(f'    Calendar year range: {df["panel_time"].min()} - {df["panel_time"].max()}')
    if len(ft) > 0:
        print(f'    Cohort range: {ft.min()} - {ft.max()} ({ft.nunique()} cohorts)')

    outcome_cols = [c for c in ['S', 'G', 'F', 'B', 's_titles', 'S_titles'] if c in df.columns]
    print(f'    Available outcomes: {outcome_cols}')

    for col in ['gender_label', 'gender_exact', 'career_stage', 'geo_group']:
        vals = df.groupby('artist_id')[col].first()
        n_na = vals.isna().sum()
        print(f'    {col}: {len(vals) - n_na:,} known, {n_na:,} NaN')

    return df, {'n_treated': n_t, 'n_control': n_c}


# =============================================================================
# csdid estimation
# =============================================================================

def build_xformla(outcome, covariates, available_cols):
    if covariates is None:
        covariates = COVARIATE_COLS
    covs = [c for c in covariates if c in available_cols and c != outcome]
    return f"{outcome}~" + "+".join(covs) if covs else f"{outcome}~1"


def run_csdid(df, outcome, est_method='reg', covariates=None, n_boot=999, label='',
              control_group='nevertreated'):
    """
    Run csdid ATTgt estimation.

    control_group: "nevertreated" or "notyettreated"
    """
    prefix = f'[{label}] ' if label else ''
    print(f'\n  {prefix}Estimating ATT(g,t) for: {outcome}')
    print(f'    Method: {est_method}, Bootstrap: {n_boot}, Control group: {control_group}')

    if outcome not in df.columns:
        print(f'    WARNING: "{outcome}" not in data. Skipping.')
        return None

    avail_covs = [c for c in (covariates or COVARIATE_COLS)
                  if c in df.columns and c != outcome]
    xformla = build_xformla(outcome, covariates, set(df.columns))
    print(f'    Formula: {xformla}')

    keep = ['artist_int_id', 'panel_time', 'first_treat', outcome] + avail_covs
    keep = list(dict.fromkeys(keep))
    work = df[keep].copy()

    for col in [outcome] + avail_covs:
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0)
    work['panel_time'] = work['panel_time'].astype(int)
    work['first_treat'] = work['first_treat'].fillna(0).astype(int)
    work['artist_int_id'] = work['artist_int_id'].astype(int)

    nt = work.loc[work['first_treat'] > 0, 'artist_int_id'].nunique()
    nc = work.loc[work['first_treat'] == 0, 'artist_int_id'].nunique()
    print(f'    Data: {len(work):,} rows, T={nt:,}, C={nc:,}')
    cohorts = sorted(work.loc[work['first_treat'] > 0, 'first_treat'].unique())
    if cohorts:
        print(f'    Cohorts: {len(cohorts)} ({cohorts[0]}..{cohorts[-1]})')

    def _fit(formula):
        try:
            return ATTgt(
                yname=outcome, gname="first_treat", idname="artist_int_id",
                tname="panel_time", xformla=formula, data=work,
                control_group=control_group, panel=False, est_method=est_method,
            ).fit()
        except TypeError:
            att = ATTgt(
                yname=outcome, gname="first_treat", idname="artist_int_id",
                tname="panel_time", xformla=formula, data=work,
                control_group=control_group, panel=False,
            )
            try:
                return att.fit(est_method=est_method)
            except TypeError:
                return att.fit()

    DROP_ORDER = ['Oceania', 'Africa', 'SouthAmerica', 'Asia', 'NorthAmerica',
                  'Europe', 'biennale_decade', 'career_code', 'gender_code']

    t0 = time.time()
    try:
        result = _fit(xformla)
        print(f'    Fitted in {time.time()-t0:.1f}s')

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

        print('\n    --- csdid aggte("group") ---')
        try:
            result.aggte("group")
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
        remaining = list(avail_covs)
        for drop in DROP_ORDER:
            if drop in remaining:
                remaining.remove(drop)
                f = f"{outcome}~" + "+".join(remaining) if remaining else f"{outcome}~1"
                print(f'    Singular, dropped {drop}, trying: {f}')
                try:
                    result = _fit(f)
                    print(f'    Fitted in {time.time()-t0:.1f}s')
                    return result
                except (np.linalg.LinAlgError, Exception):
                    continue
        print(f'    All failed, trying ~1')
        try:
            return _fit(f"{outcome}~1")
        except Exception as e:
            print(f'    ERROR: {e}')
            return None

    except Exception as e:
        print(f'    ERROR: {e}')
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# Extraction / aggregation — ROBUST version from science script
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
        print(f'    WARNING: summ_attgt failed: {e}')

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
        print(f'    WARNING: direct attgt extraction failed: {e}')
        return pd.DataFrame()


def _capture_aggte(result, agg_type):
    """Capture printed output from result.aggte()."""
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        agg = result.aggte(agg_type)
    finally:
        sys.stdout = old
    return buf.getvalue(), agg


def _extract_aggte_results(result, agg_type):
    """
    Extract results from aggte() by accessing the return object directly.
    Ported from the science version — tries multiple attribute patterns.
    """
    from scipy.stats import norm

    printed, agg = _capture_aggte(result, agg_type)

    att_values = None
    se_values = None
    index_values = None

    # Try various attribute names used by different csdid versions
    for att_attr in ['att_egt', 'att_eg', 'att', 'overall_att']:
        val = getattr(agg, att_attr, None)
        if val is not None:
            arr = np.atleast_1d(val)
            if len(arr) > 0 and not np.all(np.isnan(arr)):
                att_values = arr
                break

    for se_attr in ['se_egt', 'se_eg', 'se', 'overall_se']:
        val = getattr(agg, se_attr, None)
        if val is not None:
            arr = np.atleast_1d(val)
            if len(arr) > 0:
                se_values = arr
                break

    for idx_attr in ['egt', 'eg', 'group', 't']:
        val = getattr(agg, idx_attr, None)
        if val is not None:
            arr = np.atleast_1d(val)
            if len(arr) > 0:
                index_values = arr
                break

    if att_values is not None and index_values is not None:
        n = len(att_values)
        rows = []
        for i in range(n):
            a = float(att_values[i])
            s = float(se_values[i]) if se_values is not None and i < len(se_values) else np.nan
            z = a / s if s > 0 else 0
            rows.append({
                'index': int(index_values[i]) if i < len(index_values) else i,
                'att': a,
                'se': s,
                'ci_lower': a - 1.96 * s,
                'ci_upper': a + 1.96 * s,
                'pvalue': 2 * norm.sf(abs(z)),
            })
        if rows:
            return pd.DataFrame(rows), printed

    # Try summary2
    if hasattr(agg, 'summary2'):
        try:
            return agg.summary2.copy(), printed
        except Exception:
            pass

    return None, printed


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
                    if len(parts) >= 6:
                        event_time = int(float(parts[1]))
                        att = float(parts[2])
                        se = float(parts[3])
                        ci_lower = float(parts[4])
                        ci_upper = float(parts[5])
                    else:
                        event_time = int(float(parts[1]))
                        att = float(parts[2])
                        se = float(parts[3])
                        ci_lower = att - 1.96 * se
                        ci_upper = att + 1.96 * se
                    z = att / se if se > 0 else 0
                    rows.append({
                        'event_time': event_time, 'att': att, 'se': se,
                        'ci_lower': ci_lower, 'ci_upper': ci_upper,
                        'pvalue': 2 * norm.sf(abs(z)),
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
        if in_group and line and (line[0].isdigit() or line[0] == '-'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    if len(parts) >= 6:
                        group = int(float(parts[1]))
                        att = float(parts[2])
                        se = float(parts[3])
                        ci_lower = float(parts[4])
                        ci_upper = float(parts[5])
                    else:
                        group = int(float(parts[1]))
                        att = float(parts[2])
                        se = float(parts[3])
                        ci_lower = att - 1.96 * se
                        ci_upper = att + 1.96 * se
                    z = att / se if se > 0 else 0
                    rows.append({
                        'group': group, 'att': att, 'se': se,
                        'ci_lower': ci_lower, 'ci_upper': ci_upper,
                        'pvalue': 2 * norm.sf(abs(z)),
                    })
                except (ValueError, IndexError):
                    continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('group').reset_index(drop=True)


def aggregate_dynamic(result):
    """Dynamic/event-study effects with 3-level fallback."""
    from scipy.stats import norm

    # Method 1: Extract from aggte object directly
    try:
        df, printed = _extract_aggte_results(result, "dynamic")
        if df is not None and not df.empty:
            df = df.rename(columns={'index': 'event_time'})
            df['to_year'] = df['event_time']
            return df.sort_values('event_time').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") object extraction failed: {e}')

    # Method 2: Parse printed output
    try:
        printed, _ = _capture_aggte(result, "dynamic")
        df = _parse_dynamic_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") print parsing failed: {e}')

    # Method 3: Manual aggregation from ATT(g,t)
    try:
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        attgt = attgt.copy()
        attgt['event_time'] = attgt['time'] - attgt['group']

        valid = attgt[attgt['se'].notna() & (attgt['se'] > 0)].copy()
        if valid.empty:
            dynamic = attgt.groupby('event_time').agg(
                att=('att', 'mean'),
                n_groups=('group', 'nunique'),
            ).reset_index()
            dynamic['se'] = np.nan
        else:
            dynamic = valid.groupby('event_time').agg(
                att=('att', 'mean'),
                se=('se', lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))),
                n_groups=('group', 'nunique'),
            ).reset_index()

        dynamic['ci_lower'] = dynamic['att'] - 1.96 * dynamic['se']
        dynamic['ci_upper'] = dynamic['att'] + 1.96 * dynamic['se']
        z = (dynamic['att'] / dynamic['se']).replace([np.inf, -np.inf], 0)
        dynamic['pvalue'] = 2 * norm.sf(np.abs(z))
        dynamic['to_year'] = dynamic['event_time']

        return dynamic.sort_values('event_time').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: Manual dynamic aggregation failed: {e}')
        return pd.DataFrame()


def aggregate_cohort(result):
    """Cohort/group effects with 3-level fallback."""
    from scipy.stats import norm

    # Method 1: Object extraction
    try:
        df, printed = _extract_aggte_results(result, "group")
        if df is not None and not df.empty:
            df = df.rename(columns={'index': 'group'})
            return df.sort_values('group').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: aggte("group") object extraction failed: {e}')

    # Method 2: Parse printed
    try:
        printed, _ = _capture_aggte(result, "group")
        df = _parse_group_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("group") print parsing failed: {e}')

    # Method 3: Manual from ATT(g,t)
    try:
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        post = attgt[attgt['time'] >= attgt['group']].copy()
        if post.empty:
            return pd.DataFrame()

        cohort_list = []
        for g, gdf in post.groupby('group'):
            row = {'group': g, 'att': gdf['att'].mean(), 'n_periods': len(gdf)}
            valid_se = gdf.loc[gdf['se'].notna() & (gdf['se'] > 0), 'se']
            row['se'] = np.sqrt(np.mean(np.asarray(valid_se) ** 2)) if len(valid_se) > 0 else np.nan
            cohort_list.append(row)

        cohort = pd.DataFrame(cohort_list)
        cohort['ci_lower'] = cohort['att'] - 1.96 * cohort['se']
        cohort['ci_upper'] = cohort['att'] + 1.96 * cohort['se']
        z = (cohort['att'] / cohort['se']).replace([np.inf, -np.inf], 0)
        cohort['pvalue'] = 2 * norm.sf(np.abs(z))

        return cohort.sort_values('group').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: Manual cohort aggregation failed: {e}')
        return pd.DataFrame()


def aggregate_overall(result):
    """Overall ATT with 2-level fallback."""
    from scipy.stats import norm

    # Method 1: Object extraction
    try:
        df, printed = _extract_aggte_results(result, "simple")
        if df is not None and not df.empty:
            return df.head(1)
    except Exception as e:
        print(f'    WARNING: aggte("simple") extraction failed: {e}')

    # Method 2: Manual from ATT(g,t)
    try:
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        post = attgt[attgt['time'] >= attgt['group']].copy()
        if post.empty:
            return pd.DataFrame()

        overall_att = post['att'].mean()
        valid_se = post.loc[post['se'].notna() & (post['se'] > 0), 'se']
        overall_se = np.sqrt(np.mean(np.asarray(valid_se) ** 2)) if len(valid_se) > 0 else np.nan
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
    except Exception as e:
        print(f'    WARNING: Manual overall aggregation failed: {e}')
        return pd.DataFrame()


# =============================================================================
# Heterogeneity
# =============================================================================

def subset_by(df, col, val):
    attr = df.groupby('artist_id')[col].first()
    keep = set(attr[attr == val].index)
    return df[df['artist_id'].isin(keep)].copy()


def run_heterogeneity(df, outcome, col, values, control_group='nevertreated', **kwargs):
    results = {}
    for val in values:
        sub = subset_by(df, col, val)
        nt = sub[sub['is_treated'] == 1]['artist_id'].nunique()
        nc = sub[sub['is_treated'] == 0]['artist_id'].nunique()
        print(f'\n  --- {col}={val} (T={nt:,}, C={nc:,}) ---')
        if nt == 0 or nc == 0:
            print('    SKIP')
            continue
        res = run_csdid(sub, outcome, label=f'{col}={val}',
                        control_group=control_group, **kwargs)
        if res is not None:
            results[val] = res
    return results


def run_geo(df, outcome, control_group='nevertreated', **kwargs):
    results = {}
    for g in GEO_GROUPS:
        sub = subset_by(df, 'geo_group', g)
        nt = sub[sub['is_treated'] == 1]['artist_id'].nunique()
        nc = sub[sub['is_treated'] == 0]['artist_id'].nunique()
        print(f'\n  --- Geo: {g} (T={nt:,}, C={nc:,}) ---')
        if nt == 0 or nc == 0:
            print('    SKIP')
            continue
        res = run_csdid(sub, outcome, label=f'geo={g}',
                        control_group=control_group, **kwargs)
        if res is not None:
            results[g] = res
    return results


# =============================================================================
# Save — INCREMENTAL: save after each outcome
# =============================================================================

def save_results(results_dict, output_dir, file_label):
    out_dir = Path(output_dir) / file_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, data in results_dict.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            p = out_dir / f'{key}.csv'
            data.to_csv(p, index=False)
            print(f'    Saved: {p.name} ({len(data)} rows)')
    print(f'\n  All saved to: {out_dir}')


def save_one_result(key, data, output_dir, file_label):
    """Save a single result CSV immediately."""
    out_dir = Path(output_dir) / file_label
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame) and not data.empty:
        p = out_dir / f'{key}.csv'
        data.to_csv(p, index=False)
        print(f'    Saved: {p.name} ({len(data)} rows)')


# =============================================================================
# Main pipeline
# =============================================================================

def process_one_file(input_path, output_dir, outcomes, het_types,
                     est_method='reg', covariates=None, n_boot=999,
                     control_group='nevertreated'):
    input_path = Path(input_path)
    file_label = input_path.stem

    print(f'\n{"="*70}')
    print(f'  ART VENUE DiD (DYNAMIC/CALENDAR-YEAR/{control_group.upper()}): {file_label}')
    print(f'{"="*70}')

    df, meta = load_and_prepare(input_path)
    all_results = {}

    for outcome in outcomes:
        if outcome not in df.columns:
            print(f'\n  WARNING: {outcome} not in data, skip.')
            continue

        otag = outcome
        print(f'\n{"─"*50}\n  OUTCOME: {outcome}\n{"─"*50}')

        result = run_csdid(df, outcome, est_method=est_method,
                           covariates=covariates, n_boot=n_boot, label='general',
                           control_group=control_group)
        if result is None:
            continue

        dyn = aggregate_dynamic(result)
        if not dyn.empty:
            all_results[f'dynamic_{otag}'] = dyn
            save_one_result(f'dynamic_{otag}', dyn, output_dir, file_label)
            print('\n    Dynamic ATT:')
            for _, r in dyn.iterrows():
                e, a, s = r.get('event_time','?'), r['att'], r['se']
                sig = '*' if s > 0 and abs(a) > 1.96*s else ''
                print(f'      e={e:>3}: ATT={a:>10.4f} (SE={s:.4f}){sig}')

        coh = aggregate_cohort(result)
        if not coh.empty:
            all_results[f'cohort_{otag}'] = coh
            save_one_result(f'cohort_{otag}', coh, output_dir, file_label)
            print('\n    Cohort ATT:')
            for _, r in coh.iterrows():
                g, a, s = r['group'], r['att'], r['se']
                sig = '*' if s > 0 and abs(a) > 1.96*s else ''
                print(f'      {int(g)}: ATT={a:>10.4f} (SE={s:.4f}){sig}')

        ovr = aggregate_overall(result)
        if not ovr.empty:
            all_results[f'overall_{otag}'] = ovr
            save_one_result(f'overall_{otag}', ovr, output_dir, file_label)
            print(f'\n    Overall ATT: {ovr.iloc[0]["att"]:.4f} '
                  f'(p={ovr.iloc[0]["pvalue"]:.4g})')

        for ht in het_types:
            hcovs = covariates
            if hcovs is None:
                drop = COVARIATES_TO_DROP.get(ht, [])
                hcovs = [c for c in COVARIATE_COLS if c not in drop]

            if ht == 'gender':
                print(f'\n  ── Gender het: {outcome} ──')
                hr = run_heterogeneity(df, outcome, 'gender_exact', ['female','male'],
                                       est_method=est_method, covariates=hcovs, n_boot=n_boot,
                                       control_group=control_group)
                hp = 'gender'
            elif ht == 'career_stage':
                print(f'\n  ── Career het: {outcome} ──')
                hr = run_heterogeneity(df, outcome, 'career_stage',
                                       ['early-career','mid-career','late-career'],
                                       est_method=est_method, covariates=hcovs, n_boot=n_boot,
                                       control_group=control_group)
                hp = 'career'
            elif ht == 'region':
                print(f'\n  ── Region het: {outcome} ──')
                hr = run_geo(df, outcome, est_method=est_method,
                             covariates=hcovs, n_boot=n_boot,
                             control_group=control_group)
                hp = 'region'
            else:
                continue

            for sval, sres in hr.items():
                sv = sanitize(sval)
                d = aggregate_dynamic(sres)
                if not d.empty:
                    d['subgroup'] = sval
                    key = f'dynamic_{otag}_{hp}_{sv}'
                    all_results[key] = d
                    save_one_result(key, d, output_dir, file_label)
                c = aggregate_cohort(sres)
                if not c.empty:
                    c['subgroup'] = sval
                    key = f'cohort_{otag}_{hp}_{sv}'
                    all_results[key] = c
                    save_one_result(key, c, output_dir, file_label)
                o = aggregate_overall(sres)
                if not o.empty:
                    o['subgroup'] = sval
                    key = f'overall_{otag}_{hp}_{sv}'
                    all_results[key] = o
                    save_one_result(key, o, output_dir, file_label)

    print(f'\n{"─"*50}\n  DONE — all results in: {Path(output_dir) / file_label}\n{"─"*50}')
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Art Venue DiD (dynamic/calendar-year)')
    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default='../../data/did_dynamic')
    ap.add_argument('--outcomes', nargs='+', default=DEFAULT_OUTCOMES)
    ap.add_argument('--heterogeneity', nargs='*', default=[],
                    choices=['gender', 'career_stage', 'region'])
    ap.add_argument('--est_method', default='reg', choices=['reg', 'ipw', 'dr'])
    ap.add_argument('--n_boot', type=int, default=999)
    ap.add_argument('--covariates', nargs='*', default=None)
    ap.add_argument('--glob', type=str, default='matched_*',
                    help='Glob pattern for input files (default: matched_*)')
    ap.add_argument('--control_group', default='nevertreated',
                    choices=['nevertreated', 'notyettreated'],
                    help='Control group type: nevertreated (default) or notyettreated')
    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide --input or --input_dir')

    cov = args.covariates
    if cov and len(cov) == 1 and cov[0].lower() == 'none':
        cov = []

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_one_file(args.input, out, args.outcomes, args.heterogeneity,
                         est_method=args.est_method, covariates=cov, n_boot=args.n_boot,
                         control_group=args.control_group)
    else:
        input_dir = Path(args.input_dir)
        pattern = args.glob
        files = sorted(
            list(input_dir.glob(f'{pattern}.csv')) +
            list(input_dir.glob(f'{pattern}.parquet'))
        )
        seen = {}
        for f in files:
            stem = f.stem
            if stem not in seen or f.suffix == '.parquet':
                seen[stem] = f
        files = sorted(seen.values())

        print(f'\nFound {len(files)} files in {input_dir} (pattern: {pattern})')
        for f in files:
            process_one_file(f, out, args.outcomes, args.heterogeneity,
                             est_method=args.est_method, covariates=cov, n_boot=args.n_boot,
                             control_group=args.control_group)


if __name__ == '__main__':
    main()