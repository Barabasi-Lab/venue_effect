#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Heterogeneous DiD Estimation via csdid (Python)
==============================================================================

Translates the Stata xthdidregress workflow into Python using the `csdid`
package (Callaway & Sant'Anna 2021 framework).

Produces three levels of results:
  1. General effect upon exposure  (dynamic/event-study aggregation)
  2. Effect by treatment cohort    (cohort/group aggregation)
  3. Heterogeneity by subgroup     (gender, career_stage, region)

Each result is saved as a CSV with columns:
  - ATT, se, ci_lower, ci_upper, pvalue
  - For dynamic:  event_time (= to_year relative to treatment)
  - For cohort:   group (= venue_year cohort)
  - For subgroup: subgroup label

Data expectations (from matching.py + enrich_citations.py):
  - Semicolon-delimited CSV
  - Key columns: author_id, year, to_year, venue_year, is_venue,
    Gender (0=M, 1=F), first_year_of_publication,
    venue_region_code (pipe-separated),
    citations_na, cum_citations_na, cum_publication_count,
    cum_funding_count, cum_corresponding_count
  - Treated authors: venue_year > 0
  - Control authors: venue_year = NaN or is_venue always 0

Usage:
    # Single file, all outcomes
    python venue_did_csdid.py \
        --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
        --outcomes citations_na cum_citations_na cum_publication_count cum_funding_count

    # All files in directory
    python venue_did_csdid.py \
        --input_dir ../../data/matches/enriched_citations \
        --outcomes citations_na cum_citations_na

    # With heterogeneity analysis
    python venue_did_csdid.py \
        --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
        --outcomes cum_citations_na \
        --heterogeneity gender career_stage region

    # Unconditional (no covariates)
    python venue_did_csdid.py \
        --input ... --outcomes cum_citations_na --covariates none

Requirements:
    pip install csdid pandas numpy
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

# Default outcomes to estimate
DEFAULT_OUTCOMES = [
    'cum_citations_na',
    'cum_publication_count',
    'cum_funding_count',
]

# Covariates for csdid. All time-invariant author-level attributes.
# K-1 dummies per variable (reference category omitted to avoid
# perfect multicollinearity with the intercept).
# References: female, early-career, Europe
COVARIATE_COLS = [
    'gender_male',          # 1 if male (reference: female)
    'career_mid',           # 1 if mid-career (reference: early-career)
    'career_late',          # 1 if late-career
    'geo_Asia',             # 1 if Asia (reference: Europe)
    'geo_Africa',           # 1 if Africa
    'geo_Oceania',          # 1 if Oceania
    'geo_NAmerica',         # 1 if Northern America
    'geo_LatAm',            # 1 if Latin America and the Caribbean
    'venue_decade',         # decade of venue access (1990, 2000, ...), controls for cohort trends
]

# Which covariates to DROP for each heterogeneity type
# (these are constant within each subgroup → would cause singularity)
COVARIATES_TO_DROP = {
    'gender':       ['gender_male'],
    'career_stage': ['career_mid', 'career_late'],
    'region':       ['geo_Asia', 'geo_Africa', 'geo_Oceania',
                     'geo_NAmerica', 'geo_LatAm'],
}

HEP_BROAD_FIELDS = [
    'Particle and High Energy Physics',
    'Synchrotrons and Accelerators',
    'Nuclear and Plasma Physics',
]

# Geography groups for region heterogeneity.
# Uses a unified geo_group column built from region + subregion.
# Europe/Asia/Africa/Oceania come from `region` column;
# Northern America and Latin America come from `subregion` column.
GEO_GROUPS = [
    'Europe',
    'Asia',
    'Africa',
    'Oceania',
    'Northern America',
    'Latin America and the Caribbean',
]

# Columns that are genuinely numeric counts where NaN means 0
# (e.g. "no citations that year" = 0, not missing data)
NUMERIC_ZERO_FILL_COLS = [
    'citations_na', 'cum_citations_na',
    'citations_old', 'cum_citations_old',
    'normalized_citations_na', 'cum_normalized_citations_na',
    'normalized_citations_old', 'cum_normalized_citations_old',
    'cum_publication_count', 'cum_funding_count',
    'cum_corresponding_count',
    'publication_count', 'funding_count', 'corresponding_count',
    'publication_count_adj', 'cum_publication_count_adj',
    'venue_decade',
]

CAREER_STAGE_DEFS = {
    'early-career':  (0, 10),   # venue_career_age <= 10
    'mid-career':    (11, 25),  # 10 < venue_career_age <= 25
    'late-career':   (26, 999), # venue_career_age > 25
}


# =============================================================================
# Data preparation
# =============================================================================

def load_and_prepare(input_path):
    """
    Load enriched matched panel CSV and prepare for csdid.

    KEY DESIGN: We use `to_year` (relative time) shifted to positive integers
    as the panel time variable, NOT raw calendar year. This is because:
      - Our panel only covers to_year in [-5, +10] per author (16 periods)
      - Using calendar year creates a (100+ cohorts × 100+ years) grid,
        most cells empty → "No units in group X in time period Y" warnings
      - With relative time, ALL authors share the same 16-period grid

    Time mapping: panel_time = to_year + 6  (so range is 1..16)
    Treatment:    first_treat = 6  for treated (treatment at panel_time=6)
                  first_treat = 0  for never-treated (csdid convention)

    For cohort-level analysis (Figure 3), we run separately by venue_year bins.

    Returns:
        df: prepared DataFrame
        meta: dict with file metadata
    """
    print(f'\n  Loading: {input_path}')
    df = pd.read_csv(input_path, sep=';')
    print(f'    {len(df):,} rows, {df["author_id"].nunique():,} authors')
    # --- Optional: exclude HEP-like authors if L2 modal field is available ---
    if 'l2_field_modal_name' in df.columns:
        n_rows_before = len(df)
        n_authors_before = df['author_id'].nunique()

        # Keep missing L2 labels; exclude only explicit HEP-like fields
        df = df[
            df['l2_field_modal_name'].isna() |
            (~df['l2_field_modal_name'].isin(HEP_BROAD_FIELDS))
        ].copy()

        n_rows_after = len(df)
        n_authors_after = df['author_id'].nunique()

        print(
            f'    Excluded HEP-like modal L2 fields: '
            f'{n_authors_before - n_authors_after:,} authors, '
            f'{n_rows_before - n_rows_after:,} rows removed'
        )
    else:
        print('    l2_field_modal_name not found; keeping all authors')

    # --- Ensure required columns ---
    required = ['author_id', 'year', 'to_year', 'venue_year', 'is_venue']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    # --- Create integer author ID for csdid (needs numeric id) ---
    author_ids = df['author_id'].unique()
    id_map = {aid: i + 1 for i, aid in enumerate(author_ids)}
    df['author_int_id'] = df['author_id'].map(id_map)

    # --- Identify treated vs control ---
    treated_ids = set(df.loc[df['is_venue'] == 1, 'author_id'].unique())
    df['is_treated'] = df['author_id'].isin(treated_ids).astype(int)

    # --- Create panel_time from to_year (shifted to positive) ---
    # to_year ranges from -5 to +10 → panel_time from 1 to 16
    TIME_SHIFT = 6  # so to_year=0 becomes panel_time=6
    df['panel_time'] = df['to_year'].astype(int) + TIME_SHIFT

    # --- Create first_treat ---
    # Treated: first_treat = TIME_SHIFT (= 6, treatment at panel_time=6)
    # Control: first_treat = 0 (csdid convention for never-treated)
    df['first_treat'] = 0
    df.loc[df['is_treated'] == 1, 'first_treat'] = TIME_SHIFT

    # --- Fill NaN → 0 ONLY for known numeric count/cumulative columns ---
    for col in NUMERIC_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Career stage at venue year (categorical — NaN stays NaN) ---
    # venue_career_age = venue_year - first_year_of_publication
    # This is the career age at the time of treatment (venue access).
    # For controls, venue_year is their matched treated author's venue year,
    # so this gives career age at the same calendar point.
    # Cutoffs: early ≤ 10, mid 10-25, late > 25
    if 'first_year_of_publication' in df.columns:
        df['venue_career_age'] = (
            pd.to_numeric(df['venue_year'], errors='coerce') -
            pd.to_numeric(df['first_year_of_publication'], errors='coerce')
        )
        # Assign career stage using explicit conditions (mirrors Stata logic)
        df['career_stage'] = np.nan
        df.loc[df['venue_career_age'] <= 10, 'career_stage'] = 'early-career'
        df.loc[(df['venue_career_age'] > 10) &
               (df['venue_career_age'] <= 25), 'career_stage'] = 'mid-career'
        df.loc[df['venue_career_age'] > 25, 'career_stage'] = 'late-career'
        # Keep NaN for missing/negative career ages
        df.loc[df['venue_career_age'].isna() |
               (df['venue_career_age'] < 0), 'career_stage'] = np.nan
    else:
        df['venue_career_age'] = np.nan
        df['career_stage'] = np.nan

    # --- Venue decade (numeric covariate for time-trend control) ---
    # venue_year is the calendar year of first top-venue pub (same for
    # matched controls). Decade as integer: 1990, 2000, 2010, etc.
    df['venue_decade'] = (df['venue_year'] // 10 * 10).astype(int)

    # --- Region (categorical — NaN stays NaN if missing) ---
    if 'venue_region_code' in df.columns:
        df['region'] = df['venue_region_code'].apply(
            lambda x: str(x).split('|')[0].strip("[] '\"")
            if pd.notna(x) and str(x) not in ('nan', 'None', '') else np.nan
        )
    else:
        df['region'] = np.nan

    # --- Subregion (categorical — NaN stays NaN) ---
    if 'venue_subregion_code' in df.columns:
        df['subregion'] = df['venue_subregion_code'].apply(
            lambda x: str(x).split('|')[0].strip("[] '\"")
            if pd.notna(x) and str(x) not in ('nan', 'None', '') else np.nan
        )
    else:
        df['subregion'] = np.nan

    # --- Unified geo_group column for region heterogeneity ---
    # Americas use subregion (Northern America / Latin America and the Caribbean)
    # Others use region (Europe / Asia / Africa / Oceania)
    def _geo(row):
        sub = row.get('subregion')
        reg = row.get('region')
        if pd.notna(sub) and sub in ('Northern America',
                                      'Latin America and the Caribbean'):
            return sub
        if pd.notna(reg) and reg in ('Europe', 'Asia', 'Africa', 'Oceania'):
            return reg
        return np.nan
    df['geo_group'] = df.apply(_geo, axis=1)

    # --- Gender label (categorical — NaN stays NaN) ---
    if 'Gender' in df.columns:
        df['gender_label'] = df['Gender'].map({0: 'male', 1: 'female'})
    else:
        df['gender_label'] = np.nan

    # ─── Create K-1 dummy covariates for csdid ───
    # NaN in source → 0 in dummy (neutral for regression)

    # Gender: male=1 (reference: female)
    df['gender_male'] = (df['gender_label'] == 'male').astype(int)

    # Career stage (reference: early-career)
    df['career_mid'] = (df['career_stage'] == 'mid-career').astype(int)
    df['career_late'] = (df['career_stage'] == 'late-career').astype(int)

    # Geography (reference: Europe)
    df['geo_Asia'] = (df['geo_group'] == 'Asia').astype(int)
    df['geo_Africa'] = (df['geo_group'] == 'Africa').astype(int)
    df['geo_Oceania'] = (df['geo_group'] == 'Oceania').astype(int)
    df['geo_NAmerica'] = (df['geo_group'] == 'Northern America').astype(int)
    df['geo_LatAm'] = (df['geo_group'] ==
                        'Latin America and the Caribbean').astype(int)

    # --- Summary ---
    n_treated = df[df['is_treated'] == 1]['author_id'].nunique()
    n_control = df[df['is_treated'] == 0]['author_id'].nunique()
    vy_range = df.loc[df['is_treated'] == 1, 'venue_year']
    print(f'    Treated: {n_treated:,}, Control: {n_control:,}')
    print(f'    Venue year range: {vy_range.min()} - {vy_range.max()}')
    print(f'    panel_time range: {df["panel_time"].min()} - {df["panel_time"].max()}')
    print(f'    first_treat: treated={TIME_SHIFT}, control=0')
    print(f'    to_year range: {df["to_year"].min()} to {df["to_year"].max()}')

    # Report NaN rates for subgroup columns (per-author)
    for col in ['gender_label', 'career_stage', 'geo_group']:
        if col in df.columns:
            author_vals = df.groupby('author_id')[col].first()
            n_na = author_vals.isna().sum()
            n_tot = len(author_vals)
            print(f'    {col}: {n_tot - n_na:,} known, {n_na:,} NaN '
                  f'({100 * n_na / n_tot:.1f}%)')

    # Report which covariate columns actually exist
    present_covs = [c for c in COVARIATE_COLS if c in df.columns]
    missing_covs = [c for c in COVARIATE_COLS if c not in df.columns]
    print(f'    Covariates present: {present_covs}')
    if missing_covs:
        print(f'    Covariates MISSING: {missing_covs}')

    meta = {
        'n_treated': n_treated,
        'n_control': n_control,
        'input_file': str(input_path),
        'time_shift': TIME_SHIFT,
    }

    return df, meta


# =============================================================================
# Run csdid estimation
# =============================================================================

def build_xformla(outcome, covariates, available_cols):
    """
    Build the covariate formula for csdid.
    
    csdid Python expects format: "outcome~covar1+covar2" or "outcome~1"
    (outcome on the LHS, covariates on the RHS).
    """
    if covariates is None:
        covariates = COVARIATE_COLS
    covs = [c for c in covariates if c in available_cols and c != outcome]
    if not covs:
        return f"{outcome}~1"
    return f"{outcome}~" + "+".join(covs)


def run_csdid(df, outcome, est_method='reg', covariates=None,
              n_boot=999, label=''):
    """
    Run csdid ATTgt estimation and return the fitted object.

    Uses panel_time (= to_year + 6) as tname and first_treat (= 6 for treated,
    0 for never-treated) as gname. This ensures a compact (group × time) grid
    with no empty cells.

    Includes singular-matrix fallback: if covariates cause a LinAlgError
    (e.g. in a small subgroup where a dummy is constant), automatically
    retries with outcome~1 (unconditional).
    """
    prefix = f'[{label}] ' if label else ''
    print(f'\n  {prefix}Estimating ATT(g,t) for: {outcome}')
    print(f'    Method: {est_method}, Bootstrap: {n_boot}')

    if outcome not in df.columns:
        print(f'    WARNING: outcome "{outcome}" not in data. Skipping.')
        return None

    # Determine available covariates
    avail_covs = [c for c in (covariates or COVARIATE_COLS)
                  if c in df.columns and c != outcome]

    xformla = build_xformla(outcome, covariates, set(df.columns))
    print(f'    Covariates formula: {xformla}')

    # Select columns needed for estimation
    keep_cols = ['author_int_id', 'panel_time', 'first_treat', outcome] + avail_covs
    # Deduplicate column list
    keep_cols = list(dict.fromkeys(keep_cols))
    work = df[keep_cols].copy()

    # Fill NaN → 0 for all numeric columns (outcome + covariates)
    # These are count/cumulative variables where NaN genuinely means 0
    numeric_cols = [outcome] + avail_covs
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0)

    # Ensure types
    work['panel_time'] = work['panel_time'].astype(int)
    work['first_treat'] = work['first_treat'].astype(int)
    work['author_int_id'] = work['author_int_id'].astype(int)

    n_units = work['author_int_id'].nunique()
    n_treated_units = work.loc[work['first_treat'] > 0, 'author_int_id'].nunique()
    n_control_units = work.loc[work['first_treat'] == 0, 'author_int_id'].nunique()
    print(f'    Working data: {len(work):,} rows, {n_units:,} units '
          f'(T={n_treated_units:,}, C={n_control_units:,})')
    print(f'    panel_time range: {work["panel_time"].min()}-{work["panel_time"].max()}, '
          f'first_treat values: {sorted(work["first_treat"].unique())}')

    def _fit(formula):
        """Try both API patterns for est_method."""
        try:
            att_gt = ATTgt(
                yname=outcome,
                gname="first_treat",
                idname="author_int_id",
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
                idname="author_int_id",
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

        # Print raw csdid output for verification
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
        # Singular matrix — covariate(s) constant in this subgroup
        print(f'    WARNING: Singular matrix with covariates, '
              f'falling back to unconditional (~1)')
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
# Extract and aggregate results
# =============================================================================

def extract_attgt(result):
    """Extract the group-time ATT(g,t) table from csdid result.
    
    Returns DataFrame with columns: group, time, att, se, ci_lower, ci_upper, pvalue
    """
    from scipy.stats import norm
    
    # Primary: .summ_attgt().summary2
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
        # Drop empty columns
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

    # Fallback: attributes
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


def _capture_aggte(result, agg_type):
    """
    Call result.aggte(agg_type) and capture the results.
    
    csdid's .aggte() prints the correct aggregated results to stdout
    but the returned object's .summ_attgt().summary2 still contains the
    raw ATT(g,t). We capture stdout and parse the printed table.
    
    Returns: (parsed_df, agg_object)
    """
    import io
    import sys
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        agg = result.aggte(agg_type)
    finally:
        sys.stdout = old_stdout
    
    printed = buffer.getvalue()
    return printed, agg


def _parse_dynamic_output(printed_text):
    """Parse the 'Dynamic Effects:' table from csdid's printed output."""
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
        if in_dynamic and line and line[0].isdigit():
            # Parse: "0           -4   -0.0658      0.3918   ..."
            parts = line.split()
            if len(parts) >= 4:
                try:
                    event_time = int(parts[1])
                    att = float(parts[2])
                    se = float(parts[3])
                    ci_lower = float(parts[4]) if len(parts) > 4 else att - 1.96 * se
                    ci_upper = float(parts[5]) if len(parts) > 5 else att + 1.96 * se
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
    """Parse the 'Group Effects:' table from csdid's printed output."""
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
    """Parse the overall ATT line from csdid's printed output."""
    from scipy.stats import norm
    
    lines = printed_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for the line with ATT value (after the header line)
        # Header: "     ATT Std. Error   [95.0%  Conf. Int.]"
        # Value:  "200.5527    14.4372 172.2564      228.849 *"
        if line and line[0] not in ('-', 'A', 'S', 'C', 'E', 'O', '\n', ' '):
            # Try to parse as: att se ci_lower ci_upper [*]
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
    """Aggregate ATT(g,t) by event time e = time - group.
    
    Captures csdid's .aggte("dynamic") printed output which contains
    the properly computed dynamic effects with bootstrap SEs.
    """
    try:
        printed, agg = _capture_aggte(result, "dynamic")
        df = _parse_dynamic_output(printed)
        if not df.empty:
            return df
        print('    WARNING: Could not parse aggte("dynamic") output')
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") failed: {e}')

    # Manual fallback from ATT(g,t)
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
        df = _parse_group_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("group") failed: {e}')

    # Manual fallback
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
    """Overall ATT. Equivalent to `estat aggregation` (simple)."""
    try:
        printed, agg = _capture_aggte(result, "simple")
        df = _parse_overall_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("simple") failed: {e}')

    # Fallback
    from scipy.stats import norm
    attgt = extract_attgt(result)
    if attgt.empty:
        return pd.DataFrame()
    
    post = attgt[attgt['time'] >= attgt['group']].copy()
    if post.empty:
        return pd.DataFrame()
    
    overall_att = post['att'].mean()
    overall_se = np.sqrt(np.mean(post['se']**2))
    z = overall_att / overall_se if overall_se > 0 else 0
    
    return pd.DataFrame([{
        'att': overall_att,
        'se': overall_se,
        'ci_lower': overall_att - 1.96 * overall_se,
        'ci_upper': overall_att + 1.96 * overall_se,
        'pvalue': 2 * norm.sf(abs(z)),
        'n_groups': post['group'].nunique(),
        'n_periods': len(post),
    }])


# =============================================================================
# Cohort analysis (Figure 3: effect by treatment year)
# =============================================================================

def run_cohort_analysis(df, outcome, est_method='reg', covariates=None,
                        n_boot=999, year_bins=None):
    """
    For Figure 3 (effect by treatment cohort / venue_year):
    Run csdid separately for each venue_year bin and collect the
    overall ATT per bin.

    Since our panel_time design collapses all cohorts into one group,
    we recover cohort variation by subsetting the data by venue_year.

    Default bins: decades (1960s, 1970s, ..., 2010s)
    """
    if year_bins is None:
        # Create decade bins from the data
        vy = df.loc[df['is_treated'] == 1, 'venue_year']
        min_decade = (vy.min() // 10) * 10
        max_decade = (vy.max() // 10) * 10 + 10
        year_bins = list(range(int(min_decade), int(max_decade) + 1, 10))

    results = []
    for i in range(len(year_bins) - 1):
        lo, hi = year_bins[i], year_bins[i + 1]
        label = f'{lo}-{hi-1}'

        # Keep treated authors in this venue_year range + ALL their controls
        treated_in_bin = set(
            df.loc[(df['is_treated'] == 1) &
                   (df['venue_year'] >= lo) &
                   (df['venue_year'] < hi), 'author_id'].unique()
        )
        if len(treated_in_bin) == 0:
            print(f'    Cohort {label}: {len(treated_in_bin)} treated, skipping')
            continue

        # Get matched controls for these treated authors
        if 'matched_to' in df.columns:
            ctrl_ids = set(
                df.loc[df['matched_to'].isin(treated_in_bin), 'author_id'].unique()
            )
        else:
            # If no matched_to column, keep all controls
            ctrl_ids = set(
                df.loc[df['is_treated'] == 0, 'author_id'].unique()
            )

        keep_ids = treated_in_bin | ctrl_ids
        sub = df[df['author_id'].isin(keep_ids)].copy()
        n_t = sub[sub['is_treated'] == 1]['author_id'].nunique()
        n_c = sub[sub['is_treated'] == 0]['author_id'].nunique()
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
    """
    Subset panel to authors where `col == value`.

    The attribute is checked per-author (first non-NaN value), not per-row,
    since gender/career_stage/region are author-level attributes.
    All rows for qualifying authors are kept (preserving panel structure).
    Authors with NaN in `col` are excluded.
    """
    author_attr = df.groupby('author_id')[col].first()
    keep_authors = set(author_attr[author_attr == value].index)
    return df[df['author_id'].isin(keep_authors)].copy()


def run_heterogeneity(df, outcome, subgroup_col, subgroup_values,
                      est_method='reg', covariates=None, n_boot=999):
    """
    Run csdid separately for each subgroup value.
    Authors with NaN in subgroup_col are automatically excluded.
    Returns dict of {subgroup_value: result_object}
    """
    results = {}
    for val in subgroup_values:
        sub = subset_by_attribute(df, subgroup_col, val)
        n_treated = sub[sub['is_treated'] == 1]['author_id'].nunique()
        n_control = sub[sub['is_treated'] == 0]['author_id'].nunique()
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
    """
    Run csdid for each geography group using the unified geo_group column.

    Groups: Europe, Asia, Africa, Oceania, Northern America,
            Latin America and the Caribbean

    This replaces the old run_region_heterogeneity() which filtered on
    separate region/subregion columns. Using the unified geo_group column
    ensures correct subsetting for all groups including Americas.
    """
    results = {}
    for geo_label in GEO_GROUPS:
        sub = subset_by_attribute(df, 'geo_group', geo_label)
        n_treated = sub[sub['is_treated'] == 1]['author_id'].nunique()
        n_control = sub[sub['is_treated'] == 0]['author_id'].nunique()
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
    """Full pipeline for one enriched matched panel file."""
    input_path = Path(input_path)
    file_label = input_path.stem.replace('_enriched', '')

    print(f'\n{"="*70}')
    print(f'  VENUE DiD ESTIMATION: {file_label}')
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

        # Extract raw ATT(g,t) — needed for aggregation but not saved
        attgt = extract_attgt(result)

        # Dynamic (event-study) aggregation → SAVE
        dynamic = aggregate_dynamic(result)
        if not isinstance(dynamic, pd.DataFrame):
            dynamic = pd.DataFrame()
        if not dynamic.empty:
            all_results[f'dynamic_{outcome_tag}'] = dynamic
            print(f'\n    Dynamic ATT (event-study):')
            for _, r in dynamic.iterrows():
                e = r.get('event_time', '?')
                a = r.get('att', 0)
                se_val = r.get('se', 0)
                pv = r.get('pvalue', np.nan)
                try:
                    a = float(a)
                    se_val = float(se_val)
                    sig = '*' if se_val > 0 and abs(a) > 1.96 * se_val else ''
                    print(f'      e={e:>3}: ATT={a:>10.2f} (SE={se_val:.2f}) p={pv:.4g}{sig}')
                except (ValueError, TypeError):
                    print(f'      e={e}: ATT={a} (SE={se_val})')

        # Overall ATT → SAVE
        overall = aggregate_overall(result)
        if not isinstance(overall, pd.DataFrame):
            overall = pd.DataFrame()
        if not overall.empty:
            all_results[f'overall_{outcome_tag}'] = overall
            try:
                att_val = float(overall.iloc[0].get('att', 0))
                pv_val = float(overall.iloc[0].get('pvalue', 0))
                print(f'\n    Overall ATT: {att_val:.2f} (p={pv_val:.4g})')
            except (ValueError, TypeError):
                print(f'\n    Overall ATT: {overall.iloc[0].to_dict()}')

        # ── 2. Cohort analysis (Figure 3) ──
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
                continue  # already handled above

            # Build covariates for this het_type: drop the splitting variable
            het_covs = covariates  # user override if provided
            if het_covs is None:
                # Use default covariates minus the ones related to this het_type
                drop_cols = COVARIATES_TO_DROP.get(het_type, [])
                het_covs = [c for c in COVARIATE_COLS if c not in drop_cols]
                print(f'\n  Covariates for {het_type} heterogeneity: '
                      f'{het_covs} (dropped {drop_cols})')

            if het_type == 'gender':
                print(f'\n  ── Gender heterogeneity: {outcome} ──')
                het_results = run_heterogeneity(
                    df, outcome, 'gender_label', ['female', 'male'],
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
        description='Venue Effect — Heterogeneous DiD via csdid')

    ap.add_argument('--input', type=str, default=None,
                    help='Single enriched CSV file')
    ap.add_argument('--input_dir', type=str, default=None,
                    help='Directory of enriched CSV files')
    ap.add_argument('--output_dir', type=str, default='../../data/did',
                    help='Output directory for results')

    ap.add_argument('--outcomes', nargs='+', default=DEFAULT_OUTCOMES,
                    help='Outcome variables to estimate')
    ap.add_argument('--heterogeneity', nargs='*', default=[],
                    choices=['gender', 'career_stage', 'region', 'cohort'],
                    help='Heterogeneity analyses to run')
    ap.add_argument('--est_method', default='reg',
                    choices=['reg', 'ipw', 'dr'],
                    help='Estimation method (reg=RA, ipw=IPW, dr=doubly robust)')
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
        cov = []  # empty list → "outcome~1"

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
        files = sorted(input_dir.glob('*_enriched*.csv'))
        print(f'\nFound {len(files)} enriched files in {input_dir}')

        for fpath in files:
            process_one_file(
                fpath, output_dir, args.outcomes, args.heterogeneity,
                est_method=args.est_method, covariates=cov,
                n_boot=args.n_boot
            )


if __name__ == '__main__':
    main()