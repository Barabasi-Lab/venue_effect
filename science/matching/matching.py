#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Matching
==============================================================================

Temporal matching framework:
  - Treated author's venue_year (first pub in selected journal) is t=0
  - We require to_year in {-5, -4, -3, -2, -1, 0, 1, ..., 10} for both
    treated and matched controls (16 years of panel data)
  - Matching uses pre-treatment window (to_year < 0) for DTW distance

Blocking (exact match, all must be non-null):
  - Gender
  - Region (set-overlap of current_year_affiliation at venue_year)
  - Career age at treatment ±2 years
  - CEM: cum_publication_count and cum_citations at to_year=-1 within ±10%

After matching:
  - Only keep treated authors who found exactly k controls
  - Report how many are dropped at each filtering step
  - Output merged panel with: venue_year, is_venue, to_year,
    venue_subregion_code, venue_region_code, matched_to, match_distance

Usage:
    python matching.py --field physics --journal_id jour.1018957 --test 50
    python matching.py --field physics --journal_id jour.1018957 --n_jobs 32
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from fastdtw import fastdtw
except ImportError:
    raise ImportError("pip install fastdtw")
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

warnings.filterwarnings('ignore')

# =============================================================================
# Config
# =============================================================================

FIELD_CONFIG = {
    'physics':   {'code': '51', 'journals': {
        'jour.1018957': 'Nature', 'jour.1346339': 'Science', 'jour.1082971': 'PNAS',
        'jour.1034717': 'Nature_Physics', 'jour.1018277': 'PRL',
        'jour.1053349': 'PRA', 'jour.1320488': 'PRB',
        'jour.1320490': 'PRC', 'jour.1320496': 'PRD', 'jour.1312290': 'PRE'}},
    'biology':   {'code': '31', 'journals': {
        'jour.1018957': 'Nature', 'jour.1346339': 'Science', 'jour.1082971': 'PNAS',
        'jour.1019114': 'Cell', 'jour.1103138': 'Nature_Genetics',
        'jour.1021344': 'Nature_Cell_Biology', 'jour.1295033': 'Nature_Struct_Mol_Bio'}},
    'chemistry': {'code': '34', 'journals': {
        'jour.1018957': 'Nature', 'jour.1346339': 'Science', 'jour.1082971': 'PNAS',
        'jour.1081898': 'JACS', 'jour.1017044': 'Angewandte_Chemie',
        'jour.1041224': 'Nature_Chemistry', 'jour.1155085': 'Chem'}},
    'sociology': {'code': '44', 'journals': {
        'jour.1013002': 'AJS', 'jour.1017026': 'ASR',
        'jour.1027842': 'ESR', 'jour.1068714': 'Social_Forces',
        'jour.1126009': 'Gender_and_Society', 'jour.1008496': 'Sociology'}},
}

DEFAULT_PARAMS = {
    'num_matches': 3,
    'pre_window': -5,
    'post_window': 10,
    'career_age_tolerance': 5,
    'caliper': None,
    'dtw_cols': ['cum_publication_count', 'cum_corresponding_count',
                 'cum_citations', 'cum_funding_count'],
    # CEM pre-filter at to_year = -1 (before DTW)
    'cem_tolerance': 0.10,    # ±10% on cumulative levels
    'cem_cols': ['cum_publication_count', 'cum_citations', 'cum_funding_count'],
}

# =============================================================================
# Country mapping
# =============================================================================
    
def load_country_mapping(path):
    """Load country_code -> subregion/region CSV. Drop rows with NaN."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['country_code', 'region', 'subregion'])
    print(f'  Country mapping: {len(df)} countries from {path}')
    print(f'    Regions: {sorted(df["region"].dropna().unique())}')
    print(f'    Subregions: {df["subregion"].nunique()} distinct')
    code_to_sub = dict(zip(df['country_code'], df['subregion']))
    code_to_reg = dict(zip(df['country_code'], df['region']))
    return code_to_sub, code_to_reg


def _to_set(x):
    """Convert a country/subregion field to a set."""
    if x is None: return set()
    if isinstance(x, (set, frozenset)): return set(x)
    if isinstance(x, (list, np.ndarray)): return {v for v in x if v is not None and isinstance(v, str)}
    if isinstance(x, str): return {x}
    return set()


def countries_to_subregions(country_list, code_to_sub):
    return {code_to_sub[c] for c in _to_set(country_list) if c in code_to_sub}


def countries_to_regions(country_list, code_to_reg):
    return {code_to_reg[c] for c in _to_set(country_list) if c in code_to_reg}


# =============================================================================
# Helpers
# =============================================================================

def read_parquet_fast(path):
    import pyarrow.dataset as ds
    t0 = time.time()
    df = ds.dataset(str(path), format='parquet').to_table().to_pandas()
    print(f'  Read {path}: {len(df):,} rows ({time.time()-t0:.1f}s)')
    return df


def require_full_window(df, pre=-5, post=10, max_missing=0):
    """
    Keep authors who have at LEAST (total - max_missing) of the required
    to_year values in [pre, ..., post].

    With max_missing=0 this is the original strict filter.
    With max_missing=2, an author missing e.g. to_year=-5 and -4 still passes.
    """
    required_years = set(range(pre, post + 1))
    author_years = df.groupby('author_id')['to_year'].apply(set)
    valid = author_years[
        author_years.apply(lambda s: len(required_years - s) <= max_missing)
    ].index
    return df[df['author_id'].isin(valid) & df['to_year'].between(pre, post)]


def compute_balance(treated_df, control_df, cols):
    rows = []
    for col in cols:
        if col not in treated_df.columns or col not in control_df.columns: continue
        tv = pd.to_numeric(treated_df[col], errors='coerce').dropna()
        cv = pd.to_numeric(control_df[col], errors='coerce').dropna()
        if len(tv) == 0 or len(cv) == 0: continue
        ps = np.sqrt((tv.var() + cv.var()) / 2)
        asd = abs(tv.mean() - cv.mean()) / ps if ps > 0 else 0
        rows.append({'variable': col, 't_mean': tv.mean(), 'c_mean': cv.mean(), 'ASD': asd})
    return pd.DataFrame(rows)


# =============================================================================
# DTW
# =============================================================================

def dtw_distance(g, gp, cols, g_means, g_stds):
    g, gp = g.sort_values('year'), gp.sort_values('year')
    X = np.column_stack([pd.to_numeric(g[c], errors='coerce').to_numpy() for c in cols])
    Y = np.column_stack([pd.to_numeric(gp[c], errors='coerce').to_numpy() for c in cols])
    X, Y = X[~np.isnan(X).any(1)], Y[~np.isnan(Y).any(1)]
    if X.size == 0 or Y.size == 0: return np.inf
    X, Y = (X - g_means) / g_stds, (Y - g_means) / g_stds
    dist, _ = fastdtw(list(map(tuple, X)), list(map(tuple, Y)), dist=euclidean)
    return dist


# =============================================================================
# Worker
# =============================================================================

_G = {}

def _init_worker(ctrl, cols, params, g_means, g_stds, code_to_reg):
    _G.update(dict(ctrl=ctrl, cols=cols, params=params,
                   g_means=g_means, g_stds=g_stds, code_to_reg=code_to_reg))


def _match_one(args):
    """Match one treated author. Returns (matched_df, author_id, n_found)."""
    key, group = args
    ref = group.iloc[0]
    ctrl = _G['ctrl']
    cols = _G['cols']
    p = _G['params']

    venue_year = ref['venue_year']
    career_age = ref['career_age_at_treatment']
    gender = ref['Gender']
    treated_regs = ref['_venue_regions']  # set of regions
    tol = p['career_age_tolerance']
    k = p['num_matches']

    # --- Blocking: Gender exact ---
    mask = ctrl['Gender'] == gender

    # --- Blocking: Career age band ---
    ctrl_ca = venue_year - ctrl['first_year_of_publication']
    mask = mask & (ctrl_ca >= career_age - tol) & (ctrl_ca <= career_age + tol)

    pot = ctrl[mask]
    if pot.empty:
        return pd.DataFrame(), ref['author_id'], 0

    # --- Blocking: Region overlap at venue_year ---
    # If the treated author has no region info (common before ~1980),
    # match against controls who also have no region info at venue_year.
    # Otherwise, require region overlap as usual.
    pot_at_vy = pot[pot['year'] == venue_year].copy()
    if pot_at_vy.empty:
        return pd.DataFrame(), ref['author_id'], 0

    code_to_reg = _G['code_to_reg']
    treated_has_region = bool(treated_regs)  # non-empty set?

    valid_ids = set()
    for _, row in pot_at_vy.iterrows():
        c_regs = countries_to_regions(row.get('current_year_affiliation_countries'), code_to_reg)
        if treated_has_region:
            # Normal case: require region overlap
            if c_regs & treated_regs:
                valid_ids.add(row['author_id'])
        else:
            # Treated has no region → accept controls that also have no region
            if not c_regs:
                valid_ids.add(row['author_id'])

    if not valid_ids:
        return pd.DataFrame(), ref['author_id'], 0

    # --- CEM pre-filter: cumulative levels at to_year = -1 (±tol%) ---
    cem_tol = p.get('cem_tolerance', 0.10)
    cem_cols = p.get('cem_cols', [])
    pre_year = venue_year - 1

    # Get treated author's values at to_year = -1
    t_at_pre = group[group['year'] == pre_year] if 'year' in group.columns else pd.DataFrame()

    if not t_at_pre.empty and cem_cols:
        # Get control values at the same calendar year
        c_at_pre = pot[(pot['author_id'].isin(valid_ids)) & (pot['year'] == pre_year)]

        for cc in cem_cols:
            if cc not in t_at_pre.columns or cc not in c_at_pre.columns:
                continue
            t_val = pd.to_numeric(t_at_pre[cc].iloc[0], errors='coerce')
            if pd.isna(t_val):
                continue
            # ±tol% band, with a floor of 3 to handle very small values
            margin = max(abs(t_val) * cem_tol, 3)
            c_vals = c_at_pre.set_index('author_id')[cc].apply(
                pd.to_numeric, errors='coerce')
            within = c_vals[
                (c_vals >= t_val - margin) & (c_vals <= t_val + margin)
            ]
            valid_ids = valid_ids & set(within.index)

        if not valid_ids:
            return pd.DataFrame(), ref['author_id'], 0

    # --- Build candidate panel with to_year ---
    pot = pot[pot['author_id'].isin(valid_ids)].copy()
    pot['to_year'] = pot['year'] - venue_year

    # Require full window
    pot = require_full_window(pot, p['pre_window'], p['post_window'])
    if pot.empty:
        return pd.DataFrame(), ref['author_id'], 0

    # --- DTW on pre-treatment ---
    pot_pre = pot[pot['to_year'] < 0]
    if pot_pre.empty:
        return pd.DataFrame(), ref['author_id'], 0

    dists = {}
    for aid, gp in pot_pre.groupby('author_id'):
        dists[aid] = dtw_distance(group, gp, cols, _G['g_means'], _G['g_stds'])

    if not dists:
        return pd.DataFrame(), ref['author_id'], 0

    # Caliper
    if p['caliper'] is not None:
        dists = {a: d for a, d in dists.items() if d <= p['caliper']}
    if not dists:
        return pd.DataFrame(), ref['author_id'], 0

    top_ids = sorted(dists, key=dists.get)[:k]
    n_found = len(top_ids)

    result = pot[pot['author_id'].isin(top_ids)].copy()
    result['match_distance'] = result['author_id'].map(dists)
    result['matched_to'] = ref['author_id']
    result['venue_year'] = venue_year

    return result, ref['author_id'], n_found


# =============================================================================
# Main matching
# =============================================================================

def run_matching(treated_df, control_df, journal_id, country_mapping_path,
                 params=None, n_jobs=1, show_progress=True):
    p = {**DEFAULT_PARAMS, **(params or {})}
    cols = p['dtw_cols']
    k = p['num_matches']
    pre, post = p['pre_window'], p['post_window']

    code_to_sub, code_to_reg = load_country_mapping(country_mapping_path)

    # =====================================================================
    # TREATED: filter step by step, report numbers at each stage
    # =====================================================================
    print('\n  --- Filtering treated ---')

    t = treated_df[treated_df['journal_id'] == journal_id].copy()
    n0 = t['author_id'].nunique()
    print(f'  [0] Journal {journal_id}: {n0:,} authors')

    if t.empty:
        return pd.DataFrame()

    # Rename for clarity
    t['venue_year'] = t['first_publish_year']

    # Filter 1: Gender must be known (0=female, 1=male; drop NaN only)
    t = t[t['Gender'].notna()]
    n1 = t['author_id'].nunique()
    print(f'  [1] Gender non-null (0=F, 1=M): {n1:,}  (dropped {n0 - n1:,})')
    print(f'      Female (0): {t.drop_duplicates("author_id")["Gender"].eq(0).sum():,}  '
          f'Male (1): {t.drop_duplicates("author_id")["Gender"].eq(1).sum():,}')

    # Filter 2: current_year_affiliation_countries non-null at venue_year
    #           For authors before 1980, skip this filter (data too sparse)
    COUNTRY_CUTOFF_YEAR = 1980

    def _has_country_at_vy(sub):
        vy = sub['venue_year'].iloc[0]
        if vy < COUNTRY_CUTOFF_YEAR:
            return True  # bypass for early authors
        row = sub[sub['year'] == vy]
        if row.empty: return False
        c = row.iloc[0].get('current_year_affiliation_countries')
        return c is not None and isinstance(c, (list, np.ndarray)) and len(c) > 0

    keep2 = set()
    for aid, grp in t.groupby('author_id'):
        if _has_country_at_vy(grp):
            keep2.add(aid)
    t = t[t['author_id'].isin(keep2)]
    n2 = t['author_id'].nunique()
    print(f'  [2] Country non-null at venue_year (skip before {COUNTRY_CUTOFF_YEAR}): '
          f'{n2:,}  (dropped {n1 - n2:,})')

    # Filter 3: full temporal window [-5, ..., 10]
    t['to_year'] = t['year'] - t['venue_year']
    t = require_full_window(t, pre, post)
    n3 = t['author_id'].nunique()
    print(f'  [3] Full window [{pre}..{post}]: {n3:,}  (dropped {n2 - n3:,})')

    if t.empty:
        return pd.DataFrame()

    # Compute venue subregions/regions for each treated author
    venue_subs = {}
    venue_regs = {}
    for aid in t['author_id'].unique():
        grp = t[t['author_id'] == aid]
        vy = grp['venue_year'].iloc[0]
        vy_row = grp[grp['year'] == vy]
        countries = vy_row.iloc[0].get('current_year_affiliation_countries') if not vy_row.empty else None
        venue_subs[aid] = countries_to_subregions(countries, code_to_sub)
        venue_regs[aid] = countries_to_regions(countries, code_to_reg)

    # Store regions on treated df for blocking in _match_one
    t['_venue_regions'] = t['author_id'].map(venue_regs)

    # =====================================================================
    # CONTROL: filter step by step
    # =====================================================================
    print('\n  --- Filtering control ---')
    c = control_df.copy()
    nc0 = c['author_id'].nunique()
    print(f'  [0] Total control: {nc0:,}')

    c = c[c['Gender'].notna()]
    nc1 = c['author_id'].nunique()
    print(f'  [1] Gender non-null (0=F, 1=M): {nc1:,}  (dropped {nc0 - nc1:,})')

    # =====================================================================
    # Global standardization stats
    # =====================================================================
    g_means = np.array([pd.to_numeric(c[col], errors='coerce').dropna().mean() for col in cols])
    g_stds  = np.array([pd.to_numeric(c[col], errors='coerce').dropna().std(ddof=1) for col in cols])
    g_stds[g_stds == 0] = 1.0

    # =====================================================================
    # Build tasks (pre-treatment rows for each treated author)
    # =====================================================================
    t_pre = t[t['to_year'] < 0]
    groups = list(t_pre.groupby(['author_id', 'venue_year']))
    tasks = [(key, grp.copy()) for key, grp in groups]
    print(f'\n  Matching {len(tasks):,} treated against {nc1:,} controls ...')
    print(f'  Blocking: Gender + Region (null-null for pre-{COUNTRY_CUTOFF_YEAR}) + Career age +/-{p["career_age_tolerance"]}yr')
    print(f'  CEM pre-filter: +/-{p.get("cem_tolerance", 0.10)*100:.0f}% on {p.get("cem_cols", [])} at to_year=-1')

    # =====================================================================
    # Run matching
    # =====================================================================
    t0 = time.time()
    results = []
    match_counts = {}

    if n_jobs <= 1:
        _init_worker(c, cols, p, g_means, g_stds, code_to_reg)
        it = tqdm(tasks, desc='Matching') if show_progress else tasks
        for task in it:
            res, aid, n_found = _match_one(task)
            match_counts[aid] = n_found
            if not isinstance(res, pd.DataFrame): continue
            if not res.empty:
                results.append(res)
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker,
            initargs=(c, cols, p, g_means, g_stds, code_to_reg),
        ) as ex:
            futs = {ex.submit(_match_one, t_): t_ for t_ in tasks}
            it = as_completed(futs)
            if show_progress:
                it = tqdm(it, total=len(futs), desc='Matching')
            for fut in it:
                res, aid, n_found = fut.result()
                match_counts[aid] = n_found
                if not isinstance(res, pd.DataFrame): continue
                if not res.empty:
                    results.append(res)

    elapsed = time.time() - t0

    if not results:
        print(f'\n  WARNING: No matches found ({elapsed:.0f}s)')
        return pd.DataFrame()

    matched_raw = pd.concat(results, ignore_index=True)

    # =====================================================================
    # Report match counts and enforce exactly k
    # =====================================================================
    mc = pd.Series(match_counts)
    n_exact_k = (mc == k).sum()
    n_fewer = ((mc > 0) & (mc < k)).sum()
    n_zero = (mc == 0).sum()

    print(f'\n  --- Match results ({elapsed:.0f}s) ---')
    print(f'    Exactly {k} matches: {n_exact_k:,}')
    print(f'    1 to {k-1} matches:   {n_fewer:,}  -> DROPPED')
    print(f'    0 matches:          {n_zero:,}  -> DROPPED')

    keep_ids = set(mc[mc == k].index)
    n_dropped = len(tasks) - len(keep_ids)
    print(f'    KEEPING {len(keep_ids):,} treated, DROPPING {n_dropped:,}')

    matched = matched_raw[matched_raw['matched_to'].isin(keep_ids)].copy()
    t_keep = t[t['author_id'].isin(keep_ids)].copy()

    # =====================================================================
    # Build merged output
    # =====================================================================

    # --- Treated panel ---
    t_out = t_keep.copy()
    t_out['is_venue'] = (t_out['to_year'] >= 0).astype(int)
    t_out['venue_subregion_code'] = t_out['author_id'].map(
        lambda x: sorted(venue_subs.get(x, [])) or None)
    t_out['venue_region_code'] = t_out['author_id'].map(
        lambda x: sorted(venue_regs.get(x, [])) or None)
    # Drop internal column
    t_out = t_out.drop(columns=['_venue_regions'], errors='ignore')

    # --- Matched control panel ---
    m_out = matched.copy()
    m_out['is_venue'] = 0

    # Venue subregion/region for controls: their OWN affiliations at venue_year
    def _ctrl_geo(row):
        countries = row.get('current_year_affiliation_countries')
        sub = sorted(countries_to_subregions(countries, code_to_sub)) or None
        reg = sorted(countries_to_regions(countries, code_to_reg)) or None
        return pd.Series({'venue_subregion_code': sub, 'venue_region_code': reg})

    if not m_out.empty:
        geo = m_out.apply(_ctrl_geo, axis=1)
        m_out['venue_subregion_code'] = geo['venue_subregion_code']
        m_out['venue_region_code'] = geo['venue_region_code']

    # --- Concat ---
    common = sorted(set(t_out.columns) & set(m_out.columns))
    merged = pd.concat([t_out[common], m_out[common]], ignore_index=True)
    merged = merged.sort_values(['author_id', 'year']).reset_index(drop=True)

    # =====================================================================
    # Adjust for focal publication
    # =====================================================================
    print('\n  --- Adjusting for focal publication ---')

    treat_post = merged['author_id'].isin(keep_ids) & (merged['to_year'] >= 0)
    ctrl_post = (~merged['author_id'].isin(keep_ids)) & (merged['to_year'] >= 0)

    # --- Publication count ---
    if 'publication_count' in merged.columns:
        # Annual adjusted
        merged['publication_count_adj'] = merged['publication_count'].copy()
        merged.loc[treat_post, 'publication_count_adj'] = \
            (merged.loc[treat_post, 'publication_count'] - 1).clip(lower=0)
        ctrl_has_pub = ctrl_post & (merged['publication_count'] >= 1)
        merged.loc[ctrl_has_pub, 'publication_count_adj'] = \
            merged.loc[ctrl_has_pub, 'publication_count'] - 1

        # Cumulative adjusted: subtract 1 from original cum at to_year >= 0
        if 'cum_publication_count' in merged.columns:
            merged['cum_publication_count_adj'] = merged['cum_publication_count'].copy()
            merged.loc[treat_post, 'cum_publication_count_adj'] = \
                (merged.loc[treat_post, 'cum_publication_count'] - 1).clip(lower=0)
            merged.loc[ctrl_has_pub, 'cum_publication_count_adj'] = \
                merged.loc[ctrl_has_pub, 'cum_publication_count'] - 1

        print(f'    Created: publication_count_adj, cum_publication_count_adj')

    # =====================================================================
    # Balance check
    # =====================================================================
    print('\n  --- Covariate Balance (to_year = -1) ---')
    t_bal = t_out[t_out['to_year'] == -1]
    m_bal = m_out[m_out['to_year'] == -1]
    bal_cols = list(set(cols) | set(p.get('cem_cols', [])))
    bal = compute_balance(t_bal, m_bal, bal_cols)
    if not bal.empty:
        for _, r in bal.iterrows():
            flag = '✓' if r['ASD'] < 0.1 else ('~' if r['ASD'] < 0.25 else '✗')
            print(f'    {flag} {r["variable"]:<35s}  ASD={r["ASD"]:.3f}  '
                  f'(T={r["t_mean"]:.1f} C={r["c_mean"]:.1f})')

    # Final summary
    print(f'\n  Final merged: {len(merged):,} rows, '
          f'{merged["author_id"].nunique():,} unique authors')

    # Venue year range
    t_merged = merged[merged['author_id'].isin(keep_ids)]
    vy_min = t_merged['venue_year'].min()
    vy_max = t_merged['venue_year'].max()
    print(f'  Venue year range (treated): {vy_min} - {vy_max}')

    return merged


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Venue Effect Matching')
    ap.add_argument('--field', required=True, choices=list(FIELD_CONFIG.keys()))
    ap.add_argument('--journal_id', required=True)
    ap.add_argument('--data_dir', default='../../data/matching_needed')
    ap.add_argument('--output_dir', default='../../data/matches')
    ap.add_argument('--country_map', default='../../data/matching_needed/country_region.csv')
    ap.add_argument('--n_jobs', type=int, default=1)
    ap.add_argument('--num_matches', type=int, default=3)
    ap.add_argument('--test', type=int, default=0)
    ap.add_argument('--caliper', type=float, default=None)
    ap.add_argument('--pre_window', type=int, default=-5)
    ap.add_argument('--career_age_tol', type=int, default=2)
    ap.add_argument('--cem_tol', type=float, default=0.10,
                    help='CEM tolerance for cum levels at to_year=-1 (default 0.10 = +/-10%%)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jlabel = FIELD_CONFIG[args.field]['journals'].get(args.journal_id, args.journal_id)
    print(f'=== Matching: {args.field} / {jlabel} ({args.journal_id}) ===')

    treated_df = read_parquet_fast(data_dir / f'treated_df_{args.field}.parquet')
    control_df = read_parquet_fast(data_dir / f'control_df_{args.field}.parquet')

    if args.test > 0:
        test_ids = treated_df[treated_df['journal_id'] == args.journal_id] \
            .drop_duplicates('author_id')['author_id'].head(args.test).tolist()
        treated_df = treated_df[treated_df['author_id'].isin(test_ids)]
        print(f'  TEST MODE: {args.test} treated authors')

    params = {
        'num_matches': args.num_matches,
        'pre_window': args.pre_window,
        'caliper': args.caliper,
        'career_age_tolerance': args.career_age_tol,
        'cem_tolerance': args.cem_tol,
    }

    merged = run_matching(treated_df, control_df, args.journal_id,
                          args.country_map, params=params, n_jobs=args.n_jobs)

    if merged.empty:
        print('Nothing to save.')
        return

    safe = jlabel.replace(' ', '_').replace('/', '_').replace('&', 'and')
    out_path = output_dir / f'merged_{args.field}_{safe}.csv'

    # Convert list/array columns to pipe-joined strings for CSV
    save_df = merged.copy()
    for col in save_df.columns:
        if save_df[col].dtype == object:
            sample = save_df[col].dropna().head(50)
            if sample.apply(type).eq(list).any() or sample.apply(type).eq(np.ndarray).any():
                save_df[col] = save_df[col].apply(
                    lambda x: '|'.join(str(v) for v in x) if isinstance(x, (list, np.ndarray)) else x
                )

    save_df.to_csv(out_path, index=False, sep=';')
    print(f'\n  Saved: {out_path.name}')
    print(f'    {merged["author_id"].nunique():,} unique authors, {len(merged):,} rows')


if __name__ == '__main__':
    main()