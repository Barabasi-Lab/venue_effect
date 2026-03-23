#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Art Biennale Matching
==============================================================================

Temporal matching framework for biennale artists:
  - Treated artist's biennale_year (first appearance at venue) is t=0
  - We require year_diff in {-5, -4, ..., 0, 1, ..., 10} (16 years)
  - Matching uses pre-treatment window (year_diff < 0) for DTW distance

Blocking (all must pass):
  1. Gender fuzzy match (mostly_male→male, mostly_female→female)
  2. Continent overlap (≥1 shared continent between treated & control)
  3. Career age ±2 years (career age at biennale year)
  4. CEM: cumulative S, G, B, F at year_diff=-1 within ±30% (or floor)

After blocking + CEM:
  - DTW distance on pre-treatment S, G, F, B trajectories
  - Slope filter: walk through DTW-ranked candidates, reject those whose
    pre-treatment OLS slope in S or G diverges from treated by > slope_max_diff
  - Control usage cap: each control can be matched at most max_control_usage
    times (default 3). This prevents a small set of controls from being
    reused excessively, improving effective sample size.
  - Select k nearest passing controls per treated artist
  - Only keep treated artists who found exactly k matches

Usage:
    python matching_art.py --venue venice_biennale --n_jobs 16
    python matching_art.py --venue venice_biennale --n_jobs 16 --max_control_usage 5
    python matching_art.py --venue venice_biennale --n_jobs 16 --max_control_usage 0
    python matching_art.py --venue all --n_jobs 16
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path
from collections import Counter

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
    def tqdm(it, **kw):
        return it

warnings.filterwarnings('ignore')

# =============================================================================
# Config
# =============================================================================

VENUE_CONFIG = {
    'venice_biennale':    {'id': 2367, 'tier': 'top',   'label': 'Venice Biennale'},
    'documenta':          {'id': 2538, 'tier': 'top',   'label': 'Documenta'},
    'bienal_sao_paulo':   {'id': 2549, 'tier': 'lower', 'label': 'Bienal de São Paulo'},
    'whitney_biennial':   {'id': 5722, 'tier': 'lower', 'label': 'Whitney Biennial'},
    'biennale_of_sydney': {'id': 4081, 'tier': 'lower', 'label': 'Biennale of Sydney'},
    'istanbul_biennial':  {'id': 3933, 'tier': 'lower', 'label': 'Istanbul Biennial'},
    'manifesta':          {'id': 6058, 'tier': 'lower', 'label': 'Manifesta'},
    'gwangju_biennale':   {'id': 3458, 'tier': 'lower', 'label': 'Gwangju Biennale'},
}

CONTINENT_COLS = ['Europe', 'North America', 'South America',
                  'Asia', 'Oceania', 'Africa']

GENDER_FUZZY_MAP = {
    'male': 'male',
    'female': 'female',
    'mostly_male': 'male',
    'mostly_female': 'female',
}

DEFAULT_PARAMS = {
    'num_matches': 3,
    'pre_window': -5,
    'post_window': 10,
    'career_age_tolerance': 2,
    'caliper': None,
    'dtw_cols': ['S', 'G', 'F', 'B'],
    'cem_tolerance': 0.30,
    'cem_floor': 3,
    'cem_cols': [
        ('S', 2),
        ('G', None),
        ('B', 2),
        ('F', 2),
    ],
    'slope_max_diff': 0.15,
    'slope_cols': ['S', 'G'],
    'max_control_usage': 3,
}


# =============================================================================
# Helpers
# =============================================================================

def require_full_window(df, pre=-5, post=10, id_col='artist_id',
                        time_col='year_diff', max_missing=0):
    required = set(range(pre, post + 1))
    artist_years = df.groupby(id_col)[time_col].apply(set)
    valid = artist_years[
        artist_years.apply(lambda s: len(required - s) <= max_missing)
    ].index
    return df[df[id_col].isin(valid) & df[time_col].between(pre, post)]


def compute_balance(treated_df, control_df, cols):
    rows = []
    for col in cols:
        if col not in treated_df.columns or col not in control_df.columns:
            continue
        tv = pd.to_numeric(treated_df[col], errors='coerce').dropna()
        cv = pd.to_numeric(control_df[col], errors='coerce').dropna()
        if len(tv) == 0 or len(cv) == 0:
            continue
        ps = np.sqrt((tv.var() + cv.var()) / 2)
        asd = abs(tv.mean() - cv.mean()) / ps if ps > 0 else 0
        rows.append({'variable': col, 't_mean': tv.mean(),
                     'c_mean': cv.mean(), 'ASD': asd})
    return pd.DataFrame(rows)


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return ((x - xm) * (y - y.mean())).sum() / denom


# =============================================================================
# DTW
# =============================================================================

def dtw_distance(treated_pre, control_pre, cols, g_means, g_stds):
    t = treated_pre.sort_values('end_year')
    c = control_pre.sort_values('end_year')

    X = np.column_stack([
        pd.to_numeric(t[col], errors='coerce').to_numpy() for col in cols
    ])
    Y = np.column_stack([
        pd.to_numeric(c[col], errors='coerce').to_numpy() for col in cols
    ])

    X = X[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]

    if X.size == 0 or Y.size == 0:
        return np.inf

    X = (X - g_means) / g_stds
    Y = (Y - g_means) / g_stds

    dist, _ = fastdtw(list(map(tuple, X)), list(map(tuple, Y)), dist=euclidean)
    return dist


# =============================================================================
# Single-artist matching (called sequentially to enforce usage cap)
# =============================================================================

def match_one_artist(artist_id, treated_group, ctrl, cols, params,
                     g_means, g_stds, usage_counter, max_usage):
    ref = treated_group.iloc[0]
    p = params

    biennale_year = ref['biennale_year']
    k = p['num_matches']

    # Block 1: Gender fuzzy
    gender = ref.get('gender_fuzzy')
    if pd.isna(gender):
        return pd.DataFrame(), artist_id, 0
    mask = ctrl['_gender'] == gender

    # Block 2: Continent overlap
    treated_conts = {c: int(ref.get(c, 0)) for c in CONTINENT_COLS}
    treated_has_any = any(v == 1 for v in treated_conts.values())
    if treated_has_any:
        cont_mask = np.zeros(len(ctrl), dtype=bool)
        for c in CONTINENT_COLS:
            if treated_conts.get(c, 0) == 1 and f'_{c}' in ctrl.columns:
                cont_mask |= (ctrl[f'_{c}'].values == 1)
        mask = mask & cont_mask

    # Block 3: Career age ±tolerance
    career_age = ref.get('career_age')
    tol = p['career_age_tolerance']
    if pd.notna(career_age):
        mask = mask & (
            ctrl['_career_age'].between(career_age - tol, career_age + tol)
        )

    # Usage cap: exclude controls already at max
    if max_usage > 0:
        saturated = {aid for aid, cnt in usage_counter.items() if cnt >= max_usage}
        if saturated:
            mask = mask & (~ctrl['artist_id'].isin(saturated))

    pot_ids = ctrl.loc[mask, 'artist_id'].unique()
    if len(pot_ids) == 0:
        return pd.DataFrame(), artist_id, 0

    pot = ctrl[ctrl['artist_id'].isin(pot_ids)].copy()
    pot['year_diff'] = pot['end_year'] - biennale_year
    pot = require_full_window(pot, p['pre_window'], p['post_window'])
    if pot.empty:
        return pd.DataFrame(), artist_id, 0
    pot_ids = set(pot['artist_id'].unique())

    # Block 4: CEM at year_diff=-1
    cem_tol = p.get('cem_tolerance', 0.30)
    default_floor = p.get('cem_floor', 2)
    raw_cem_cols = p.get('cem_cols', [])

    cem_specs = []
    for entry in raw_cem_cols:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            col_name, floor_override = entry
            floor_val = floor_override if floor_override is not None else default_floor
            cem_specs.append((col_name, floor_val))
        else:
            cem_specs.append((str(entry), default_floor))

    t_at_m1 = treated_group[treated_group['year_diff'] == -1]
    if not t_at_m1.empty and cem_specs:
        c_at_m1 = pot[pot['year_diff'] == -1]
        for cc, floor_val in cem_specs:
            if cc not in t_at_m1.columns or cc not in c_at_m1.columns:
                continue
            t_val = pd.to_numeric(t_at_m1[cc].iloc[0], errors='coerce')
            if pd.isna(t_val):
                continue
            margin = max(abs(t_val) * cem_tol, floor_val)
            c_vals = c_at_m1.set_index('artist_id')[cc].apply(
                pd.to_numeric, errors='coerce'
            )
            within = c_vals[
                (c_vals >= t_val - margin) & (c_vals <= t_val + margin)
            ]
            pot_ids = pot_ids & set(within.index)
        if not pot_ids:
            return pd.DataFrame(), artist_id, 0

    # DTW on pre-treatment
    pot = pot[pot['artist_id'].isin(pot_ids)]
    pot_pre = pot[pot['year_diff'] < 0]
    if pot_pre.empty:
        return pd.DataFrame(), artist_id, 0

    t_pre = treated_group[treated_group['year_diff'] < 0]

    dists = {}
    for aid, gp in pot_pre.groupby('artist_id'):
        dists[aid] = dtw_distance(t_pre, gp, cols, g_means, g_stds)
    if not dists:
        return pd.DataFrame(), artist_id, 0

    if p['caliper'] is not None:
        dists = {a: d for a, d in dists.items() if d <= p['caliper']}
    if not dists:
        return pd.DataFrame(), artist_id, 0

    # Slope filter + usage cap in DTW-ranked walk
    slope_max = p.get('slope_max_diff')
    slope_cols = p.get('slope_cols', ['S', 'G'])

    ranked_ids = sorted(dists, key=dists.get)

    if slope_max and slope_max > 0 and slope_cols:
        t_pre_sorted = t_pre.sort_values('year_diff')
        t_x = t_pre_sorted['year_diff'].values
        t_slopes = {}
        for sc in slope_cols:
            if sc in t_pre_sorted.columns:
                t_slopes[sc] = ols_slope(
                    t_x,
                    pd.to_numeric(t_pre_sorted[sc], errors='coerce').values
                )

        top_ids = []
        for aid in ranked_ids:
            if len(top_ids) >= k:
                break
            if max_usage > 0 and usage_counter.get(aid, 0) >= max_usage:
                continue

            c_pre_i = pot_pre[pot_pre['artist_id'] == aid].sort_values('year_diff')
            if c_pre_i.empty:
                continue

            c_x = c_pre_i['year_diff'].values
            passes = True
            for sc in slope_cols:
                if sc not in c_pre_i.columns or sc not in t_slopes:
                    continue
                t_sl = t_slopes[sc]
                if np.isnan(t_sl):
                    continue
                c_sl = ols_slope(
                    c_x,
                    pd.to_numeric(c_pre_i[sc], errors='coerce').values
                )
                if np.isnan(c_sl) or abs(t_sl - c_sl) > slope_max:
                    passes = False
                    break

            if passes:
                top_ids.append(aid)
    else:
        top_ids = []
        for aid in ranked_ids:
            if len(top_ids) >= k:
                break
            if max_usage > 0 and usage_counter.get(aid, 0) >= max_usage:
                continue
            top_ids.append(aid)

    n_found = len(top_ids)
    if n_found == 0:
        return pd.DataFrame(), artist_id, 0

    result = pot[pot['artist_id'].isin(top_ids)].copy()
    result['match_distance'] = result['artist_id'].map(dists)
    result['matched_to'] = artist_id
    result['biennale_year'] = biennale_year

    return result, artist_id, n_found


# =============================================================================
# Parallel worker (used when max_control_usage is disabled)
# =============================================================================

_G = {}


def _init_worker(ctrl, cols, params, g_means, g_stds):
    _G.update(dict(ctrl=ctrl, cols=cols, params=params,
                   g_means=g_means, g_stds=g_stds))


def _match_one_parallel(args):
    artist_id, treated_group = args
    dummy_counter = Counter()
    return match_one_artist(
        artist_id, treated_group,
        _G['ctrl'], _G['cols'], _G['params'],
        _G['g_means'], _G['g_stds'],
        dummy_counter, max_usage=0,
    )


# =============================================================================
# Main matching
# =============================================================================

def run_matching(treated_df, control_df, venue_key, params=None,
                 n_jobs=1, show_progress=True):
    p = {**DEFAULT_PARAMS, **(params or {})}
    cols = p['dtw_cols']
    k = p['num_matches']
    pre, post = p['pre_window'], p['post_window']
    max_usage = p.get('max_control_usage', 0)

    cfg = VENUE_CONFIG[venue_key]
    label = cfg['label']

    # TREATED
    print(f'\n  --- Filtering treated ({label}) ---')
    t = treated_df.copy()
    n0 = t['artist_id'].nunique()
    print(f'  [0] Total treated: {n0:,}')

    t['gender_fuzzy'] = t['gender_est'].map(GENDER_FUZZY_MAP)
    t = t[t['gender_fuzzy'].notna()]
    n1 = t['artist_id'].nunique()
    print(f'  [1] Usable gender (fuzzy): {n1:,}  (dropped {n0 - n1:,})')
    g_orig = t.drop_duplicates('artist_id')['gender_est'].value_counts()
    g_fuzzy = t.drop_duplicates('artist_id')['gender_fuzzy'].value_counts()
    print(f'      Original: {g_orig.to_dict()}')
    print(f'      Fuzzy:    {g_fuzzy.to_dict()}')

    t = require_full_window(t, pre, post)
    n2 = t['artist_id'].nunique()
    print(f'  [2] Full window [{pre}..{post}]: {n2:,}  (dropped {n1 - n2:,})')

    if t.empty:
        print('  No treated artists remain.')
        return pd.DataFrame()

    # CONTROL
    print(f'\n  --- Filtering control ---')
    c = control_df.copy()
    nc0 = c['artist_id'].nunique()
    print(f'  [0] Total control: {nc0:,}')

    c['gender_fuzzy'] = c['gender_est'].map(GENDER_FUZZY_MAP)
    c = c[c['gender_fuzzy'].notna()]
    nc1 = c['artist_id'].nunique()
    print(f'  [1] Usable gender (fuzzy): {nc1:,}  (dropped {nc0 - nc1:,})')

    c['_gender'] = c['gender_fuzzy']
    c['_career_age'] = pd.to_numeric(c['career_age'], errors='coerce')
    for cont in CONTINENT_COLS:
        c[f'_{cont}'] = pd.to_numeric(c.get(cont, 0), errors='coerce').fillna(0).astype(int)

    # DTW stats
    g_means = np.array([
        pd.to_numeric(c[col], errors='coerce').dropna().mean() for col in cols
    ])
    g_stds = np.array([
        pd.to_numeric(c[col], errors='coerce').dropna().std(ddof=1) for col in cols
    ])
    g_stds[g_stds == 0] = 1.0

    # Build tasks
    tasks = [(aid, grp.copy()) for aid, grp in t.groupby('artist_id')]

    print(f'\n  Matching {len(tasks):,} treated against {nc1:,} controls ...')
    print(f'  Blocking: Gender (fuzzy) + Continent overlap + Career age '
          f'±{p["career_age_tolerance"]}yr')
    cem_desc = []
    for entry in p.get('cem_cols', []):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            col_name, floor_override = entry
            fl = floor_override if floor_override is not None else p.get('cem_floor', 2)
            cem_desc.append(f'{col_name}(floor={fl})')
        else:
            cem_desc.append(f'{entry}(floor={p.get("cem_floor", 2)})')
    print(f'  CEM: ±{p["cem_tolerance"]*100:.0f}% on [{", ".join(cem_desc)}] '
          f'at year_diff=-1')
    print(f'  DTW on: {cols}')

    slope_max = p.get('slope_max_diff')
    slope_cols = p.get('slope_cols', ['S', 'G'])
    if slope_max and slope_max > 0:
        print(f'  Slope filter: ±{slope_max}/yr on {slope_cols}')
    else:
        print(f'  Slope filter: DISABLED')

    if max_usage > 0:
        print(f'  Control usage cap: each control used at most {max_usage} times')
        print(f'  NOTE: Sequential matching required (n_jobs=1 enforced)')
    else:
        print(f'  Control usage cap: UNLIMITED')

    print(f'  k = {k}')

    # Run
    t0 = time.time()
    results = []
    match_counts = {}

    if max_usage > 0:
        # ── Sequential matching with usage tracking ──
        rng = np.random.default_rng(42)
        task_order = list(range(len(tasks)))
        rng.shuffle(task_order)

        usage_counter = Counter()

        it = tqdm(task_order, desc='Matching (sequential)') if show_progress else task_order
        for idx in it:
            aid, grp = tasks[idx]
            res, aid_out, n_found = match_one_artist(
                aid, grp, c, cols, p, g_means, g_stds,
                usage_counter, max_usage,
            )
            match_counts[aid_out] = n_found

            if isinstance(res, pd.DataFrame) and not res.empty:
                results.append(res)
                # Update usage counter for selected controls
                if n_found == k:
                    matched_ctrl_ids = res['artist_id'].unique()
                    for ctrl_id in matched_ctrl_ids:
                        usage_counter[ctrl_id] += 1

        # Report usage stats
        if usage_counter:
            counts = list(usage_counter.values())
            print(f'\n  Control usage stats:')
            print(f'    Unique controls used: {len(usage_counter):,}')
            print(f'    Usage: mean={np.mean(counts):.1f}, '
                  f'median={np.median(counts):.0f}, '
                  f'max={max(counts)}, '
                  f'at_cap={sum(1 for v in counts if v >= max_usage):,}')

    elif n_jobs <= 1:
        # ── Sequential, no usage cap ──
        _init_worker(c, cols, p, g_means, g_stds)
        it = tqdm(tasks, desc='Matching') if show_progress else tasks
        for task in it:
            res, aid, n_found = _match_one_parallel(task)
            match_counts[aid] = n_found
            if isinstance(res, pd.DataFrame) and not res.empty:
                results.append(res)
    else:
        # ── Parallel, no usage cap ──
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker,
            initargs=(c, cols, p, g_means, g_stds),
        ) as ex:
            futs = {ex.submit(_match_one_parallel, t_): t_ for t_ in tasks}
            it = as_completed(futs)
            if show_progress:
                it = tqdm(it, total=len(futs), desc='Matching')
            for fut in it:
                res, aid, n_found = fut.result()
                match_counts[aid] = n_found
                if isinstance(res, pd.DataFrame) and not res.empty:
                    results.append(res)

    elapsed = time.time() - t0

    if not results:
        print(f'\n  WARNING: No matches found ({elapsed:.0f}s)')
        return pd.DataFrame()

    matched_raw = pd.concat(results, ignore_index=True)

    # Enforce exactly k
    mc = pd.Series(match_counts)
    n_exact = (mc == k).sum()
    n_fewer = ((mc > 0) & (mc < k)).sum()
    n_zero = (mc == 0).sum()

    print(f'\n  --- Match results ({elapsed:.0f}s) ---')
    print(f'    Exactly {k} matches: {n_exact:,}')
    print(f'    1 to {k-1} matches:   {n_fewer:,}  → DROPPED')
    print(f'    0 matches:          {n_zero:,}  → DROPPED')

    keep_ids = set(mc[mc == k].index)
    print(f'    KEEPING {len(keep_ids):,} treated')

    matched = matched_raw[matched_raw['matched_to'].isin(keep_ids)].copy()
    t_keep = t[t['artist_id'].isin(keep_ids)].copy()

    # Report control reuse in final output
    if not matched.empty:
        unique_pairs = matched[['artist_id', 'matched_to']].drop_duplicates()
        pair_counts = unique_pairs['artist_id'].value_counts()
        print(f'\n  Final control reuse:')
        print(f'    Unique controls: {len(pair_counts):,}')
        print(f'    Times matched: mean={pair_counts.mean():.1f}, '
              f'median={pair_counts.median():.0f}, '
              f'max={pair_counts.max()}')
        if max_usage > 0:
            print(f'    Controls at cap ({max_usage}): '
                  f'{(pair_counts >= max_usage).sum():,}')

    # Build merged
    t_out = t_keep.copy()
    t_out['is_venue'] = (t_out['year_diff'] >= 0).astype(int)
    t_out['matched_to'] = t_out['artist_id']
    t_out['match_distance'] = 0.0

    m_out = matched.copy()
    m_out['is_venue'] = 0

    drop_cols = [f'_{cont}' for cont in CONTINENT_COLS] + ['_gender', '_career_age']
    m_out = m_out.drop(columns=[c_ for c_ in drop_cols if c_ in m_out.columns],
                       errors='ignore')

    common = sorted(set(t_out.columns) & set(m_out.columns))
    merged = pd.concat([t_out[common], m_out[common]], ignore_index=True)
    merged = merged.sort_values(['artist_id', 'end_year']).reset_index(drop=True)

    # B adjustment
    print('\n  --- Adjusting cumulative B for biennale appearances ---')
    if 'B' in merged.columns:
        merged['B_adj'] = merged['B'].copy()
        b_baseline = (
            merged.loc[merged['year_diff'] == -1, ['artist_id', 'B']]
            .drop_duplicates(subset='artist_id')
            .set_index('artist_id')['B']
            .rename('_b_baseline')
        )
        merged = merged.merge(
            b_baseline, left_on='artist_id', right_index=True, how='left'
        )
        merged['_b_baseline'] = merged['_b_baseline'].fillna(0)

        post_mask = merged['year_diff'] >= 0
        b_increased = post_mask & (merged['B'] > merged['_b_baseline'])
        merged.loc[b_increased, 'B_adj'] = (
            merged.loc[b_increased, 'B'] - 1
        ).clip(lower=0)
        merged = merged.drop(columns=['_b_baseline'])

        n_adj_treat = b_increased[merged['artist_id'].isin(keep_ids)].sum()
        n_adj_ctrl = b_increased[~merged['artist_id'].isin(keep_ids)].sum()
        print(f'    B_adj: subtracted 1 where B increased post-treatment')
        print(f'    Treated rows adjusted: {n_adj_treat:,}')
        print(f'    Control rows adjusted: {n_adj_ctrl:,}')

    # Balance
    print('\n  --- Covariate Balance (year_diff = -1) ---')
    t_bal = t_out[t_out['year_diff'] == -1]
    m_bal = m_out[m_out['year_diff'] == -1]
    cem_col_names = set()
    for entry in p.get('cem_cols', []):
        if isinstance(entry, (list, tuple)):
            cem_col_names.add(entry[0])
        else:
            cem_col_names.add(str(entry))
    bal_cols = list(set(cols) | cem_col_names)
    bal = compute_balance(t_bal, m_bal, bal_cols)
    if not bal.empty:
        for _, r in bal.iterrows():
            flag = '✓' if r['ASD'] < 0.1 else ('~' if r['ASD'] < 0.25 else '✗')
            print(f'    {flag} {r["variable"]:<15s}  ASD={r["ASD"]:.3f}  '
                  f'(T={r["t_mean"]:.1f} C={r["c_mean"]:.1f})')

    n_treat = merged[merged['artist_id'].isin(keep_ids)]['artist_id'].nunique()
    n_ctrl = merged[~merged['artist_id'].isin(keep_ids)]['artist_id'].nunique()
    vy_min = merged.loc[merged['artist_id'].isin(keep_ids), 'biennale_year'].min()
    vy_max = merged.loc[merged['artist_id'].isin(keep_ids), 'biennale_year'].max()

    print(f'\n  Final merged: {len(merged):,} rows')
    print(f'    Treated: {n_treat:,}  Control: {n_ctrl:,}')
    print(f'    Biennale year range: {vy_min} – {vy_max}')

    return merged


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Art Biennale Matching')
    ap.add_argument('--venue', required=True,
                    help='Venue key (e.g., venice_biennale) or "all"')
    ap.add_argument('--data_dir', default='../../data')
    ap.add_argument('--output_dir', default='../../data/matches')
    ap.add_argument('--n_jobs', type=int, default=1)
    ap.add_argument('--num_matches', type=int, default=3)
    ap.add_argument('--test', type=int, default=0,
                    help='Test mode: match only N treated artists')
    ap.add_argument('--caliper', type=float, default=None)
    ap.add_argument('--career_age_tol', type=int, default=2)
    ap.add_argument('--cem_tol', type=float, default=0.30)
    ap.add_argument('--cem_floor', type=int, default=3)
    ap.add_argument('--slope_max_diff', type=float, default=0.15,
                    help='Max slope difference in S/G pre-treatment. 0=disable.')
    ap.add_argument('--max_control_usage', type=int, default=3,
                    help='Max times a control can be reused. 0=unlimited.')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.venue == 'all':
        venues = list(VENUE_CONFIG.keys())
    else:
        venues = [args.venue]

    params = {
        'num_matches': args.num_matches,
        'caliper': args.caliper,
        'career_age_tolerance': args.career_age_tol,
        'cem_tolerance': args.cem_tol,
        'cem_floor': args.cem_floor,
        'slope_max_diff': args.slope_max_diff if args.slope_max_diff > 0 else None,
        'max_control_usage': args.max_control_usage,
    }

    ctrl_path = data_dir / 'matching_needed' / 'control_df_biennale.parquet'
    print(f'Loading control: {ctrl_path}')
    control_all = pd.read_parquet(ctrl_path)
    print(f'  → {control_all["artist_id"].nunique():,} artists, '
          f'{len(control_all):,} rows')

    id_list_path = data_dir / 'matching_needed' / 'venue_artist_id_lists.pkl'
    venue_id_lists = {}
    if id_list_path.exists():
        with open(id_list_path, 'rb') as f:
            venue_id_lists = pickle.load(f)
        print(f'Loaded venue ID lists: {list(venue_id_lists.keys())}')

    for venue_key in venues:
        if venue_key not in VENUE_CONFIG:
            print(f'\nWARNING: Unknown venue "{venue_key}", skipping')
            continue

        cfg = VENUE_CONFIG[venue_key]
        label = cfg['label']
        tier = cfg['tier']

        print('\n' + '=' * 70)
        print(f'MATCHING: {label} ({venue_key}, tier={tier})')
        print('=' * 70)

        treated_path = (data_dir / 'matching_needed' /
                        f'treated_df_{venue_key}.parquet')
        if not treated_path.exists():
            treated_path = (data_dir / 'artist_info' /
                            f'treated_df_{venue_key}.parquet')
        if not treated_path.exists():
            print(f'  SKIP: {treated_path} not found')
            continue

        print(f'  Loading treated: {treated_path}')
        treated_df = pd.read_parquet(treated_path)
        print(f'  → {treated_df["artist_id"].nunique():,} artists')

        control_df = control_all.copy()

        if tier == 'lower':
            excl_ids = venue_id_lists.get(venue_key, set())
            if excl_ids:
                before = control_df['artist_id'].nunique()
                control_df = control_df[
                    ~control_df['artist_id'].isin(excl_ids)
                ]
                after = control_df['artist_id'].nunique()
                print(f'  Lower-tier exclusion: removed {before - after:,} '
                      f'{venue_key} artists from control')
            else:
                print(f'  WARNING: No ID list found for {venue_key}, '
                      f'control not filtered')

        if args.test > 0:
            test_ids = (treated_df.drop_duplicates('artist_id')['artist_id']
                        .head(args.test).tolist())
            treated_df = treated_df[treated_df['artist_id'].isin(test_ids)]
            print(f'  TEST MODE: {args.test} treated artists')

        merged = run_matching(treated_df, control_df, venue_key,
                              params=params, n_jobs=args.n_jobs)

        if merged.empty:
            print(f'  Nothing to save for {venue_key}.')
            continue

        out_path = output_dir / f'matched_{venue_key}.csv'
        merged.to_csv(out_path, index=False)
        print(f'\n  Saved: {out_path}')
        print(f'    {merged["artist_id"].nunique():,} artists, '
              f'{len(merged):,} rows')

        pq_path = output_dir / f'matched_{venue_key}.parquet'
        merged.to_parquet(pq_path, index=False)
        print(f'  Saved: {pq_path}')

    print('\n' + '=' * 70)
    print('ALL DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()