#!/usr/bin/env python3
"""
==============================================================================
Venue Effect — Heterogeneous DiD Estimation via csdid (Dynamic / Calendar-Year)
==============================================================================

This version uses REAL CALENDAR YEAR in csdid:

    tname = year
    gname = venue_year for treated, 0 for never-treated controls

That means csdid directly estimates:
  1. Dynamic/event-study effects relative to actual treatment year
  2. Cohort/group effects by actual treatment cohort (venue_year)
  3. Heterogeneity by subgroup (gender, career_stage, region)

Compared with the relative-time version, this script preserves real cohort
timing, which is what you want when estimating both:
  - dynamic effects
  - effect by cohort

Supports two control group modes via --control_group flag:
  - "nevertreated" (default): only never-treated controls (first_treat=0)
  - "notyettreated": later-cohort treated authors also serve as controls

Outputs are saved as CSV with columns such as:
  - att, se, ci_lower, ci_upper, pvalue
  - dynamic: event_time, to_year
  - cohort: group (= venue_year cohort)
  - subgroup: subgroup label

Data expectations:
  - Semicolon-delimited CSV
  - Key columns: author_id, year, to_year, venue_year, is_venue,
    Gender (0=M, 1=F), first_year_of_publication,
    venue_region_code (pipe-separated),
    citations_na, cum_citations_na, cum_publication_count,
    cum_funding_count, cum_corresponding_count
  - Treated authors: venue_year > 0 and/or is_venue == 1 at some point
  - Controls: never treated, first_treat = 0

Usage:
    # Single file (never-treated, default)
    python venue_did_csdid_dynamic.py \
        --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
        --outcomes cum_citations_na cum_publication_count cum_funding_count \
        --heterogeneity gender career_stage region

    # Not-yet-treated control group
    python venue_did_csdid_dynamic.py \
        --input ../../data/matches/enriched_l2/merged_physics_Nature_enriched_l2.csv \
        --outcomes cum_citations_na cum_publication_count cum_funding_count \
        --control_group notyettreated \
        --output_dir ../../data/did_notyettreated

    # Directory of files
    python venue_did_csdid_dynamic.py \
        --input_dir ../../data/matches/enriched_citations \
        --outcomes cum_citations_na cum_publication_count

    # Unconditional
    python venue_did_csdid_dynamic.py \
        --input ... \
        --outcomes cum_citations_na \
        --covariates none

Requirements:
    pip install csdid pandas numpy scipy
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
    raise ImportError(
        "Install csdid: pip install csdid\n"
        "See: https://d2cml-ai.github.io/csdid/"
    )

warnings.filterwarnings("ignore")


# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTCOMES = [
    "cum_citations_na",
    "cum_publication_count",
    "cum_funding_count",
]

# References: female, early-career, Europe
COVARIATE_COLS = [
    "gender_male",
    "career_mid",
    "career_late",
    "geo_Asia",
    "geo_Africa",
    "geo_Oceania",
    "geo_NAmerica",
    "geo_LatAm",
    "venue_decade",
]

COVARIATES_TO_DROP = {
    "gender": ["gender_male"],
    "career_stage": ["career_mid", "career_late"],
    "region": [
        "geo_Asia",
        "geo_Africa",
        "geo_Oceania",
        "geo_NAmerica",
        "geo_LatAm",
    ],
}

HEP_BROAD_FIELDS = [
    "Particle and High Energy Physics",
    "Synchrotrons and Accelerators",
    "Nuclear and Plasma Physics",
]

GEO_GROUPS = [
    "Europe",
    "Asia",
    "Africa",
    "Oceania",
    "Northern America",
    "Latin America and the Caribbean",
]

NUMERIC_ZERO_FILL_COLS = [
    "citations_na",
    "cum_citations_na",
    "citations_old",
    "cum_citations_old",
    "normalized_citations_na",
    "cum_normalized_citations_na",
    "normalized_citations_old",
    "cum_normalized_citations_old",
    "cum_publication_count",
    "cum_funding_count",
    "cum_corresponding_count",
    "publication_count",
    "funding_count",
    "corresponding_count",
    "publication_count_adj",
    "cum_publication_count_adj",
    "venue_decade",
]

CAREER_STAGE_DEFS = {
    "early-career": (0, 10),
    "mid-career": (11, 25),
    "late-career": (26, 999),
}


# =============================================================================
# Helpers
# =============================================================================

def safe_print_exception(prefix, e):
    print(f"{prefix}: {e}")


def sanitize_filename_piece(x):
    return (
        str(x)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


# =============================================================================
# Data preparation
# =============================================================================

def load_and_prepare(input_path, min_venue_year=1930, min_treated_per_cohort=1):
    """
    Load enriched matched panel CSV and prepare for csdid.

    KEY DESIGN:
      - panel_time = actual calendar year
      - first_treat = actual venue_year for treated authors
      - first_treat = 0 for never-treated controls

    This lets csdid directly estimate:
      - aggte("dynamic")  -> event-time effects relative to treatment year
      - aggte("group")    -> cohort effects by real treatment year

    Args:
        input_path: path to enriched CSV
        min_venue_year: drop venue cohorts before this year (default 1930)
        min_treated_per_cohort: minimum treated authors per venue year (default 1)

    Returns:
        df: prepared DataFrame
        meta: dict with file metadata
    """
    print(f"\n  Loading: {input_path}")
    df = pd.read_csv(input_path, sep=";")
    print(f'    {len(df):,} rows, {df["author_id"].nunique():,} authors')

    # --- Optional HEP exclusion ---
    if "l2_field_modal_name" in df.columns:
        n_rows_before = len(df)
        n_authors_before = df["author_id"].nunique()

        df = df[
            df["l2_field_modal_name"].isna()
            | (~df["l2_field_modal_name"].isin(HEP_BROAD_FIELDS))
        ].copy()

        n_rows_after = len(df)
        n_authors_after = df["author_id"].nunique()

        print(
            f"    Excluded HEP-like modal L2 fields: "
            f"{n_authors_before - n_authors_after:,} authors, "
            f"{n_rows_before - n_rows_after:,} rows removed"
        )
    else:
        print("    l2_field_modal_name not found; keeping all authors")

    required = ["author_id", "year", "to_year", "venue_year", "is_venue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Numeric author ID ---
    author_ids = df["author_id"].unique()
    id_map = {aid: i + 1 for i, aid in enumerate(author_ids)}
    df["author_int_id"] = df["author_id"].map(id_map)

    # --- Treated indicator ---
    treated_ids = set(df.loc[df["is_venue"] == 1, "author_id"].unique())
    df["is_treated"] = df["author_id"].isin(treated_ids).astype(int)

    # --- Use real calendar year for panel time ---
    df["panel_time"] = pd.to_numeric(df["year"], errors="coerce")

    # --- Use actual venue_year as first treatment year ---
    df["first_treat"] = 0
    df.loc[df["is_treated"] == 1, "first_treat"] = pd.to_numeric(
        df.loc[df["is_treated"] == 1, "venue_year"],
        errors="coerce",
    )

    # Drop rows with invalid calendar year
    n_before = len(df)
    df = df[df["panel_time"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"    Dropped {n_dropped:,} rows with missing year")

    df["panel_time"] = df["panel_time"].astype(int)
    df["first_treat"] = df["first_treat"].fillna(0).astype(int)

    # Optional: if a treated author somehow has first_treat <= 0, make them control
    bad_treated = (df["is_treated"] == 1) & (df["first_treat"] <= 0)
    if bad_treated.any():
        bad_authors = df.loc[bad_treated, "author_id"].nunique()
        print(
            f"    WARNING: {bad_authors:,} treated authors have invalid venue_year; "
            f"setting them to never-treated"
        )
        bad_ids = set(df.loc[bad_treated, "author_id"].unique())
        df.loc[df["author_id"].isin(bad_ids), "is_treated"] = 0
        df.loc[df["author_id"].isin(bad_ids), "first_treat"] = 0

    # =====================================================================
    # NEW: Filter venue year range and minimum cohort size
    # =====================================================================

    # --- Filter 1: Drop venue cohorts before min_venue_year ---
    n_before_vy = df["author_id"].nunique()
    old_cohort_ids = set(
        df.loc[
            (df["first_treat"] > 0) & (df["first_treat"] < min_venue_year),
            "author_id",
        ].unique()
    )
    if old_cohort_ids:
        df = df[~df["author_id"].isin(old_cohort_ids)].copy()
        print(
            f"    Dropped {len(old_cohort_ids):,} treated authors with "
            f"venue_year < {min_venue_year}"
        )

    # --- Filter 2: Drop venue cohorts with too few treated units ---
    cohort_sizes = (
        df[df["first_treat"] > 0]
        .groupby("first_treat")["author_id"]
        .nunique()
    )
    small_cohorts = set(cohort_sizes[cohort_sizes < min_treated_per_cohort].index)
    valid_cohorts = set(cohort_sizes[cohort_sizes >= min_treated_per_cohort].index)

    if small_cohorts:
        small_cohort_ids = set(
            df.loc[df["first_treat"].isin(small_cohorts), "author_id"].unique()
        )
        df = df[~df["author_id"].isin(small_cohort_ids)].copy()
        print(
            f"    Dropped {len(small_cohorts)} cohorts with < {min_treated_per_cohort} "
            f"treated authors ({len(small_cohort_ids):,} authors removed): "
            f"{sorted(small_cohorts)[:10]}{'...' if len(small_cohorts) > 10 else ''}"
        )

    # --- Filter 3: Trim calendar years to reasonable range ---
    if len(valid_cohorts) > 0:
        earliest_venue = min(valid_cohorts)
        latest_venue = max(valid_cohorts)
        trim_min = earliest_venue - 6  # 5 pre-treatment + 1 buffer
        trim_max = latest_venue + 11   # 10 post-treatment + 1 buffer

        n_before_trim = len(df)
        df = df[
            (df["panel_time"] >= trim_min) & (df["panel_time"] <= trim_max)
        ].copy()
        n_trimmed = n_before_trim - len(df)
        if n_trimmed > 0:
            print(
                f"    Trimmed calendar years to [{trim_min}, {trim_max}]: "
                f"{n_trimmed:,} rows removed"
            )

    n_after_vy = df["author_id"].nunique()
    print(
        f"    After all filters: {n_after_vy:,} authors "
        f"({n_before_vy - n_after_vy:,} removed), "
        f"{len(valid_cohorts)} valid cohorts"
    )
    if valid_cohorts:
        vc_sorted = sorted(valid_cohorts)
        print(
            f"    Valid cohort range: {vc_sorted[0]}-{vc_sorted[-1]} "
            f"({len(vc_sorted)} cohorts)"
        )
        # Show cohort sizes
        for vy in vc_sorted[:5]:
            print(f"      {vy}: {cohort_sizes[vy]} treated authors")
        if len(vc_sorted) > 10:
            print(f"      ...")
        for vy in vc_sorted[-5:]:
            print(f"      {vy}: {cohort_sizes[vy]} treated authors")

    # =====================================================================
    # END NEW FILTERS
    # =====================================================================

    # --- Fill numeric count vars with 0 ---
    for col in NUMERIC_ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Career stage ---
    if "first_year_of_publication" in df.columns:
        df["venue_career_age"] = (
            pd.to_numeric(df["venue_year"], errors="coerce")
            - pd.to_numeric(df["first_year_of_publication"], errors="coerce")
        )

        df["career_stage"] = np.nan
        df.loc[df["venue_career_age"] <= 10, "career_stage"] = "early-career"
        df.loc[
            (df["venue_career_age"] > 10) & (df["venue_career_age"] <= 25),
            "career_stage",
        ] = "mid-career"
        df.loc[df["venue_career_age"] > 25, "career_stage"] = "late-career"

        df.loc[
            df["venue_career_age"].isna() | (df["venue_career_age"] < 0),
            "career_stage",
        ] = np.nan
    else:
        df["venue_career_age"] = np.nan
        df["career_stage"] = np.nan

    # --- Venue decade ---
    df["venue_decade"] = pd.to_numeric(df["venue_year"], errors="coerce")
    df["venue_decade"] = (df["venue_decade"] // 10 * 10).fillna(0).astype(int)

    # --- Region ---
    if "venue_region_code" in df.columns:
        df["region"] = df["venue_region_code"].apply(
            lambda x: str(x).split("|")[0].strip("[] '\"")
            if pd.notna(x) and str(x) not in ("nan", "None", "")
            else np.nan
        )
    else:
        df["region"] = np.nan

    # --- Subregion ---
    if "venue_subregion_code" in df.columns:
        df["subregion"] = df["venue_subregion_code"].apply(
            lambda x: str(x).split("|")[0].strip("[] '\"")
            if pd.notna(x) and str(x) not in ("nan", "None", "")
            else np.nan
        )
    else:
        df["subregion"] = np.nan

    # --- Unified geo group ---
    def _geo(row):
        sub = row.get("subregion")
        reg = row.get("region")
        if pd.notna(sub) and sub in (
            "Northern America",
            "Latin America and the Caribbean",
        ):
            return sub
        if pd.notna(reg) and reg in ("Europe", "Asia", "Africa", "Oceania"):
            return reg
        return np.nan

    df["geo_group"] = df.apply(_geo, axis=1)

    # --- Gender ---
    if "Gender" in df.columns:
        df["gender_label"] = df["Gender"].map({0: "male", 1: "female"})
    else:
        df["gender_label"] = np.nan

    # --- K-1 dummies ---
    df["gender_male"] = (df["gender_label"] == "male").astype(int)

    df["career_mid"] = (df["career_stage"] == "mid-career").astype(int)
    df["career_late"] = (df["career_stage"] == "late-career").astype(int)

    df["geo_Asia"] = (df["geo_group"] == "Asia").astype(int)
    df["geo_Africa"] = (df["geo_group"] == "Africa").astype(int)
    df["geo_Oceania"] = (df["geo_group"] == "Oceania").astype(int)
    df["geo_NAmerica"] = (df["geo_group"] == "Northern America").astype(int)
    df["geo_LatAm"] = (
        df["geo_group"] == "Latin America and the Caribbean"
    ).astype(int)

    # --- Summary ---
    n_treated = df[df["is_treated"] == 1]["author_id"].nunique()
    n_control = df[df["is_treated"] == 0]["author_id"].nunique()

    treated_first_treat = df.loc[df["first_treat"] > 0, "first_treat"]
    treated_vy = df.loc[df["is_treated"] == 1, "venue_year"]

    print(f"    Treated: {n_treated:,}, Control: {n_control:,}")
    print(f"    Calendar year range: {df['panel_time'].min()} - {df['panel_time'].max()}")

    if len(treated_first_treat) > 0:
        print(
            f"    first_treat range (treated cohorts): "
            f"{treated_first_treat.min()} - {treated_first_treat.max()}"
        )
    else:
        print("    first_treat range (treated cohorts): none")

    if len(treated_vy.dropna()) > 0:
        print(f"    Venue year range: {treated_vy.min()} - {treated_vy.max()}")
    else:
        print("    Venue year range: none")

    print(
        f"    to_year range (reference only): "
        f"{df['to_year'].min()} to {df['to_year'].max()}"
    )

    for col in ["gender_label", "career_stage", "geo_group"]:
        if col in df.columns:
            author_vals = df.groupby("author_id")[col].first()
            n_na = author_vals.isna().sum()
            n_tot = len(author_vals)
            print(
                f"    {col}: {n_tot - n_na:,} known, {n_na:,} NaN "
                f"({100 * n_na / max(n_tot, 1):.1f}%)"
            )

    present_covs = [c for c in COVARIATE_COLS if c in df.columns]
    missing_covs = [c for c in COVARIATE_COLS if c not in df.columns]
    print(f"    Covariates present: {present_covs}")
    if missing_covs:
        print(f"    Covariates MISSING: {missing_covs}")

    meta = {
        "n_treated": n_treated,
        "n_control": n_control,
        "input_file": str(input_path),
        "year_min": int(df["panel_time"].min()),
        "year_max": int(df["panel_time"].max()),
    }

    return df, meta


# =============================================================================
# Run csdid estimation
# =============================================================================

def build_xformla(outcome, covariates, available_cols):
    """
    Build covariate formula for csdid:
        outcome~covar1+covar2
    or
        outcome~1
    """
    if covariates is None:
        covariates = COVARIATE_COLS
    covs = [c for c in covariates if c in available_cols and c != outcome]
    if not covs:
        return f"{outcome}~1"
    return f"{outcome}~" + "+".join(covs)


def run_csdid(df, outcome, est_method="reg", covariates=None, n_boot=999, label="",
              control_group="nevertreated"):
    """
    Run csdid ATTgt estimation using:
      - tname = panel_time = calendar year
      - gname = first_treat = actual venue_year for treated, 0 for controls

    control_group: "nevertreated" or "notyettreated"
      - nevertreated: only first_treat=0 authors serve as controls
      - notyettreated: later-cohort treated authors also serve as controls
    """
    prefix = f"[{label}] " if label else ""
    print(f"\n  {prefix}Estimating ATT(g,t) for: {outcome}")
    print(f"    Method: {est_method}, Bootstrap: {n_boot}, Control group: {control_group}")

    if outcome not in df.columns:
        print(f'    WARNING: outcome "{outcome}" not in data. Skipping.')
        return None

    avail_covs = [c for c in (covariates or COVARIATE_COLS) if c in df.columns and c != outcome]
    xformla = build_xformla(outcome, covariates, set(df.columns))
    print(f"    Covariates formula: {xformla}")

    keep_cols = ["author_int_id", "panel_time", "first_treat", outcome] + avail_covs
    keep_cols = list(dict.fromkeys(keep_cols))
    work = df[keep_cols].copy()

    numeric_cols = [outcome] + avail_covs
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    work["panel_time"] = pd.to_numeric(work["panel_time"], errors="coerce").astype(int)
    work["first_treat"] = pd.to_numeric(work["first_treat"], errors="coerce").fillna(0).astype(int)
    work["author_int_id"] = pd.to_numeric(work["author_int_id"], errors="coerce").astype(int)

    n_units = work["author_int_id"].nunique()
    n_treated_units = work.loc[work["first_treat"] > 0, "author_int_id"].nunique()
    n_control_units = work.loc[work["first_treat"] == 0, "author_int_id"].nunique()

    print(
        f"    Working data: {len(work):,} rows, {n_units:,} units "
        f"(T={n_treated_units:,}, C={n_control_units:,})"
    )
    print(
        f"    Calendar time range: {work['panel_time'].min()}-{work['panel_time'].max()}, "
        f"first_treat values: {sorted(work['first_treat'].unique())[:10]}"
        f"{' ...' if len(sorted(work['first_treat'].unique())) > 10 else ''}"
    )

    def _fit(formula):
        try:
            att_gt = ATTgt(
                yname=outcome,
                gname="first_treat",
                idname="author_int_id",
                tname="panel_time",
                xformla=formula,
                data=work,
                control_group=control_group,
                panel=False,
                est_method=est_method,
                bstrap=True,
                biters=n_boot,
            )
            return att_gt.fit()
        except TypeError:
            try:
                att_gt = ATTgt(
                    yname=outcome,
                    gname="first_treat",
                    idname="author_int_id",
                    tname="panel_time",
                    xformla=formula,
                    data=work,
                    control_group=control_group,
                    panel=False,
                    est_method=est_method,
                )
                return att_gt.fit(bstrap=True, biters=n_boot)
            except TypeError:
                att_gt = ATTgt(
                    yname=outcome,
                    gname="first_treat",
                    idname="author_int_id",
                    tname="panel_time",
                    xformla=formula,
                    data=work,
                    control_group=control_group,
                    panel=False,
                )
                try:
                    return att_gt.fit(est_method=est_method, bstrap=True, biters=n_boot)
                except TypeError:
                    return att_gt.fit(est_method=est_method)

    t0 = time.time()
    try:
        result = _fit(xformla)
        elapsed = time.time() - t0
        print(f"    Fitted in {elapsed:.1f}s")

        print("\n    --- csdid ATT(g,t) summary ---")
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

        print("    --- end csdid output ---\n")
        return result

    except np.linalg.LinAlgError:
        print("    WARNING: Singular matrix with covariates, falling back to unconditional (~1)")
        try:
            result = _fit(f"{outcome}~1")
            elapsed = time.time() - t0
            print(f"    Fitted (unconditional) in {elapsed:.1f}s")
            return result
        except Exception as e2:
            safe_print_exception("    ERROR even unconditional", e2)
            return None

    except Exception as e:
        safe_print_exception("    ERROR", e)
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Extraction and aggregation
# =============================================================================

def extract_attgt(result):
    """
    Extract ATT(g,t) table from csdid result.

    Returns columns like:
      group, time, att, se, ci_lower, ci_upper, pvalue
    """
    from scipy.stats import norm

    try:
        summ = result.summ_attgt()
        df = summ.summary2.copy()

        col_map = {
            "Group": "group",
            "Time": "time",
            "ATT(g, t)": "att",
            "ATT(g,t)": "att",
            "Std. Error": "se",
            "Post": "post",
            "[95% Pointwise": "ci_lower",
            "Conf. Band]": "ci_upper",
            "[95.0% Pointwise": "ci_lower",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.loc[:, df.columns.str.strip() != ""]

        for col in ["att", "se", "ci_lower", "ci_upper", "group", "time"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "pvalue" not in df.columns and "att" in df.columns and "se" in df.columns:
            z = (df["att"] / df["se"]).replace([np.inf, -np.inf], 0)
            df["pvalue"] = 2 * norm.sf(np.abs(z))
        return df

    except Exception as e:
        print(f"    WARNING: Could not extract ATT(g,t) via summ_attgt: {e}")

    try:
        df = pd.DataFrame(
            {
                "group": result.group,
                "time": result.t,
                "att": result.att,
                "se": result.se,
            }
        )
        df["ci_lower"] = df["att"] - 1.96 * df["se"]
        df["ci_upper"] = df["att"] + 1.96 * df["se"]
        z = (df["att"] / df["se"]).replace([np.inf, -np.inf], 0)
        from scipy.stats import norm
        df["pvalue"] = 2 * norm.sf(np.abs(z))
        return df

    except Exception as e:
        print(f"    WARNING: Could not extract ATT(g,t): {e}")
        return pd.DataFrame()


def _capture_aggte(result, agg_type):
    """
    Capture printed output from result.aggte(agg_type).
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        agg = result.aggte(agg_type)
    finally:
        sys.stdout = old_stdout

    printed = buffer.getvalue()
    return printed, agg


def _parse_dynamic_output(printed_text):
    """
    Parse the 'Dynamic Effects:' table from csdid output.
    """
    from scipy.stats import norm

    lines = printed_text.strip().split("\n")
    rows = []
    in_dynamic = False

    for line in lines:
        line = line.strip()
        if "Dynamic Effects:" in line:
            in_dynamic = True
            continue
        if in_dynamic and line.startswith("---"):
            break
        if in_dynamic and line and (line[0].isdigit() or line[0] == "-"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # usually: row_index event_time att se ci_low ci_high ...
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
                    pvalue = 2 * norm.sf(abs(z))

                    rows.append(
                        {
                            "event_time": event_time,
                            "att": att,
                            "se": se,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "pvalue": pvalue,
                        }
                    )
                except (ValueError, IndexError):
                    continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["to_year"] = df["event_time"]
    return df.sort_values("event_time").reset_index(drop=True)


def _parse_group_output(printed_text):
    """
    Parse the 'Group Effects:' table from csdid output.
    """
    from scipy.stats import norm

    lines = printed_text.strip().split("\n")
    rows = []
    in_group = False

    for line in lines:
        line = line.strip()
        if "Group Effects:" in line:
            in_group = True
            continue
        if in_group and line.startswith("---"):
            break
        if in_group and line and (line[0].isdigit() or line[0] == "-"):
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
                    pvalue = 2 * norm.sf(abs(z))

                    rows.append(
                        {
                            "group": group,
                            "att": att,
                            "se": se,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "pvalue": pvalue,
                        }
                    )
                except (ValueError, IndexError):
                    continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def _parse_overall_output(printed_text):
    """
    Parse the overall ATT line from csdid simple aggregation output.
    """
    from scipy.stats import norm

    lines = printed_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "A", "S", "C", "E", "O")):
            continue

        parts = line.replace("*", "").split()
        if len(parts) >= 4:
            try:
                att = float(parts[0])
                se = float(parts[1])
                ci_lower = float(parts[2])
                ci_upper = float(parts[3])
                z = att / se if se > 0 else 0
                pvalue = 2 * norm.sf(abs(z))
                return pd.DataFrame(
                    [
                        {
                            "att": att,
                            "se": se,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "pvalue": pvalue,
                        }
                    ]
                )
            except (ValueError, IndexError):
                continue

    return pd.DataFrame()


"""
PATCH: Replace aggregate_dynamic() and aggregate_cohort() in venue_did_csdid_dynamic.py
with these versions that properly extract results from the aggte object.

The key fix: instead of parsing printed stdout (which is fragile and loses data),
we access the aggte return object's attributes directly.
"""


def _extract_aggte_results(result, agg_type):
    """
    Extract results from aggte() by accessing the return object directly,
    not by parsing printed output.

    The csdid aggte() returns an object with attributes like:
      - .egt or .group: the event times or group labels
      - .att or .att.egt: the ATT estimates
      - .se or .se.egt: the standard errors

    Different versions of csdid use different attribute names,
    so we try multiple possibilities.
    """
    from scipy.stats import norm
    import io, sys

    # Capture stdout but also get the return object
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        agg = result.aggte(agg_type)
    finally:
        sys.stdout = old_stdout

    printed = buffer.getvalue()

    # Try to extract from the object directly
    att_values = None
    se_values = None
    index_values = None

    # Try various attribute names used by different csdid versions
    for att_attr in ['att_egt', 'att.egt', 'att', 'overall_att']:
        val = getattr(agg, att_attr.replace('.', '_'), None)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            att_values = val
            break

    for se_attr in ['se_egt', 'se.egt', 'se', 'overall_se']:
        val = getattr(agg, se_attr.replace('.', '_'), None)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            se_values = val
            break

    for idx_attr in ['egt', 'group', 't']:
        val = getattr(agg, idx_attr, None)
        if val is not None and hasattr(val, '__len__') and len(val) > 0:
            index_values = val
            break

    # If direct extraction worked
    if att_values is not None and index_values is not None:
        import pandas as pd
        import numpy as np

        n = len(att_values)
        df = pd.DataFrame({
            'index': list(index_values)[:n],
            'att': list(att_values)[:n],
        })

        if se_values is not None and len(se_values) >= n:
            df['se'] = list(se_values)[:n]
        else:
            df['se'] = np.nan

        df['ci_lower'] = df['att'] - 1.96 * df['se']
        df['ci_upper'] = df['att'] + 1.96 * df['se']
        z = (df['att'] / df['se']).replace([np.inf, -np.inf], 0)
        df['pvalue'] = 2 * norm.sf(np.abs(z))

        return df, printed

    # If we also have a summary2 attribute (some versions)
    if hasattr(agg, 'summary2'):
        return agg.summary2.copy(), printed

    # Try to get from the agg object as a dataframe
    try:
        if hasattr(agg, 'to_frame'):
            return agg.to_frame(), printed
    except:
        pass

    return None, printed


def aggregate_dynamic(result):
    """
    Dynamic/event-study effects.
    event_time = calendar year - venue_year
    """
    from scipy.stats import norm
    import pandas as pd
    import numpy as np

    # Method 1: Extract from aggte object directly
    try:
        df, printed = _extract_aggte_results(result, "dynamic")
        if df is not None and not df.empty:
            df = df.rename(columns={'index': 'event_time'})
            df['to_year'] = df['event_time']
            return df.sort_values('event_time').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") object extraction failed: {e}')

    # Method 2: Parse printed output (fallback)
    try:
        from venue_did_csdid_dynamic import _capture_aggte, _parse_dynamic_output
        printed, _ = _capture_aggte(result, "dynamic")
        df = _parse_dynamic_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("dynamic") print parsing failed: {e}')

    # Method 3: Manual aggregation from ATT(g,t)
    try:
        from venue_did_csdid_dynamic import extract_attgt
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        attgt = attgt.copy()
        attgt["event_time"] = attgt["time"] - attgt["group"]

        # Only use rows with valid SE
        valid = attgt[attgt["se"].notna() & (attgt["se"] > 0)].copy()
        if valid.empty:
            # If no valid SEs, still compute ATT means without SE
            dynamic = attgt.groupby("event_time").agg(
                att=("att", "mean"),
                n_groups=("group", "nunique"),
            ).reset_index()
            dynamic["se"] = np.nan
        else:
            dynamic = valid.groupby("event_time").agg(
                att=("att", "mean"),
                se=("se", lambda x: np.sqrt(np.mean(np.asarray(x) ** 2))),
                n_groups=("group", "nunique"),
            ).reset_index()

        dynamic["ci_lower"] = dynamic["att"] - 1.96 * dynamic["se"]
        dynamic["ci_upper"] = dynamic["att"] + 1.96 * dynamic["se"]
        z = (dynamic["att"] / dynamic["se"]).replace([np.inf, -np.inf], 0)
        dynamic["pvalue"] = 2 * norm.sf(np.abs(z))
        dynamic["to_year"] = dynamic["event_time"]

        return dynamic.sort_values("event_time").reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: Manual dynamic aggregation failed: {e}')
        return pd.DataFrame()


def aggregate_cohort(result):
    """
    Cohort/group effects by actual treatment cohort.
    group = actual venue_year
    """
    from scipy.stats import norm
    import pandas as pd
    import numpy as np

    # Method 1: Extract from aggte object directly
    try:
        df, printed = _extract_aggte_results(result, "group")
        if df is not None and not df.empty:
            df = df.rename(columns={'index': 'group'})
            return df.sort_values('group').reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: aggte("group") object extraction failed: {e}')

    # Method 2: Parse printed output (fallback)
    try:
        from venue_did_csdid_dynamic import _capture_aggte, _parse_group_output
        printed, _ = _capture_aggte(result, "group")
        df = _parse_group_output(printed)
        if not df.empty:
            return df
    except Exception as e:
        print(f'    WARNING: aggte("group") print parsing failed: {e}')

    # Method 3: Manual aggregation from ATT(g,t)
    try:
        from venue_did_csdid_dynamic import extract_attgt
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        post = attgt[attgt["time"] >= attgt["group"]].copy()
        if post.empty:
            return pd.DataFrame()

        # Separate valid and invalid SE rows
        has_se = post["se"].notna() & (post["se"] > 0)

        cohort_list = []
        for g, group_df in post.groupby("group"):
            row = {"group": g, "att": group_df["att"].mean(), "n_periods": len(group_df)}

            valid_se = group_df.loc[has_se[group_df.index], "se"]
            if len(valid_se) > 0:
                row["se"] = np.sqrt(np.mean(np.asarray(valid_se) ** 2))
            else:
                row["se"] = np.nan

            cohort_list.append(row)

        cohort = pd.DataFrame(cohort_list)
        cohort["ci_lower"] = cohort["att"] - 1.96 * cohort["se"]
        cohort["ci_upper"] = cohort["att"] + 1.96 * cohort["se"]
        z = (cohort["att"] / cohort["se"]).replace([np.inf, -np.inf], 0)
        cohort["pvalue"] = 2 * norm.sf(np.abs(z))

        return cohort.sort_values("group").reset_index(drop=True)
    except Exception as e:
        print(f'    WARNING: Manual cohort aggregation failed: {e}')
        return pd.DataFrame()


def aggregate_overall(result):
    """
    Overall ATT from aggte("simple").
    """
    from scipy.stats import norm
    import pandas as pd
    import numpy as np

    # Method 1: Extract from aggte object
    try:
        df, printed = _extract_aggte_results(result, "simple")
        if df is not None and not df.empty:
            return df.head(1)
    except Exception as e:
        print(f'    WARNING: aggte("simple") extraction failed: {e}')

    # Method 2: Manual
    try:
        from venue_did_csdid_dynamic import extract_attgt
        attgt = extract_attgt(result)
        if attgt.empty:
            return pd.DataFrame()

        post = attgt[attgt["time"] >= attgt["group"]].copy()
        if post.empty:
            return pd.DataFrame()

        overall_att = post["att"].mean()

        valid_se = post.loc[post["se"].notna() & (post["se"] > 0), "se"]
        if len(valid_se) > 0:
            overall_se = np.sqrt(np.mean(np.asarray(valid_se) ** 2))
        else:
            overall_se = np.nan

        z = overall_att / overall_se if pd.notna(overall_se) and overall_se > 0 else 0

        return pd.DataFrame([{
            "att": overall_att,
            "se": overall_se,
            "ci_lower": overall_att - 1.96 * overall_se if pd.notna(overall_se) else np.nan,
            "ci_upper": overall_att + 1.96 * overall_se if pd.notna(overall_se) else np.nan,
            "pvalue": 2 * norm.sf(abs(z)) if z != 0 else np.nan,
            "n_groups": post["group"].nunique(),
            "n_periods": len(post),
        }])
    except Exception as e:
        print(f'    WARNING: Manual overall aggregation failed: {e}')
        return pd.DataFrame()


# =============================================================================
# Heterogeneity
# =============================================================================

def subset_by_attribute(df, col, value):
    """
    Subset to authors where author-level attribute col == value.
    """
    author_attr = df.groupby("author_id")[col].first()
    keep_authors = set(author_attr[author_attr == value].index)
    return df[df["author_id"].isin(keep_authors)].copy()


def run_heterogeneity(
    df,
    outcome,
    subgroup_col,
    subgroup_values,
    est_method="reg",
    covariates=None,
    n_boot=999,
    control_group="nevertreated",
):
    """
    Run csdid separately for each subgroup value.
    """
    results = {}
    for val in subgroup_values:
        sub = subset_by_attribute(df, subgroup_col, val)
        n_treated = sub[sub["is_treated"] == 1]["author_id"].nunique()
        n_control = sub[sub["is_treated"] == 0]["author_id"].nunique()

        print(
            f"\n  --- Subgroup: {subgroup_col}={val} "
            f"(T={n_treated:,}, C={n_control:,}) ---"
        )

        if n_treated == 0 or n_control == 0:
            print("    SKIPPING: no treated or no control units")
            continue

        res = run_csdid(
            sub,
            outcome,
            est_method=est_method,
            covariates=covariates,
            n_boot=n_boot,
            label=f"{subgroup_col}={val}",
            control_group=control_group,
        )
        if res is not None:
            results[val] = res

    return results


def run_geo_heterogeneity(df, outcome, est_method="reg", covariates=None, n_boot=999,
                          control_group="nevertreated"):
    """
    Run csdid for each geography group using geo_group.
    """
    results = {}
    for geo_label in GEO_GROUPS:
        sub = subset_by_attribute(df, "geo_group", geo_label)
        n_treated = sub[sub["is_treated"] == 1]["author_id"].nunique()
        n_control = sub[sub["is_treated"] == 0]["author_id"].nunique()

        print(f"\n  --- Geo: {geo_label} (T={n_treated:,}, C={n_control:,}) ---")

        if n_treated == 0 or n_control == 0:
            print("    SKIPPING: no treated or no control units")
            continue

        res = run_csdid(
            sub,
            outcome,
            est_method=est_method,
            covariates=covariates,
            n_boot=n_boot,
            label=f"geo={geo_label}",
            control_group=control_group,
        )
        if res is not None:
            results[geo_label] = res

    return results


# =============================================================================
# Save
# =============================================================================

def save_results(results_dict, output_dir, file_label):
    """
    Save all results to CSV files inside output_dir/file_label.
    """
    out_dir = Path(output_dir) / file_label
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, data in results_dict.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            fpath = out_dir / f"{key}.csv"
            data.to_csv(fpath, index=False)
            print(f"    Saved: {fpath.name}  ({len(data)} rows)")
        elif isinstance(data, dict):
            for subkey, subdata in data.items():
                if isinstance(subdata, pd.DataFrame) and not subdata.empty:
                    fpath = out_dir / f"{key}_{subkey}.csv"
                    subdata.to_csv(fpath, index=False)
                    print(f"    Saved: {fpath.name}  ({len(subdata)} rows)")

    print(f"\n  All results saved to: {out_dir}")


# =============================================================================
# Main pipeline
# =============================================================================

def process_one_file(
    input_path,
    output_dir,
    outcomes,
    heterogeneity_types,
    est_method="reg",
    covariates=None,
    n_boot=999,
    control_group="nevertreated",
):
    """
    Full pipeline for one enriched matched panel file.
    """
    input_path = Path(input_path)
    file_label = input_path.stem.replace("_enriched", "")

    print(f'\n{"=" * 70}')
    print(f"  VENUE DiD ESTIMATION (DYNAMIC / CALENDAR-YEAR / {control_group.upper()}): {file_label}")
    print(f'{"=" * 70}')

    df, meta = load_and_prepare(input_path)
    all_results = {}

    for outcome in outcomes:
        if outcome not in df.columns:
            print(f'\n  WARNING: "{outcome}" not in data, skipping.')
            continue

        outcome_tag = sanitize_filename_piece(outcome)

        print(f'\n{"─" * 50}')
        print(f"  OUTCOME: {outcome}")
        print(f'{"─" * 50}')

        # --- General model ---
        result = run_csdid(
            df,
            outcome,
            est_method=est_method,
            covariates=covariates,
            n_boot=n_boot,
            label="general",
            control_group=control_group,
        )

        if result is None:
            continue

        # Raw ATT(g,t) available if needed, but not saved by default
        _ = extract_attgt(result)

        # --- Dynamic ---
        dynamic = aggregate_dynamic(result)
        if not isinstance(dynamic, pd.DataFrame):
            dynamic = pd.DataFrame()

        if not dynamic.empty:
            all_results[f"dynamic_{outcome_tag}"] = dynamic
            print("\n    Dynamic ATT (event-study):")
            for _, r in dynamic.iterrows():
                e = r.get("event_time", "?")
                a = r.get("att", 0)
                se_val = r.get("se", 0)
                pv = r.get("pvalue", np.nan)
                try:
                    a = float(a)
                    se_val = float(se_val)
                    sig = "*" if se_val > 0 and abs(a) > 1.96 * se_val else ""
                    print(
                        f"      e={e:>3}: ATT={a:>10.2f} "
                        f"(SE={se_val:.2f}) p={pv:.4g}{sig}"
                    )
                except (ValueError, TypeError):
                    print(f"      e={e}: ATT={a} (SE={se_val})")

        # --- Cohort ---
        cohort = aggregate_cohort(result)
        if not isinstance(cohort, pd.DataFrame):
            cohort = pd.DataFrame()

        if not cohort.empty:
            all_results[f"cohort_{outcome_tag}"] = cohort
            print("\n    Cohort ATT (by venue year):")
            for _, r in cohort.iterrows():
                g = r.get("group", "?")
                a = r.get("att", 0)
                se_val = r.get("se", 0)
                pv = r.get("pvalue", np.nan)
                try:
                    a = float(a)
                    se_val = float(se_val)
                    sig = "*" if se_val > 0 and abs(a) > 1.96 * se_val else ""
                    print(
                        f"      cohort={g}: ATT={a:>10.2f} "
                        f"(SE={se_val:.2f}) p={pv:.4g}{sig}"
                    )
                except (ValueError, TypeError):
                    print(f"      cohort={g}: ATT={a} (SE={se_val})")

        # --- Overall ---
        overall = aggregate_overall(result)
        if not isinstance(overall, pd.DataFrame):
            overall = pd.DataFrame()

        if not overall.empty:
            all_results[f"overall_{outcome_tag}"] = overall
            try:
                att_val = float(overall.iloc[0].get("att", 0))
                pv_val = float(overall.iloc[0].get("pvalue", 0))
                print(f"\n    Overall ATT: {att_val:.2f} (p={pv_val:.4g})")
            except (ValueError, TypeError):
                print(f"\n    Overall ATT: {overall.iloc[0].to_dict()}")

        # --- Heterogeneity ---
        for het_type in heterogeneity_types:
            het_covs = covariates
            if het_covs is None:
                drop_cols = COVARIATES_TO_DROP.get(het_type, [])
                het_covs = [c for c in COVARIATE_COLS if c not in drop_cols]
                print(
                    f"\n  Covariates for {het_type} heterogeneity: "
                    f"{het_covs} (dropped {drop_cols})"
                )

            if het_type == "gender":
                print(f"\n  ── Gender heterogeneity: {outcome} ──")
                het_results = run_heterogeneity(
                    df,
                    outcome,
                    "gender_label",
                    ["female", "male"],
                    est_method=est_method,
                    covariates=het_covs,
                    n_boot=n_boot,
                    control_group=control_group,
                )
                het_prefix = "gender"

            elif het_type == "career_stage":
                print(f"\n  ── Career stage heterogeneity: {outcome} ──")
                het_results = run_heterogeneity(
                    df,
                    outcome,
                    "career_stage",
                    ["early-career", "mid-career", "late-career"],
                    est_method=est_method,
                    covariates=het_covs,
                    n_boot=n_boot,
                    control_group=control_group,
                )
                het_prefix = "career"

            elif het_type == "region":
                print(f"\n  ── Region heterogeneity: {outcome} ──")
                het_results = run_geo_heterogeneity(
                    df,
                    outcome,
                    est_method=est_method,
                    covariates=het_covs,
                    n_boot=n_boot,
                    control_group=control_group,
                )
                het_prefix = "region"

            else:
                print(f"    WARNING: unknown heterogeneity type: {het_type}")
                continue

            # Save subgroup dynamic / cohort / overall
            for sval, sres in het_results.items():
                safe_sval = sanitize_filename_piece(sval)

                dyn = aggregate_dynamic(sres)
                if isinstance(dyn, pd.DataFrame) and not dyn.empty:
                    dyn["subgroup"] = sval
                    all_results[f"dynamic_{outcome_tag}_{het_prefix}_{safe_sval}"] = dyn

                coh = aggregate_cohort(sres)
                if isinstance(coh, pd.DataFrame) and not coh.empty:
                    coh["subgroup"] = sval
                    all_results[f"cohort_{outcome_tag}_{het_prefix}_{safe_sval}"] = coh

                ovr = aggregate_overall(sres)
                if isinstance(ovr, pd.DataFrame) and not ovr.empty:
                    ovr["subgroup"] = sval
                    all_results[f"overall_{outcome_tag}_{het_prefix}_{safe_sval}"] = ovr

    print(f'\n{"─" * 50}')
    print("  SAVING RESULTS")
    print(f'{"─" * 50}')
    save_results(all_results, output_dir, file_label)

    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Venue Effect — Heterogeneous DiD via csdid (calendar-year dynamic version)"
    )

    ap.add_argument("--input", type=str, default=None, help="Single enriched CSV file")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory of enriched CSV files")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="../../data/did_dynamic",
        help="Output directory for results",
    )

    ap.add_argument(
        "--outcomes",
        nargs="+",
        default=DEFAULT_OUTCOMES,
        help="Outcome variables to estimate",
    )
    ap.add_argument(
        "--heterogeneity",
        nargs="*",
        default=[],
        choices=["gender", "career_stage", "region"],
        help="Heterogeneity analyses to run",
    )
    ap.add_argument(
        "--est_method",
        default="reg",
        choices=["reg", "ipw", "dr"],
        help="Estimation method (reg=RA, ipw=IPW, dr=doubly robust)",
    )
    ap.add_argument(
        "--n_boot",
        type=int,
        default=999,
        help="Bootstrap iterations",
    )
    ap.add_argument(
        "--covariates",
        nargs="*",
        default=None,
        help='Override covariate columns. Use "none" for unconditional.',
    )
    ap.add_argument(
        "--control_group",
        default="nevertreated",
        choices=["nevertreated", "notyettreated"],
        help="Control group type: nevertreated (default) or notyettreated",
    )

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error("Provide either --input or --input_dir")

    cov = args.covariates
    if cov and len(cov) == 1 and cov[0].lower() == "none":
        cov = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_one_file(
            args.input,
            output_dir,
            args.outcomes,
            args.heterogeneity,
            est_method=args.est_method,
            covariates=cov,
            n_boot=args.n_boot,
            control_group=args.control_group,
        )
    else:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*_enriched*.csv"))
        print(f"\nFound {len(files)} enriched files in {input_dir}")

        for fpath in files:
            process_one_file(
                fpath,
                output_dir,
                args.outcomes,
                args.heterogeneity,
                est_method=args.est_method,
                covariates=cov,
                n_boot=args.n_boot,
                control_group=args.control_group,
            )


if __name__ == "__main__":
    main()