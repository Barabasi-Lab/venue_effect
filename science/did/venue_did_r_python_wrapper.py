#!/usr/bin/env python3
"""
Python wrapper to call R's `did` package for Callaway & Sant'Anna estimation.

R's implementation is 5-10x faster than the Python csdid port.

Setup (one time):
    # Install R packages
    Rscript -e "install.packages(c('did', 'optparse', 'data.table'), repos='https://cran.r-project.org')"

Usage:
    python run_did_wrapper.py \
        --input ../../data/matches/enriched_l2/merged_physics_Nature_enriched_l2.csv \
        --outcomes cum_citations_na cum_publication_count_adj \
        --heterogeneity gender \
        --n_boot 99
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_r_did(
    input_path,
    outcome,
    output_dir,
    label="",
    n_boot=99,
    est_method="reg",
    min_venue_year=1930,
    min_cohort_size=10,
    covariates="none",
    subgroup_col="",
    subgroup_val="",
    r_script="run_did.R",
):
    """Call the R script for one outcome + optional subgroup."""

    cmd = [
        "Rscript", r_script,
        "--input", str(input_path),
        "--outcome", outcome,
        "--output_dir", str(output_dir),
        "--label", label if label else outcome,
        "--n_boot", str(n_boot),
        "--est_method", est_method,
        "--min_venue_year", str(min_venue_year),
        "--min_cohort_size", str(min_cohort_size),
        "--covariates", covariates,
    ]

    if subgroup_col and subgroup_val:
        cmd += ["--subgroup_col", subgroup_col, "--subgroup_val", subgroup_val]

    print(f"\n{'─'*60}")
    print(f"  Running: {outcome}" + (f" [{subgroup_col}={subgroup_val}]" if subgroup_col else ""))
    print(f"{'─'*60}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"  WARNING: R script exited with code {result.returncode}")
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description="Python wrapper for R did package")

    ap.add_argument("--input", type=str, required=True, help="Enriched CSV file")
    ap.add_argument("--outcomes", nargs="+", default=["cum_citations_na"],
                    help="Outcome variables")
    ap.add_argument("--output_dir", type=str, default="../../data/did_r_results",
                    help="Output directory")
    ap.add_argument("--heterogeneity", nargs="*", default=[],
                    choices=["gender", "career_stage", "region"],
                    help="Heterogeneity analyses")
    ap.add_argument("--n_boot", type=int, default=99, help="Bootstrap iterations")
    ap.add_argument("--est_method", default="reg", choices=["reg", "ipw", "dr"])
    ap.add_argument("--min_venue_year", type=int, default=1930)
    ap.add_argument("--min_cohort_size", type=int, default=10)
    ap.add_argument("--covariates", type=str, default="none",
                    help="Comma-separated covariates or 'none'")
    ap.add_argument("--r_script", type=str, default="venue_did_r_dynamic.R",
                    help="Path to the R script")

    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define subgroup mappings
    HETEROGENEITY_MAP = {
        "gender": ("gender_label", ["female", "male"]),
        "career_stage": ("career_stage", ["early-career", "mid-career", "late-career"]),
        "region": ("geo_group", [
            "Europe", "Asia", "Africa", "Oceania",
            "Northern America", "Latin America and the Caribbean",
        ]),
    }

    input_stem = Path(args.input).stem.replace("_enriched", "").replace("_l2", "")

    for outcome in args.outcomes:
        # --- General model (no subgroup) ---
        run_r_did(
            input_path=args.input,
            outcome=outcome,
            output_dir=output_dir / input_stem,
            label=f"{outcome}_general",
            n_boot=args.n_boot,
            est_method=args.est_method,
            min_venue_year=args.min_venue_year,
            min_cohort_size=args.min_cohort_size,
            covariates=args.covariates,
            r_script=args.r_script,
        )

        # --- Heterogeneity subgroups ---
        for het_type in args.heterogeneity:
            if het_type not in HETEROGENEITY_MAP:
                print(f"  WARNING: Unknown heterogeneity type: {het_type}")
                continue

            col, values = HETEROGENEITY_MAP[het_type]

            for val in values:
                safe_val = val.replace(" ", "_").replace("/", "_")
                run_r_did(
                    input_path=args.input,
                    outcome=outcome,
                    output_dir=output_dir / input_stem,
                    label=f"{outcome}_{het_type}_{safe_val}",
                    n_boot=args.n_boot,
                    est_method=args.est_method,
                    min_venue_year=args.min_venue_year,
                    min_cohort_size=args.min_cohort_size,
                    covariates=args.covariates,
                    subgroup_col=col,
                    subgroup_val=val,
                    r_script=args.r_script,
                )

    print(f"\n{'='*60}")
    print(f"  All results saved to: {output_dir / input_stem}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
