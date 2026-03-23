#!/usr/bin/env python3
"""
==============================================================================
Enrich matched/enriched panels with author modal L2 field
==============================================================================

Adds author-level modal L2 field columns from BigQuery table:
  ccnr-success.venue_effect.author_modal_l2_field

New columns added:
  - l2_field_modal_code
  - l2_field_modal_name
  - l2_field_modal_pub_count
  - total_l2_counted_pubs
  - l2_field_modal_fraction

Usage examples:

  python enrich_l2_field.py \
      --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv

  python enrich_l2_field.py \
      --input_dir ../../data/matches/enriched_citations

Output default:
  sibling folder named enriched_l2
"""

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT = "ccnr-success"
DATASET = "venue_effect"
DS = f"{PROJECT}.{DATASET}"


def get_bq_client(project: str) -> bigquery.Client:
    return bigquery.Client(project=project)


def query_author_modal_l2_field(client: bigquery.Client) -> pd.DataFrame:
    query = f"""
    SELECT
      author_id,
      l2_field_modal_code,
      l2_field_modal_name,
      l2_field_modal_pub_count,
      total_l2_counted_pubs,
      l2_field_modal_fraction
    FROM `{DS}.author_modal_l2_field`
    """
    print("Loading author modal L2 field table from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} author rows")
    return df


def enrich_one_file(
    input_path: Path,
    output_path: Path,
    author_l2: pd.DataFrame,
    sep: str = ";"
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Enriching: {input_path}")
    print(f"{'='*60}")

    df = pd.read_csv(input_path, sep=sep)
    print(f"  Loaded {len(df):,} rows, {df['author_id'].nunique():,} authors")

    before_cols = set(df.columns)

    df = df.merge(author_l2, on="author_id", how="left")

    new_cols = [c for c in df.columns if c not in before_cols]

    n_missing = df["l2_field_modal_code"].isna().sum() if "l2_field_modal_code" in df.columns else len(df)
    print(f"  Rows missing L2 match: {n_missing:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=sep)

    print(f"  Saved: {output_path}")
    print(f"  New columns: {new_cols}")

    return df


def infer_output_dir(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return input_dir.parent / "enriched_l2" if input_dir.name == "enriched_citations" else input_dir / "enriched_l2"


def main():
    ap = argparse.ArgumentParser(
        description="Enrich matched/enriched panels with author modal L2 field"
    )
    ap.add_argument("--input", type=str, default=None, help="Single input CSV")
    ap.add_argument("--output", type=str, default=None, help="Single output CSV")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory of CSVs")
    ap.add_argument("--output_dir", type=str, default=None, help="Output directory")
    ap.add_argument("--project", type=str, default="ccnr-success")
    ap.add_argument("--suffix", type=str, default="_l2")
    ap.add_argument("--sep", type=str, default=";")

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error("Provide either --input or --input_dir")

    client = get_bq_client(args.project)
    author_l2 = query_author_modal_l2_field(client)

    if args.input:
        input_path = Path(args.input)

        if args.output:
            output_path = Path(args.output)
        else:
            if args.output_dir:
                out_dir = Path(args.output_dir)
            else:
                # Save to sibling folder named enriched_l2
                out_dir = input_path.parent.parent / "enriched_l2" if input_path.parent.name == "enriched_citations" else input_path.parent / "enriched_l2"
            output_path = out_dir / f"{input_path.stem}{args.suffix}.csv"

        enrich_one_file(input_path, output_path, author_l2, sep=args.sep)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        out_dir = infer_output_dir(input_dir, args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.csv"))
        print(f"\nFound {len(files)} CSV files in {input_dir}")
        print(f"Output directory: {out_dir}")

        for fpath in files:
            output_path = out_dir / f"{fpath.stem}{args.suffix}.csv"
            enrich_one_file(fpath, output_path, author_l2, sep=args.sep)


if __name__ == "__main__":
    main()