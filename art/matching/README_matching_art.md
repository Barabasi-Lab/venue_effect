# Art Biennale Matching Pipeline

## Overview

This script matches treated biennale artists with comparable control artists
who never appeared in top-tier biennials. It mirrors the science matching
pipeline (`matching.py`) but is adapted for exhibition data.

## Directory Structure

```
data/
├── matching_needed/                          # INPUT: from preprocess_biennale.py
│   ├── treated_df_venice_biennale.parquet
│   ├── treated_df_documenta.parquet
│   ├── treated_df_bienal_sao_paulo.parquet
│   ├── ...
│   ├── control_df_biennale.parquet           # shared control pool
│   └── venue_artist_id_lists.pkl             # full ID lists per venue
├── art/
│   └── matches/                              # OUTPUT: matched panels
│       ├── matched_venice_biennale.csv
│       ├── matched_venice_biennale.parquet
│       ├── matched_documenta.csv
│       └── ...
```

## Prerequisites

```bash
pip install fastdtw pandas numpy scipy tqdm pyarrow
```

## Usage

### Single venue (recommended to start)

```bash
# Test with 50 artists first
python matching_art.py --venue venice_biennale --test 50

# Full run, single-threaded
python matching_art.py --venue venice_biennale

# Full run, parallel (16 cores)
python matching_art.py --venue venice_biennale --n_jobs 16

# Documenta
python matching_art.py --venue documenta --n_jobs 16

# Lower-tier venue
python matching_art.py --venue bienal_sao_paulo --n_jobs 16
```

### All venues

```bash
python matching_art.py --venue all --n_jobs 16
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--venue` | required | Venue key or `all` |
| `--n_jobs` | 1 | Parallel workers |
| `--num_matches` | 3 | Controls per treated (k) |
| `--test` | 0 | Test mode: match only N treated artists |
| `--birth_tol` | 5 | Birth year tolerance (±years) |
| `--cem_tol` | 0.30 | CEM tolerance (±30%) on S, G at year_diff=-1 |
| `--cem_floor` | 2 | Minimum absolute margin for CEM |
| `--caliper` | None | Maximum DTW distance (None = no limit) |
| `--data_dir` | `../../data` | Root data directory |
| `--output_dir` | `../../data/art/matches` | Where to save matched panels |

## Matching Logic

### Step 1: Attribute Blocking

For each treated artist, candidates must pass all of:

1. **Gender**: exact match on `gender_est` (male/female/mostly_male/mostly_female)
2. **Continent overlap**: at least one shared continent flag = 1
   - An artist with `Europe=1, Asia=1` matches controls with `Europe=1` or `Asia=1`
   - Multi-nationality artists are handled correctly
3. **Birth year**: within ±5 years

### Step 2: CEM Pre-filter (year_diff = -1)

At the last pre-treatment year, candidates must have cumulative S and G
counts within ±30% (minimum margin of 2) of the treated artist.

This prevents matching a prolific artist (50 solo shows) with an emerging
one (2 solo shows) even if their temporal trajectories happen to be parallel.

### Step 3: DTW Distance

On the pre-treatment window (year_diff ∈ [-5, -1]):
- Compute DTW distance on standardized [S, G, F, B] trajectories
- Select the k nearest controls
- Standardization uses global mean/std from the entire control pool

### Step 4: Quality Enforcement

- Only keep treated artists who found **exactly k** controls
- Artists with fewer matches are dropped entirely

## Control Pool Logic

| Venue tier | Control pool |
|------------|-------------|
| **Top** (Venice, Documenta) | `control_df_biennale.parquet` as-is. These exclude Venice + Documenta artists (done in preprocessing). |
| **Lower** (São Paulo, Sydney, ...) | `control_df_biennale.parquet` MINUS that venue's artists. Uses the full, unfiltered ID list from `venue_artist_id_lists.pkl` to ensure short-career artists are also excluded. |

This means:
- A São Paulo artist who never went to Venice/Documenta **is in the control pool**
  for Venice/Documenta (they got a different, weaker prestige signal)
- But that same São Paulo artist is **excluded from the control** for São Paulo

## Output Format

Each `matched_{venue}.csv` / `.parquet` contains a panel:

| Column | Description |
|--------|-------------|
| `artist_id` | Unique artist identifier |
| `end_year` | Calendar year |
| `year_diff` | `end_year - biennale_year` (-5 to +10) |
| `biennale_year` | Treatment year (first biennale appearance) |
| `biennale_treated` | 1 if `end_year >= biennale_year` |
| `is_venue` | 1 for treated post-treatment, 0 otherwise |
| `matched_to` | Which treated artist this control is matched to |
| `match_distance` | DTW distance (0 for treated = self) |
| `S, G, F, B` | Cumulative exhibition counts |
| `B_adj` | B minus 1 for treated post-treatment (focal adjustment) |
| `gender_est` | Estimated gender |
| `career_stage` | early / mid / late |
| `Europe, North America, ...` | Continent flags (0/1) |

## Comparison with Science Pipeline

| Aspect | Science (`matching.py`) | Art (`matching_art.py`) |
|--------|------------------------|------------------------|
| ID column | `author_id` | `artist_id` |
| Time column | `year` / `to_year` | `end_year` / `year_diff` |
| Treatment event | `first_publish_year` | `biennale_year` |
| Outcome vars | citations, pubs, grants | S, G, F, B exhibitions |
| DTW columns | citations, pubs, funding | S, G, F, B |
| CEM columns | pubs, citations, funding | S, G |
| Region blocking | Country → region overlap | Continent flag overlap |
| Gender source | Inferred (0/1) | gender_guesser (string) |
| Window | [-5, +10] | [-5, +10] |
| Focal adjustment | pub_count - 1 | B - 1 |
