# Venue Effect — Matching Methodology

## Overview

This module matches treated authors (those who published in a selected high-impact journal) to comparable control authors (same field, never published in any of those journals) based on pre-treatment career trajectories. The goal is to construct counterfactual pairs for a heterogeneous difference-in-differences analysis of the "venue effect."

The matching follows a **hybrid coarsened exact matching (CEM) + dynamic time warping (DTW)** approach, consistent with the methodology in Liu, Dorantes-Gilardi, Han & Barabási and with standard practices in Callaway & Sant'Anna (2021) and Tian et al. (2025).

## Files

| File | Purpose |
|---|---|
| `inspect_columns.py` | View columns, types, and sample values in treated/control parquet files. Run first. |
| `matching.py` | Main matching script. Processes one field × journal at a time. Supports test mode and parallel execution. |
| `run_matching.sh` | Shell wrapper: runs `matching.py` for every journal in a field. |

## Quick Start

```bash
# 1. Inspect your data
python inspect_columns.py --field physics

# 2. Test on a small sample (50 treated, sequential)
python matching.py --field physics --journal_id jour.1018957 --test 50

# 3. Full run for one journal (32 cores)
python matching.py --field physics --journal_id jour.1018957 --n_jobs 32

# 4. All journals for a field
bash run_matching.sh physics 32

# 5. Test all journals (50 authors each, sequential)
bash run_matching.sh physics 1 50
```

## Matching Algorithm

For each treated author, the algorithm finds `k` (default 3) control authors with the most similar pre-treatment career trajectories. The procedure has six steps:

### Step 1: Coarsened Exact Matching (Blocking)

Before computing any distances, we restrict the candidate pool using exact or banded matches on categorical/demographic variables:

| Variable | Match type | Rationale |
|---|---|---|
| **Career age at treatment** | ±2 year band | Controls should be at a similar career stage when the treated author first published in the top venue. Using a band (not exact year) ensures a large enough candidate pool. Follows Tian et al. (2025). |
| **Gender** | Exact | Gender disparities in citation and productivity are well-documented (Huang et al. 2020). Matching on gender ensures within-group comparisons. |
| **Affiliation country** | Set overlap | Authors may have multiple country affiliations (e.g., `['US', 'CN']`). A match is valid if the treated and control author share at least one country code. This is more permissive than exact string match but still controls for regional effects. |
| **Active at treatment year** | Required | The control author must have a publication record (a row in the panel) in the same calendar year as the treated author's first top-venue publication. This ensures the control was an active researcher at the time. |

### Step 2: Pre-Treatment Window

Both treated and control panels are restricted to a window around the treatment year:

- **Pre-treatment**: `to_year` in `[pre_window, -1]` (default `pre_window = -5`)
- **Post-treatment**: `to_year` in `[0, post_window]` (default `post_window = 10`)

Only authors whose panel spans the full window are retained. This ensures enough pre-treatment trajectory data for matching and enough post-treatment data for outcome measurement.

The `to_year` variable is defined as `year - first_publish_year`, where `first_publish_year` is the treated author's first top-venue publication year. For control authors, this pseudo-treatment year is inherited from the treated author they are being compared to.

### Step 3: Multivariate Dynamic Time Warping (DTW)

On the pre-treatment rows only (`to_year < 0`), we compute a multivariate DTW distance between the treated author's trajectory and each candidate control's trajectory over four cumulative career metrics:

1. `cum_publication_count` — cumulative papers published
2. `cum_corresponding_count` — cumulative corresponding authorships
3. `cum_citations` — cumulative citations received
4. `cum_funding_count` — cumulative grants/funding awards

Each metric forms one dimension of the multivariate time series. DTW aligns the two series to find the minimum-cost warping path, using Euclidean distance between the feature vectors at each time step.

**Standardization**: All four metrics are z-scored using the **global** mean and standard deviation computed from the entire control pool (not per-pair). This ensures:
- Features with different scales (e.g., citations vs. publication count) contribute equally.
- The standardization is consistent across all comparisons for a given treated author.

### Step 4: Nearest-Neighbor Selection

For each treated author, the `k` control candidates with the smallest DTW distance are selected as matches.

### Step 5: Caliper (Optional)

If a caliper is specified (`--caliper`), any candidate whose DTW distance exceeds the threshold is rejected before selecting the top-k. If no candidates remain within the caliper, the treated author is left unmatched. This prevents poor-quality matches from entering the analysis.

### Step 6: Covariate Balance Check

After all matching is complete, the script computes the **Absolute Standardized Difference (ASD)** between treated and matched control groups at `to_year = -1` (one year before treatment):

$$\text{ASD} = \frac{|\bar{X}_T - \bar{X}_C|}{\sqrt{(s_T^2 + s_C^2) / 2}}$$

Thresholds (following Rosenbaum & Rubin 1985):
- **ASD < 0.1**: Good balance ✓
- **ASD 0.1–0.25**: Acceptable ~
- **ASD > 0.25**: Poor balance ✗

The balance check covers all DTW matching variables plus annual metrics (`publication_count`, `citations_annual`, `funding_count`).

## Matching With Replacement

A control author can be matched to multiple treated authors. This is standard practice in observational studies (Rosenbaum 2020, Abadie & Imbens 2006) and is necessary when the treated population is large relative to the control pool. The `matched_to` column in the output tracks which treated author each control row was matched to.

## Parameters

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| `num_matches` | `--num_matches` | 3 | Controls per treated author |
| `pre_window` | `--pre_window` | -5 | Start of pre-treatment window (years) |
| `post_window` | — | 10 | End of post-treatment window (years) |
| `career_age_tolerance` | `--career_age_tol` | 2 | ±years for career age blocking |
| `caliper` | `--caliper` | None | Maximum DTW distance (None = no caliper) |
| `dtw_cols` | — | See code | Cumulative series for DTW distance |
| `country_col` | — | `first_affiliation_countries` | Country column for set-overlap matching |
| `n_jobs` | `--n_jobs` | 1 | Parallel workers (1 = sequential) |
| `test` | `--test` | 0 | Test mode: match only N treated authors |

## Input / Output

### Input

From the preprocessing step (`preprocess_venue_effect.ipynb`):

```
../../data/matching_needed/
  treated_df_{field}.parquet    # all treated authors, has journal_id for filtering
  control_df_{field}.parquet    # field authors who never published in selected journals
```

### Output

```
../../data/matches/
  treated_{field}_{journal}.parquet   # treated panel (windowed)
  matched_{field}_{journal}.parquet   # matched control panel with:
                                      #   match_distance — DTW distance
                                      #   matched_to — treated author_id
                                      #   first_publish_year — pseudo-treatment year
                                      #   to_year — relative year
```

## Design Rationale and Robustness

### Why career age band instead of exact first publication year?

Requiring the treated and control author to have started publishing in the exact same year drastically reduces the candidate pool and can make matching infeasible for authors with unusual career start dates. Matching on career age at treatment (±2 years) is standard in the science of science literature (Tian et al. 2025, Azoulay et al. 2014) and ensures the control was at a comparable career stage.

### Why DTW instead of Euclidean distance?

DTW captures similarity in trajectory shape even when the exact timing of career milestones differs slightly. However, for strict parallel-trends identification, Euclidean distance on aligned `to_year` vectors is also defensible. We recommend reporting DTW as the primary specification and Euclidean as a robustness check (adjustable in the code by replacing the `dtw_distance` function).

### Why global standardization?

The original implementation standardized each treated-control pair independently (pooled z-score of two sequences). This means the same control author could have different standardized values depending on which treated author they were being compared to, making distances non-comparable across treated authors. Global standardization (using the full control pool) ensures a consistent scale.

### Why set-overlap for country?

Authors in the Dimensions database often have multiple affiliation countries (a REPEATED field from BigQuery). The original code flattened this to a single country and required exact match. Set-overlap (any shared country) is more permissive but still controls for the geographic dimension of the venue effect, which the paper documents as substantial (Figure 6).

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
pyarrow >= 12.0
fastdtw >= 0.3.4
scipy >= 1.10
tqdm >= 4.60
```

## References

- Callaway, B. & Sant'Anna, P.H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200–230.
- Tian, C., Huang, Y., Jin, C., Ma, Y. & Uzzi, B. (2025). The distinctive innovation patterns and network embeddedness of scientific prizewinners. *PNAS*, 122(40).
- Huang, J., Gates, A.J., Sinatra, R. & Barabási, A.-L. (2020). Historical comparison of gender inequality in scientific careers across countries and disciplines. *PNAS*, 117(9), 4609–4616.
- Rosenbaum, P.R. (2020). Modern algorithms for matching in observational studies. *Annual Review of Statistics and Its Application*, 7, 143–176.
- Abadie, A. & Imbens, G.W. (2006). Large sample properties of matching estimators for average treatment effects. *Econometrica*, 74(1), 235–267.
- Rambachan, A. & Roth, J. (2023). A more credible approach to parallel trends. *Review of Economic Studies*, 90(5), 2555–2591.
