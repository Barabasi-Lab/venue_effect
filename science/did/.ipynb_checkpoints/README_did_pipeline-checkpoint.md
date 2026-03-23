# Venue Effect DiD Estimation — Python `csdid` Pipeline

## Overview

This pipeline replaces the Stata `xthdidregress` workflow with Python's `csdid`
package (Callaway & Sant'Anna 2021). It produces all the results needed for
Figures 2–6 of the paper.

## Mapping: Stata → Python

### Your Stata workflow:
```stata
xthdidregress ra (citations_na cum_publication_count cum_funding_count
    cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)

estat aggregation, dynamic graph    % → Figure 2 (effect upon exposure)
estat aggregation, cohort graph     % → Figure 3 (effect by treatment year)
```

### Equivalent Python:
```python
from csdid import ATTgt

result = ATTgt(
    yname="cum_citations_na",        # outcome
    gname="first_treat",              # venue_year for treated, 0 for never-treated
    idname="author_int_id",           # numeric panel ID
    tname="year",                     # calendar year
    xformla="~cum_publication_count+cum_funding_count+cum_corresponding_count",
    data=df,
    control_group='nevertreated',
).fit(est_method='reg')              # RA estimator (same as Stata's `ra`)

dynamic  = result.aggregate('event')  # → Figure 2
cohort   = result.aggregate('group')  # → Figure 3
overall  = result.aggregate('simple') # → single ATT number
```

## Key Design Decisions

### 1. Treatment variable (`gname`)
- **Treated**: `first_treat = venue_year` (the calendar year of first top-venue pub)
- **Never-treated control**: `first_treat = 0`
- This maps directly to Stata's `(is_journal), group(researcher_id)`

### 2. Identifying treated vs control
In the matched panel, both treated and control authors have `venue_year` set
(controls inherit their matched treated author's venue_year for alignment).
We distinguish them via: does the author ever have `is_venue == 1`?

### 3. Covariates
The `xformla` mirrors Stata's covariate specification:
- `cum_publication_count`, `cum_funding_count`, `cum_corresponding_count`
- Career stage is handled via subgroup analysis rather than as a covariate
  (since `csdid` doesn't natively support `i.career_cat` factor variables
  in the same way — you'd need to create dummies)

### 4. Heterogeneity = separate estimation on subsets
For gender/career_stage/region analyses, we **subset the data** and re-run
`csdid` on each subgroup. This mirrors your Stata approach:
```stata
keep if gender == 1                    % subset to male
xthdidregress ra (...) (is_journal)    % re-estimate on subset
```

### 5. Control group choice
We use `control_group='nevertreated'` (default), matching your main results.
For the not-yet-treated robustness check (SI 5.6), change to `'notyettreated'`.

## Output Structure

```
results/did/
  merged_physics_Nature/
    attgt_cum_citations_na.csv              # Raw ATT(g,t) — all group×time cells
    dynamic_cum_citations_na.csv            # Event-study aggregation (Figure 2)
    cohort_cum_citations_na.csv             # By-cohort aggregation (Figure 3)
    overall_cum_citations_na.csv            # Single overall ATT

    dynamic_cum_citations_na_gender_male.csv       # Figure 4a
    dynamic_cum_citations_na_gender_female.csv     # Figure 4b
    cohort_cum_citations_na_gender_male.csv        # Figure 4c
    cohort_cum_citations_na_gender_female.csv

    dynamic_cum_citations_na_career_early-career.csv   # Figure 5a
    dynamic_cum_citations_na_career_mid-career.csv     # Figure 5b
    dynamic_cum_citations_na_career_late-career.csv    # Figure 5c

    dynamic_cum_citations_na_region_Europe.csv         # Figure 6a
    dynamic_cum_citations_na_region_Asia.csv            # Figure 6c
    dynamic_cum_citations_na_region_Northern_America.csv # Figure 6b
    ...
```

## CSV Output Format

Each output CSV contains:

| Column | Description |
|--------|-------------|
| `event_time` | Relative time e = t - g (in dynamic files) |
| `group` | Treatment cohort = venue_year (in cohort files) |
| `att` / `ATT` | Average Treatment Effect on the Treated |
| `se` / `SE` | Standard error |
| `ci_lower` | Lower 95% CI |
| `ci_upper` | Upper 95% CI |
| `pvalue` | p-value |
| `subgroup` | Subgroup label (in heterogeneity files) |

## Usage Examples

```bash
# 1. General effect only (Figure 2)
python venue_did_csdid.py \
    --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
    --outcomes cum_citations_na cum_publication_count cum_funding_count

# 2. With all heterogeneity analyses (Figures 2-6)
python venue_did_csdid.py \
    --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
    --outcomes cum_citations_na cum_publication_count cum_funding_count \
    --heterogeneity gender career_stage region

# 3. All files, doubly robust estimator
python venue_did_csdid.py \
    --input_dir ../../data/matches/enriched_citations \
    --outcomes cum_citations_na \
    --heterogeneity gender career_stage region \
    --est_method dr

# 4. Pre-venue citations (for the spillover analysis, Figure SI 17)
python venue_did_csdid.py \
    --input ../../data/matches/enriched_citations/merged_physics_Nature_enriched.csv \
    --outcomes cum_citations_old \
    --heterogeneity gender career_stage region

# 5. Not-yet-treated control (SI 5.6) — edit script: control_group='notyettreated'
```

## Validation Against Stata

For key results, compare Python output against your existing Stata estimates:
- The 10-year dynamic ATT for cum_citations_na should be close to SV1: 841
- Cohort effects should show emergence after 1990s
- Gender gap should show male > female in science

Small numerical differences are expected (different bootstrap implementations,
numerical precision) — see Cunningham's multi-analyst comparison.

## Dependencies

```
pip install csdid pandas numpy scipy
```

Optional for robustness:
```
pip install pyfixest   # Extended TWFE / Sun-Abraham
pip install etwfe      # Wooldridge ETWFE
```
