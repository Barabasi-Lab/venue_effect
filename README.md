# README

> The provided code and data are designed to investigate the impact of prestigious venues in science and the arts. We aim to deepen understanding of how venues influence individual career trajectories and their role in shaping opportunities and outcomes.
> 

![image.png](image.png)

## Authors

Yixuan Liu, Rodrigo Dorantes-Gilardi, Larry Han, and Albert-László Barabási.

## Description

- This repository contains the following datasets collected from:

[**Dimensions.ai**](http://Dimensions.ai), scientific research database

[**Artfacts.net**](http://Artfacts.net), art exhibitions and market database

- And with codes and snippets conducted through:

`Python 3.11.8`

`Stata/MP 18.0`

`Google BigQuery`

## Table of contents

### Science and art venue data

We assembled comprehensive datasets of scientific publication venues and visual art exhibition venues, spanning two major domains of science and art. We provided in two sections data of venue-participated artists/scientists and their matched pairs after matching, which is adequate to replicate the main results. Which could be found under:

**Science venues: top journals**

For scientific publications, the dataset comprises the complete publication histories of 3.4 million physicists extracted from the [Dimensions.ai](http://dimensions.ai/), including bibliographic details such as publication titles, author affiliations, journal names, publication dates, and citation counts. Refer to: `data/science`, including: `nature_matched.csv`, `science_matched.csv`, `pnas_matched.csv`.

Variables collected include:

- `researcher_id`
- `cum_publication_count`
- `cum_corresponding_count`
- `cum_total_citations`
- `cum_citations_na`: cumulative citations
- `old_paper_cum_citations`
- `cum_funding_count`
- `career_stage`
- `first_year`: first year of all publications
- `year`: current calendar year
- `first_publish_year`: first year when researcher publish such venue
- `to_year`: year to exposure
- `gender`: 1 for female, 0 for male
- `affiliation_country_codes`
- `is_journal`: identify at the current `year`, whether individual is treated or not, if it is treated then equals `1`, otherwise `0`.

**Art venues: top biennials**

For art exhibitions, we compiled data from reputable art database [artfacts.net](http://artfacts.net/), covering over 1 million artists with exhibition histories detailing nationality, birth and death dates, associated art movements, and records of exhibitions. This dataset enabled analysis of the impact of participation in two prestigious invitation-only (AV1) and other historically well-known art biennials on career development. Refer to: `data/art`, including: `venice_matched.csv`, and `other_biennials_matched.csv`.

Variables collected include:

- `artist_id`
- `artistic_id` (optional): only for `other_biennials_matched.csv`
- `end_year`: current calender year
- `B`: number of biennale
- `F`: number of art fairs
- `G`: number of group exhibitions
- `S`: number of solo exhibitions
- `min_year`: first year of exhibition
- `artist_birth_year`
- `career_stage`
- `age`
- `biennale_year`
- `year_diff`: year to exposure
- `institution_name` (optional): only for `other_biennials_matched.csv` , identifying the biennial names
- `institution_id` (optional): only for `other_biennials_matched.csv`
- `biennale_treated`: identify at the current `end_year`, whether individual is treated or not, if it is treated then equals `1`, otherwise `0`.

### Replicating main results

This includes demo codes to replicate the main results, we used `Stata/MP 18.0`, and the functions from `xthdidregress`(Documentation: https://www.stata.com/manuals/causalxthdidregress.pdf) Since `xthdidregress` involves estimation of combinations of group and time, we recommend users to set the maximum number of attributes larger as a start by using: `set maxvar 120000`.

To replicate the results, one needs to go through stata do_files under `code/art/effect_estimation` and `code/science/effect_estimation`, and retrieve estimates of effect by using (with `nature_exposure.do` as example):

```r
do nature_exposure.do
```

Which will provide the estimated result in `nature_exposure_cit_atet.csv`, `nature_exposure_prod_atet.csv`, `nature_exposure_grant_atet.csv`.

- Venue effect upon **exposure**:
    - Science: including `nature_exposure.do`, `science_exposure.do`, `pnas_exposure.do`, results under `results\science`: `nature_exposure_cit_atet.csv`, `science_exposure_cit_atet.csv`, `pnas_exposure_cit_atet.csv` etc.
    - Art: including `venice_exposure.do`, results under `results\art`: `venice_exposure_solo_atet.csv`, `venice_exposure_group_atet.csv`, `venice_exposure_fair_atet.csv`, etc.
- **Variation** of effect:
    - Science: including `nature_variation_cit.do`, `nature_variation_prod.do`, etc.
    - Art: including `venice_variation_solo.do`, `venice_variation_group.do`, etc.
- **Heterogeneity** of effect: (including gender, career stage, geographic representation)
    - Science: including `nature_exposure_gender.do`, `nature_exposure_career.do`, etc.
    - Art: including `venice_exposure_gender.do`, `venice_exposure_career.do`, etc.

### Data extraction

We used Google BigQuery to extract the subsets for Dimensions.ai. With detailed documentation (https://docs.dimensions.ai/bigquery/). And is accessible through paid service additional to Dimensions Analytics.

### Matching algorithms

In this part we use data after preprocessing, to find the venue-participated individuals with control pairs, through a hybrid matching scheme by coarsened exact matching and dynamic matching based on distance and nearest neiughbours.

We used `Python 3.11.8`, and modules below need to be included:

```python
pandas
numpy
glob
scipy.spatial.distance
fastdtw
```

Scientists: `code/science/matching`

Artists: `code/art/matching`