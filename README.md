# README

> The provided code and data are designed to investigate the impact of prestigious venues in science and the arts. We aim to deepen understanding of how venues influence individual career trajectories and their role in shaping opportunities and outcomes.
> 

![image.png](image.png)

## Authors

Yixuan Liu, Rodrigo Dorantes-Gilardi, Larry Han, and Albert-László Barabási.

## Description

- This repository contains the following datasets collected from:

[**Dimensions.ai**](http://Dimensions.ai) (https://www.dimensions.ai/)

[**Artfacts.net](http://Artfacts.net)** (https://artfacts.net/)

- And with codes and snippets are conducted through:

Python 3.11.8

Stata/MP 18.0

Google BigQuery

## Table of contents

### Science and art data

We assembled comprehensive datasets of scientific publication venues and visual art exhibition venues, spanning two major domains of science and art. We provided in two sections data of venue-participated artists/scientists and their matched pairs after matching, which is adequate to replicate the main results. Which could be found under:

For scientific publications, the dataset comprises the complete publication histories of 3.4 million physicists extracted from the [Dimensions.ai](http://dimensions.ai/), including bibliographic details such as publication titles, author affiliations, journal names, publication dates, and citation counts. To explore the effect of prestigious venues on scientific careers, we focused on physics publications, identifying key venues including the top three general science journals (SV1, SV2, SV3) and a leading physics disciplinary journal (SV4).  which could be found under: 

Variables collected:

- researcher_id
- cum_publication_count
- cum_corresponding_count
- cum_total_citations
- cum_citations_na
- old_paper_cum_citations
- cum_funding_count
- career_stage
- first_year
- year: current calendar year
- first_publish_year
- to_year: year to exposure
- gender
- affiliation_country_codes
- is_journal: identify at the current year, whether individual is treated or not, if it is treated then equals 1, otherwise 0

For art exhibitions, we compiled data from reputable art database [artfacts.net](http://artfacts.net/), covering over 1 million artists with exhibition histories detailing nationality, birth and death dates, associated art movements, and records of exhibitions. This dataset enabled analysis of the impact of participation in two prestigious invitation-only (AV1 and AV2) and other historical art biennials on career development. Which could be found under: 

Variables collected:

- 

### Replicating main results

This includes demo codes to replicate the main results, which includes 

- Effect of venue access upon exposure:
- Variation of effect
- Heterogeneity of effect: gender
- Heterogeneity of effect: career stage
- Heterogeneity of effect: geographic

### Data extraction

We used Google BigQuery to extract the subsets for Dimensions.ai. With detailed documentation (https://docs.dimensions.ai/bigquery/). And is accessible through paid service additional to Dimensions Analytics.

### Matching algorithms

In this part we use data after preprocessing, to find the venue-participated individuals with control pairs, through a hybrid matching scheme by coarsened exact matching and dynamic matching based on distance.

We used Python 3.11.8, with the below modules needed:

```python

```

Scientists, please refer to the codes under:

Artists, please refer to the codes under: