#!/usr/bin/env python3
"""
==============================================================================
Enrich matched art panels with pre-venue exhibition title reuse metrics
==============================================================================

ANALOGY TO SCIENCE:
  enrich_citations.py counts how many citations an author's OLD papers
  receive each year → citations_old / cum_citations_old.

  This script counts how many solo exhibitions in a given year have titles
  that reuse a thematic phrase from the artist's pre-venue solo exhibitions
  → s_titles (annual), S_titles (cumulative).

  S_titles is to S what cum_citations_old is to cum_citations:
  it measures how much of the artist's *established* body of work
  continues to circulate at exhibitions.

DICTIONARY BUILDING:
  For each artist, collect all solo exhibitions (type='S') with
  end_year STRICTLY BEFORE biennale_year.  For each title:
    1. Strip artist name (full name, last name, possessives, prefixes).
    2. Normalize: lowercase, strip accents/punctuation, collapse whitespace.
    3. Remove common filler words and year-like tokens.
    4. Split on delimiters (commas, semicolons, &, "and"/"und"/"et"/"y").
    5. From each chunk, generate:
       - The full cleaned chunk (e.g. "campbell soup cans")
       - All individual content words >= 4 chars (e.g. "campbell", "soup", "cans")
       - All consecutive bigrams (e.g. "campbell soup", "soup cans")
    6. Skip [Self-Titled] entries entirely.
    7. Remove any token in the global stopword list.
    8. Deduplicate.

MATCHING (exact, not fuzzy):
  For each solo exhibition in any year:
    1. Clean the title identically (strip name, normalize).
    2. Generate the same token set (chunks + words + bigrams).
    3. If ANY token appears in the pre-venue dictionary → match.
    4. Each exhibition counts at most once (binary).

COLUMNS ADDED:
  - s_titles:   annual count of matching solo exhibitions
  - S_titles:   cumulative sum of s_titles over time

Usage:
    python enrich_titles.py \
        --input ../../data/matches/matched_venice_biennale.csv

    python enrich_titles.py --input_dir ../../data/matches

    python enrich_titles.py \
        --input ../../data/matches/matched_venice_biennale.csv \
        --show_dictionary
"""

import argparse
import re
import unicodedata
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# I/O helpers (CSV + Parquet)
# =============================================================================

def load_df(path):
    """Load a DataFrame from CSV (auto-detect separator) or Parquet."""
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    # CSV: auto-detect separator
    with open(path, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    return pd.read_csv(path, sep=sep)

def save_df(df, path, sep=','):
    """Save a DataFrame as CSV or Parquet depending on extension."""
    path = Path(path)
    if path.suffix == '.parquet':
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, sep=sep)

def detect_csv_sep(path):
    """Detect CSV separator (for output format matching)."""
    path = Path(path)
    if path.suffix == '.parquet':
        return ','
    with open(path, 'r') as f:
        first_line = f.readline()
    return ';' if ';' in first_line else ','

# =============================================================================
# Config
# =============================================================================

ARTFACTS_PATH_DEFAULT = '../../data/artist_info/artfacts_artists_all.csv'

MIN_WORD_LENGTH = 4     # individual words must be >= 4 chars
MIN_TOKEN_LENGTH = 3    # any token (bigrams, chunks) must be >= 3 chars

# Aggressively broad stopword list — removes generic exhibition language
# so the dictionary only contains thematic content words.
STOPWORDS = {
    # ── Exhibition / art generic ──
    'exhibition', 'exhibitions', 'exhibit', 'solo', 'show', 'shows',
    'retrospective', 'retrospektive', 'retrospectiva',
    'selected', 'selection', 'selections', 'overview', 'survey',
    'works', 'work', 'oeuvre', 'oeuvres', 'opus', 'werke', 'werk',
    'recent', 'neue', 'nuevo', 'nueva', 'nouveau', 'nouvelle', 'novo',
    'paintings', 'painting', 'peintures', 'peinture', 'malerei',
    'drawings', 'drawing', 'dessins', 'dessin', 'zeichnungen',
    'prints', 'print', 'estampes', 'drucke', 'grafik', 'graphik',
    'sculptures', 'sculpture', 'skulpturen', 'plastik',
    'photographs', 'photography', 'photos', 'photo', 'fotografie',
    'images', 'image', 'bilder', 'bild',
    'installations', 'installation', 'objects', 'object', 'objekte',
    'collages', 'collage', 'watercolors', 'watercolour', 'aquarelle',
    'lithographs', 'lithograph', 'etchings', 'etching',
    'screenprints', 'screenprint', 'serigraphs', 'serigraph',
    'woodcuts', 'woodcut', 'engravings', 'engraving',
    'early', 'late', 'later', 'small', 'large', 'first', 'last',
    'part', 'parts', 'series', 'volume', 'volumes',
    'edition', 'editions', 'catalogue', 'catalog', 'katalog',
    'collection', 'collections', 'sammlung',
    'galerie', 'gallery', 'galleries', 'museum', 'museums',
    'kunsthalle', 'kunsthaus', 'kunstverein', 'haus', 'atelier',
    'studio', 'project', 'projects', 'space',
    'group', 'personal', 'individual',
    'annual', 'biennial', 'biennale', 'triennial',
    'complete', 'major', 'minor', 'various', 'miscellaneous',
    'masterworks', 'masterpieces', 'highlights', 'treasures',
    'homage', 'hommage', 'tribute', 'dedicated',
    'contemporary', 'modern', 'abstract', 'figurative',
    'canvas', 'paper', 'bronze', 'steel', 'glass', 'wood', 'stone',
    'color', 'colour', 'colors', 'colours', 'farben',
    'black', 'white', 'blue', 'green', 'yellow', 'gold', 'silver',
    'kunst', 'arte', 'arts',
    # ── Prepositions / articles / conjunctions ──
    'the', 'and', 'for', 'from', 'with', 'into', 'about',
    'that', 'this', 'these', 'those', 'than', 'then',
    'not', 'but', 'also', 'only', 'just', 'very',
    'some', 'other', 'others', 'another', 'each', 'every',
    'all', 'more', 'most', 'many', 'much', 'few',
    'new', 'old', 'after', 'before', 'between', 'through',
    'over', 'under', 'above', 'below',
    'des', 'der', 'die', 'das', 'dem', 'den',
    'ein', 'eine', 'einem', 'einen', 'einer',
    'von', 'und', 'oder', 'auf', 'aus', 'bei', 'mit',
    'nach', 'seit', 'uber', 'zum', 'zur',
    'les', 'aux', 'dans', 'sur', 'sous', 'avec', 'sans',
    'pour', 'chez', 'entre', 'vers',
    'los', 'las', 'del', 'con', 'por', 'sin',
    'una', 'uno', 'unos', 'unas', 'como', 'sobre',
    'degli', 'delle', 'della', 'dello', 'nell', 'nella', 'nelle',
    'nel', 'alla', 'alle', 'allo', 'agli',
    'dos', 'nos', 'nas', 'pelo', 'pela',
    'com', 'para', 'sem', 'sob',
    'het', 'een', 'van', 'uit', 'met', 'voor', 'naar', 'door',
    'an', 'at', 'by', 'if', 'in', 'is', 'it', 'no', 'of',
    'on', 'or', 'so', 'to', 'up', 'we',
    'de', 'di', 'du', 'el', 'en', 'et', 'il', 'la', 'le',
    'lo', 'li', 'se', 'si', 'un', 'da', 'do', 'em', 'na',
    'op', 'om', 'te',
}

YEAR_PATTERN = re.compile(r'^\d{4}[\-\u2013\u2014]?\d{0,4}$')

# =============================================================================
# Text cleaning
# =============================================================================

def strip_accents(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def clean_artist_name_from_title(title, full_name, last_name):
    """Remove artist name from title. Returns '[Self-Titled]' if nothing remains."""
    if not title or not full_name:
        return title

    t = title.strip()

    for name in [full_name, last_name]:
        if not name:
            continue
        esc = re.escape(name)
        t = re.sub(rf'(?i)^{esc}\s*[:\.\-\u2013\u2014]\s*', '', t)
        t = re.sub(rf"(?i)^{esc}'?s?\s+", '', t)
        t = re.sub(rf'(?i)\s*[-\u2013\u2014:.,]\s*(by|von|de|di)\s+{esc}$', '', t)
        t = re.sub(rf'(?i)\s+(by|von|de|di)\s+{esc}$', '', t)

    t = t.strip()
    t_lower = t.lower().strip()
    name_variants = {full_name.lower()}
    if last_name:
        name_variants.add(last_name.lower())
    if t_lower in name_variants or t_lower == '':
        return '[Self-Titled]'
    return t

def normalize_text(s):
    s = s.lower()
    s = strip_accents(s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_tokens_from_chunk(chunk_text):
    """
    From a cleaned text chunk, extract:
      - The full chunk (content words rejoined)
      - Individual words >= MIN_WORD_LENGTH
      - Consecutive bigrams
    Returns a set of normalized tokens.
    """
    norm = normalize_text(chunk_text)
    words = norm.split()

    content_words = [
        w for w in words
        if w not in STOPWORDS
        and not YEAR_PATTERN.match(w)
        and len(w) >= MIN_TOKEN_LENGTH
    ]

    if not content_words:
        return set()

    tokens = set()

    # Full chunk
    full = ' '.join(content_words)
    if len(full) >= MIN_TOKEN_LENGTH:
        tokens.add(full)

    # Individual words (stricter length for standalone words)
    for w in content_words:
        if len(w) >= MIN_WORD_LENGTH:
            tokens.add(w)

    # Consecutive bigrams
    if len(content_words) >= 2:
        for i in range(len(content_words) - 1):
            bigram = f'{content_words[i]} {content_words[i+1]}'
            tokens.add(bigram)

    return tokens

def tokenize_title(clean_title):
    """
    Split a cleaned title on delimiters, then extract tokens from each chunk.
    Returns a set of normalized tokens.
    """
    if not clean_title or clean_title == '[Self-Titled]':
        return set()

    parts = re.split(r'[,;&]+|\band\b|\bund\b|\bet\b', clean_title)

    all_tokens = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        all_tokens.update(extract_tokens_from_chunk(part))

    return all_tokens

# =============================================================================
# Dictionary building
# =============================================================================

def build_artist_dictionary(artist_exhibitions, artist_name, artist_last_name,
                            biennale_year):
    """
    Build pre-venue dictionary from solo exhibitions before biennale_year.
    Returns (frozenset of tokens, list of debug tuples).
    """
    pre = artist_exhibitions[
        (artist_exhibitions['type'] == 'S') &
        (artist_exhibitions['end_year'] < biennale_year)
    ]

    if pre.empty:
        return frozenset(), []

    all_tokens = set()
    debug_info = []

    for _, row in pre.iterrows():
        title = str(row.get('title', ''))
        if not title or title == 'nan':
            continue

        clean = clean_artist_name_from_title(title, artist_name, artist_last_name)
        if clean == '[Self-Titled]':
            debug_info.append((title, '[Self-Titled]', set()))
            continue

        tokens = tokenize_title(clean)
        all_tokens.update(tokens)
        debug_info.append((title, clean, tokens))

    return frozenset(all_tokens), debug_info

# =============================================================================
# Matching
# =============================================================================

def exhibition_matches_dictionary(title, artist_name, artist_last_name,
                                  dictionary):
    """
    Binary: does this exhibition title contain any token from the
    pre-venue dictionary?  Exact set intersection after identical
    normalization.
    """
    if not dictionary or not title or title == 'nan':
        return False

    clean = clean_artist_name_from_title(title, artist_name, artist_last_name)
    if clean == '[Self-Titled]':
        return False

    title_tokens = tokenize_title(clean)
    if not title_tokens:
        return False

    return bool(title_tokens & dictionary)

# =============================================================================
# Compute title reuse counts
# =============================================================================

def compute_title_reuse(matched_df, artfacts_df, artfacts_index=None,
                        show_dictionary=False, n_show=5):
    """
    For each artist x year, count solo exhibitions whose title
    matches the pre-venue dictionary.
    Returns DataFrame: artist_id, year, s_titles

    artfacts_index: optional dict {artist_id: DataFrame} for O(1) lookup.
    """
    artist_vy = matched_df.groupby('artist_id')['biennale_year'].first().to_dict()

    # Build index if not provided
    if artfacts_index is None:
        print('  Building artfacts index...')
        artfacts_index = {aid: grp for aid, grp in artfacts_df.groupby('artist_id')}

    all_results = []
    shown = 0
    artist_ids = matched_df['artist_id'].unique()
    n_with_dict = 0
    total_dict_size = 0

    print(f'  Processing {len(artist_ids):,} artists...')
    t0 = time.time()

    for i, artist_id in enumerate(artist_ids):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f'    {i+1:,}/{len(artist_ids):,} ({elapsed:.0f}s)', flush=True)

        biennale_year = artist_vy.get(artist_id)
        if pd.isna(biennale_year):
            continue
        biennale_year = int(biennale_year)

        aex = artfacts_index.get(artist_id)
        if aex is None or aex.empty:
            continue

        row0 = aex.iloc[0]
        artist_name = str(row0.get('artist_name', ''))
        artist_last = str(row0.get('artist_last_name', ''))

        dictionary, debug_info = build_artist_dictionary(
            aex, artist_name, artist_last, biennale_year
        )

        if dictionary:
            n_with_dict += 1
            total_dict_size += len(dictionary)

        if show_dictionary and shown < n_show and dictionary:
            print(f'\n  {"="*60}')
            print(f'  DICTIONARY: {artist_name} (biennale_year={biennale_year})')
            print(f'  {"="*60}')
            print(f'  {len(dictionary)} tokens:')
            for tok in sorted(dictionary):
                print(f'    - "{tok}"')
            print(f'  Built from {len(debug_info)} pre-venue solo exhibitions:')
            for orig, clean, toks in debug_info:
                toks_str = ', '.join(sorted(toks)) if toks else '(none)'
                print(f'    "{orig}"')
                print(f'      -> "{clean}" -> [{toks_str}]')
            print()
            shown += 1

        # Score all solo exhibitions by year
        solo = aex[aex['type'] == 'S']
        if solo.empty:
            continue

        for year, year_group in solo.groupby('end_year'):
            year = int(year)
            count = 0
            for _, exh in year_group.iterrows():
                title = str(exh.get('title', ''))
                if exhibition_matches_dictionary(
                    title, artist_name, artist_last, dictionary
                ):
                    count += 1
            if count > 0:
                all_results.append({
                    'artist_id': artist_id,
                    'year': year,
                    's_titles': count,
                })

    elapsed = time.time() - t0
    print(f'  Done ({elapsed:.0f}s)')
    print(f'  Artists with non-empty dictionary: {n_with_dict:,}')
    if n_with_dict > 0:
        print(f'  Avg dictionary size: {total_dict_size / n_with_dict:.1f} tokens')

    if not all_results:
        return pd.DataFrame(columns=['artist_id', 'year', 's_titles'])

    return pd.DataFrame(all_results)

# =============================================================================
# Enrich one file
# =============================================================================

def enrich_one_file(input_path, output_path, artfacts_df,
                    artfacts_index=None, show_dictionary=False):

    print(f'\n{"="*60}')
    print(f'Enriching: {input_path}')
    print(f'{"="*60}')

    input_path = Path(input_path)
    output_path = Path(output_path)
    sep = detect_csv_sep(input_path)

    df = load_df(input_path)
    print(f'  Loaded: {len(df):,} rows, {df["artist_id"].nunique():,} artists')

    if 'year' not in df.columns and 'end_year' in df.columns:
        df['year'] = df['end_year']

    # Ensure artist_id types match
    df['artist_id'] = df['artist_id'].astype(str)

    print('  Computing title reuse metrics...')
    annual = compute_title_reuse(df, artfacts_df,
                                 artfacts_index=artfacts_index,
                                 show_dictionary=show_dictionary)

    if annual.empty:
        print('  WARNING: No title reuse data computed.')
        df['s_titles'] = 0
        df['S_titles'] = 0
        df.to_csv(output_path.with_suffix('.csv'), index=False, sep=sep)
        df.to_parquet(output_path.with_suffix('.parquet'), index=False)
        return df

    print(f'  {len(annual):,} artist-year rows with title matches')

    # ── Cumulative on full year range ──
    annual = annual.sort_values(['artist_id', 'year'])
    annual['S_titles'] = annual.groupby('artist_id')['s_titles'].cumsum()

    # ── Merge onto panel ──
    print('  Merging onto matched panel...')
    df['year'] = df['year'].astype(int)
    annual['year'] = annual['year'].astype(int)
    annual['artist_id'] = annual['artist_id'].astype(str)

    df = df.merge(
        annual[['artist_id', 'year', 's_titles', 'S_titles']],
        on=['artist_id', 'year'], how='left'
    )
    df['s_titles'] = df['s_titles'].fillna(0).astype(int)

    # ── Forward-fill cumulative properly ──
    # Build a full cumulative lookup, then for panel years without
    # exhibition data, carry forward the last known cumulative value.
    cum_lookup = annual.set_index(['artist_id', 'year'])['S_titles'].to_dict()

    # Pre-build per-artist cumulative series for fast lookup
    artist_cum = {}
    for aid, grp in annual.groupby('artist_id'):
        artist_cum[aid] = grp.set_index('year')['S_titles'].sort_index()

    def _fill_cum(group):
        aid = group['artist_id'].iloc[0]
        cseries = artist_cum.get(aid)
        if cseries is None or cseries.empty:
            group['S_titles'] = 0
            return group

        result = []
        for y in group['year']:
            key = (aid, y)
            if key in cum_lookup:
                result.append(cum_lookup[key])
            else:
                prior = cseries[cseries.index <= y]
                if not prior.empty:
                    result.append(int(prior.iloc[-1]))
                else:
                    result.append(0)
        group['S_titles'] = result
        return group

    df = df.sort_values(['artist_id', 'year'])
    df = df.groupby('artist_id', group_keys=False).apply(_fill_cum)
    df['S_titles'] = df['S_titles'].fillna(0).astype(int)

    # ── Save (both CSV and Parquet) ──
    # Always save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, sep=sep)
    print(f'\n  Saved: {csv_path}')

    # Also save Parquet
    pq_path = output_path.with_suffix('.parquet')
    df.to_parquet(pq_path, index=False)
    print(f'  Saved: {pq_path}')
    print(f'    {df["artist_id"].nunique():,} artists, {len(df):,} rows')
    print(f'    New columns: s_titles (annual), S_titles (cumulative)')

    if 'year_diff' in df.columns:
        for label, mask in [('Pre-venue', df['year_diff'] < 0),
                            ('Post-venue', df['year_diff'] >= 0)]:
            sub = df[mask]
            print(f'    {label:12s} mean s_titles={sub["s_titles"].mean():.3f}  '
                  f'mean S_titles={sub["S_titles"].mean():.2f}')

    return df

# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Enrich matched art panels with pre-venue title reuse')
    ap.add_argument('--input', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--input_dir', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Default: ../../data/matches/enriched_titles')
    ap.add_argument('--artfacts', type=str, default=ARTFACTS_PATH_DEFAULT)
    ap.add_argument('--suffix', type=str, default='_enriched_titles')
    ap.add_argument('--show_dictionary', action='store_true')

    args = ap.parse_args()

    if not args.input and not args.input_dir:
        ap.error('Provide either --input or --input_dir')

    artfacts_path = Path(args.artfacts)
    print(f'Loading artfacts: {artfacts_path}')
    t0 = time.time()

    if artfacts_path.suffix == '.parquet':
        artfacts_df = pd.read_parquet(artfacts_path)
    else:
        artfacts_df = pd.read_csv(artfacts_path)

    if 'end_year' in artfacts_df.columns:
        artfacts_df['end_year'] = pd.to_numeric(
            artfacts_df['end_year'], errors='coerce')
    elif 'end_date' in artfacts_df.columns:
        artfacts_df['end_year'] = pd.to_datetime(
            artfacts_df['end_date'], errors='coerce').dt.year

    artfacts_df['artist_id'] = artfacts_df['artist_id'].astype(str)

    print(f'  {len(artfacts_df):,} exhibitions, '
          f'{artfacts_df["artist_id"].nunique():,} artists '
          f'({time.time() - t0:.1f}s)')

    # Pre-index artfacts by artist_id (O(1) lookup per artist)
    print('  Building artfacts index...')
    t1 = time.time()
    artfacts_index = {aid: grp for aid, grp in artfacts_df.groupby('artist_id')}
    print(f'  Index built ({time.time() - t1:.1f}s)')

    default_out = '../../data/matches/enriched_titles'

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else Path(default_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (input_path.stem + args.suffix + '.csv')
        enrich_one_file(input_path, output_path, artfacts_df,
                        artfacts_index=artfacts_index,
                        show_dictionary=args.show_dictionary)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else Path(default_out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Glob both CSV and Parquet
        files = sorted(
            list(input_dir.glob('matched_*.csv')) +
            list(input_dir.glob('matched_*.parquet'))
        )
        # Deduplicate: if both .csv and .parquet exist for same stem, prefer parquet
        seen_stems = {}
        for f in files:
            stem = f.stem
            if stem not in seen_stems or f.suffix == '.parquet':
                seen_stems[stem] = f
        files = sorted(seen_stems.values())

        print(f'\nFound {len(files)} matched files in {input_dir}')
        print(f'Output directory: {out_dir}')

        for fpath in files:
            if args.suffix in fpath.stem:
                continue
            output_path = out_dir / (fpath.stem + args.suffix + '.csv')
            enrich_one_file(fpath, output_path, artfacts_df,
                            artfacts_index=artfacts_index,
                            show_dictionary=args.show_dictionary)

    print('\n' + '=' * 60)
    print('ALL DONE')
    print('=' * 60)

if __name__ == '__main__':
    main()