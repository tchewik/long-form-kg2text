import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import requests
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "LAGRANGE-align/1.0 (contact@example.com)"}


DATA_DIR = Path("data/lagrange")
TRAIN_PATH = DATA_DIR / "lagrange_train_resolved.json"
TEST_PATH = DATA_DIR / "lagrange_test_resolved.json"
WIKI_DIR = Path(".wiki_cache")
TRAIN_OUT_ROWS_PATH = DATA_DIR / "lagrange_doc_train.jsonl"
TEST_OUT_ROWS_PATH = DATA_DIR / "lagrange_doc_test.jsonl"


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

nltk.download("punkt")


def fetch_wiki_page_plaintext(page_id: str, max_chars: int = 200_000) -> str:
    """
    Fetch a Wikipedia page by numeric pageid and return plaintext extract.
    NOTE: this only works for numeric page IDs, not QIDs.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": page_id,
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    if r.status_code == 403:
        # occasionally Wikipedia rate-limits or blocks briefly
        time.sleep(1)
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(page_id), {})
    extract = page.get("extract", "") or ""
    return extract[:max_chars]


def fetch_wiki_page_plaintext_by_title(title: str, max_chars: int = 200_000) -> str:
    """
    Fetch a Wikipedia page by its title and return plaintext extract.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    if r.status_code == 403:
        time.sleep(1.0)
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    # pages is a dict keyed by pageid, take first
    if not pages:
        return ""
    page = next(iter(pages.values()))
    extract = page.get("extract", "") or ""
    return extract[:max_chars]


def read_wiki_page(page_id, page_title: str | None = None) -> str:
    """
    Read a cached wiki page given its resolved_page_id.

    - If page_id is Q*:
        * If WIKI_DIR/pages/Q*.txt exists -> read it.
        * Else, if page_title is provided -> fetch by title from Wikipedia,
          save to WIKI_DIR/pages/Q*.txt, then use it.
    - If page_id is numeric:
        * Use WIKI_DIR/page_<id>.txt if exists,
        * else fetch by numeric pageid, save to cache.

    In all cases we drop '== References ==' tail.
    """
    page_id = str(page_id)

    if page_id.startswith("Q"):
        # ---- QID path, but use resolved_page_title to fetch if missing ----
        path = WIKI_DIR / "pages" / f"{page_id}.txt"

        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            if not page_title:
                raise FileNotFoundError(
                    f"No cached text for QID {page_id} and no page_title provided; "
                    f"expected at {path}"
                )

            # Doesn't find some (outdated) pages automatically
            fix_dict = {
                'Q1751429': 'fast-food restaurant',
                'Q4417823': 'serration',
                'Q4118378': 'Hemorheology',
                'Q4990963': 'List of circus skills',
                'Q5455089': 'Fishing techniques',
                'Q177493': 'Gauss (unit)',
                'Q4803677': 'Asbestos-related diseases',
                'Q180548': 'Neolithic Revolution',
                'Q19865437': '12/12/12 (film)',
                'Q367700': 'Ciclosporin',
                'Q5097944': 'Childbirth positions',
                'Q9639': 'Gastrointestinal tract',
                'Q7577080': 'Spiders (album)',
                'Q838062': 'occam (programming language)',
                'Q201479': 'Carbonyl group',
                'Q731350': 'Cuban convertible peso',
                'Q5384093': 'eqn (software)',
                'Q864928': 'List of life sciences',
                'Q862867': 'Taste bud',
                'Q17508553': 'Nursing Home (album)',
                'Q154549': 'Carpenter, Texas',
                'Q484652': 'International Organization (journal)',
                'Q15967387': 'pandas (software)',
            }
            for key in fix_dict:
                if page_id == key:
                    page_title = fix_dict.get(key)

            # Fetch by title from Wikipedia and cache as QID.txt
            logger.info(
                "Cache miss for QID %s (title=%r), fetching by title from Wikipedia...",
                page_id,
                page_title,
            )
            text = fetch_wiki_page_plaintext_by_title(page_title)
            if not text:
                raise RuntimeError(
                    f"Failed to fetch Wikipedia page by title {page_title!r} "
                    f"for QID {page_id}"
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    else:
        # ---- Numeric pageid path ----
        path = WIKI_DIR / f"page_{page_id}.txt"

        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            logger.info(
                "Cache miss for numeric page %s, fetching from Wikipedia...", page_id
            )
            text = fetch_wiki_page_plaintext(page_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    # Optional: drop references section to avoid noisy tail
    text = text.split("== References ==", 1)[0]
    return text


def split_into_sentences(text: str):
    """
    Sentence splitting via NLTK.
    """
    text = re.sub(r"\s+", " ", text.strip())
    return sent_tokenize(text) if text else []


def normalize(s: str) -> str:
    """
    Normalization to make fuzzy matching more robust.
    """
    s = s.lower()
    # remove [1], [23] style footnote markers
    s = re.sub(r"\[[0-9]+\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def align_page(group: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """
    Align all sentences belonging to a single resolved_page_id
    to sentences on that wiki page.

    Returns a copy of the group with added columns:
      - wiki_sent_idx
      - wiki_sentence
    """
    page_id = group.name
    resolved_page_title = group["resolved_page_title"].sample(1).values[0]
    text = read_wiki_page(page_id, page_title=resolved_page_title)
    wiki_sents = split_into_sentences(text)
    wiki_norm = [normalize(s) for s in wiki_sents]

    df_sents = group["sentence"].tolist()
    df_norm = [normalize(s) for s in df_sents]

    # Compute all pairwise scores in fast C++ code, not Python
    # cdist returns a 2D array: shape (len(df_norm), len(wiki_norm))
    scores = process.cdist(
        df_norm,
        wiki_norm,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,  # values below cutoff become 0
    )

    # Collect candidates (score, df_idx, wiki_idx) for scores >= threshold
    di_idx, wi_idx = np.where(scores >= threshold)
    candidates = [
        (scores[di, wi], int(di), int(wi))
        for di, wi in zip(di_idx, wi_idx)
    ]

    # best first
    candidates.sort(reverse=True, key=lambda x: x[0])

    # greedy one-to-one assignment
    df_to_wiki = {}
    used_wiki = set()
    for score, di, wi in candidates:
        if di in df_to_wiki or wi in used_wiki:
            continue
        df_to_wiki[di] = wi
        used_wiki.add(wi)

    wiki_indices = []
    wiki_texts = []
    for di in range(len(group)):
        wi = df_to_wiki.get(di)
        wiki_indices.append(wi)
        wiki_texts.append(wiki_sents[wi] if wi is not None else None)

    out = group.copy()
    out["wiki_sent_idx"] = wiki_indices
    out["wiki_sentence"] = wiki_texts
    return out


def _align_page_wrapper(args):
    """
    Small wrapper so we can pass (page_id, group, threshold) to pool workers.
    """
    page_id, group, threshold = args
    group = group.copy()
    group.name = page_id
    return align_page(group, threshold=threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Align dataset sentences to Wikipedia pages.")
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: number of CPUs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Similarity threshold (0–100) for candidate sentence matches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_jobs = args.jobs or 1
    threshold = args.threshold

    logger.info("Using %d worker processes", n_jobs)
    logger.info("Similarity threshold: %.1f", threshold)

    # -------------------------------------------------------------
    logger.info("Loading dataframes...")
    train = pd.read_json(TRAIN_PATH, orient="records", lines=True)
    train["split"] = "train"
    test = pd.read_json(TEST_PATH, orient="records", lines=True)
    test["split"] = "test"

    train_page_ids = set(train["resolved_page_id"].unique())
    test_page_ids = set(test["resolved_page_id"].unique())
    overlap_page_ids = train_page_ids.intersection(test_page_ids)

    logger.info(
        "Found %d overlapping pages between train and test.",
        len(overlap_page_ids),
    )

    if overlap_page_ids:
        # all rows for overlapping pages should belong to train
        mask_overlap = test["resolved_page_id"].isin(overlap_page_ids)
        n_move = mask_overlap.sum()
        logger.info(
            "Reassigning %d rows from test → train due to page overlap.",
            n_move,
        )
        test.loc[mask_overlap, "split"] = "train"

    df = pd.concat([train, test], ignore_index=True)
    rm_qids = ['Q117488', 'Q1251441', 'Q154549', 'Q170519', 'Q15983979', 'Q218218',
               'Q2302678', 'Q2468392', 'Q3275103', 'Q721834', 'Q83460', 'Q2159907',
               'Q14756018', 'Q2165914', 'Q1788166', 'Q204832', 'Q1759969', 'Q18113858',
               'Q2110300', 'Q22731', 'Q2806573', 'Q34439356', 'Q3597938', 'Q3680646',
               'Q38166', 'Q4646459', 'Q4947541', 'Q513471', 'Q773403', 'Q877674',
               'Q912985', 'Q918646'
               ]
    df = df[~df["resolved_page_id"].isin(rm_qids)]

    if "resolved_page_id" not in df.columns or "sentence" not in df.columns:
        raise ValueError("Expected columns 'resolved_page_id' and 'sentence' in dataframes.")
    if "resolved_page_title" not in df.columns:
        raise ValueError("Expected column 'resolved_page_title' in dataframes.")

    # -------------------------------------------------------------
    logger.info("Filtering by resolved_page_title frequency >= 3...")
    orig_rows = len(df)
    title_counts = df["resolved_page_title"].value_counts()
    keep_titles = title_counts[title_counts >= 3].index
    df = df[df["resolved_page_title"].isin(keep_titles)].copy()

    logger.info(
        "Rows before filter: %d, after filter: %d. Titles kept: %d / %d.",
        orig_rows,
        len(df),
        len(keep_titles),
        len(title_counts),
    )

    if df.empty:
        logger.warning("No rows left after filtering by resolved_page_title frequency >= 3. Exiting.")
        return

    # -------------------------------------------------------------
    logger.info("Preparing groups for parallel sentence-to-text alignment...")
    grouped = list(df.groupby("resolved_page_id"))
    n_pages = len(grouped)
    logger.info("Found %d unique pages after filtering.", n_pages)

    tasks = [(page_id, group, threshold) for page_id, group in grouped]

    # -------------------------------------------------------------
    logger.info("Aligning sentences page-by-page in parallel...")
    aligned_parts = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_align_page_wrapper, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=n_pages, desc="Aligning pages"):
            aligned_parts.append(fut.result())

    aligned = pd.concat(aligned_parts, ignore_index=True)
    aligned = aligned[aligned.wiki_sentence.notna()]  # Removing sentences which were not successfully aligned
    title_counts = aligned["resolved_page_title"].value_counts()
    keep_titles = title_counts[title_counts >= 3].index   # Filter out by "less than 3 sentences" rule again
    aligned = aligned[aligned["resolved_page_title"].isin(keep_titles)].copy()

    # -------------------------------------------------------------
    logger.info("Sorting rows by page and wiki sentence index...")
    aligned_sorted = aligned.sort_values(
        by=["resolved_page_id", "wiki_sent_idx"],
        kind="mergesort",  # stable
    )

    # -------------------------------------------------------------
    logger.info("Filtering pages based on predicate count and alignment...")

    kept_page_ids = set()

    for page_id, g in aligned_sorted.groupby("resolved_page_id"):

        # ---- Condition 1: Keep only pages where wiki_sent_idx 0 or 1 exists ----
        wiki_idxs = g["wiki_sent_idx"].dropna().astype(int).tolist()
        if not any(idx in (0, 1) for idx in wiki_idxs):
            continue

        # ---- Condition 2: Count total predicates per page ----
        if "triples" not in g.columns:
            raise ValueError("Expected 'triples' column.")

        graph_text = g.triples.unique()
        unique_triples = set([triple.strip() for graph in graph_text for triple in graph.split('<sep>')])
        num_predicates = len(unique_triples)

        if 3 <= num_predicates <= 20:
            kept_page_ids.add(page_id)

    logger.info(f"Keeping {len(kept_page_ids)} pages.")

    # -------------------------------------------------------------
    TRAIN_OUT = DATA_DIR / "lagrange_doc_train.jsonl"
    TEST_OUT = DATA_DIR / "lagrange_doc_test.jsonl"

    logger.info(f"Writing filtered TRAIN data → {TRAIN_OUT}")
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for _, row in tqdm(
                aligned_sorted[aligned_sorted["split"] == "train"].iterrows(),
                total=(aligned_sorted["split"] == "train").sum(),
                desc="Writing TRAIN rows"
        ):
            if row["resolved_page_id"] not in kept_page_ids:
                continue
            line = row.to_dict()
            del line['split']
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info(f"Writing filtered TEST data → {TEST_OUT}")
    with TEST_OUT.open("w", encoding="utf-8") as f:
        for _, row in tqdm(
                aligned_sorted[aligned_sorted["split"] == "test"].iterrows(),
                total=(aligned_sorted["split"] == "test").sum(),
                desc="Writing TEST rows"
        ):
            if row["resolved_page_id"] not in kept_page_ids:
                continue
            line = row.to_dict()
            del line['split']
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
