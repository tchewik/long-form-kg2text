import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

import jsonlines
import nltk
import pandas as pd
import requests
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "LAGRANGE-cleaner/1.0 (contact@example.com)"}

LOCAL_PAGES_DIR = "data/lagrange/pages"
QID_INDEX_PATH = "data/lagrange/qid_label_index.parquet"
CACHE_DIR = ".wiki_cache"
SEARCH_CACHE_DIR = os.path.join(CACHE_DIR, "search")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
SQLITE_CACHE_PATH = "data/lagrange/qid_wiki_cache.sqlite"

RESOLVE_CACHE_PATH = os.path.join(CACHE_DIR, "sentence_resolve_cache.sqlite")
WIKI_SEARCH_SQLITE_PATH = os.path.join(CACHE_DIR, "wiki_search_cache.sqlite")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

nltk.download("punkt")
nltk.download('punkt_tab')

G_PAGE_SENTS = {}
G_TITLE_CANDIDATES = {}
G_MIN_SENTENCE_SCORE = 0.5

_SQLITE_CONN = None
_SQLITE_TABLE = None


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove [1], [23] style footnote markers
    s = re.sub(r"\[[0-9]+\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_title_for_match(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def split_into_sentences(text: str):
    """
    Sentence splitting via NLTK.
    """
    text = re.sub(r"\s+", " ", text.strip())
    return sent_tokenize(text) if text else []


def sentence_cache_key(title: str, sentence: str) -> str:
    raw = f"{title}||{sentence}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _sqlite_connect(db_path=SQLITE_CACHE_PATH):
    global _SQLITE_CONN, _SQLITE_TABLE
    if _SQLITE_CONN is not None:
        return
    if not os.path.exists(db_path):
        return  # silently skip if not present
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,  # <-- add this
    )
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    candidates = [r[0] for r in cur.fetchall()]
    table_found = None
    for t in candidates:
        try:
            cur.execute(f'PRAGMA table_info("{t}")')
            cols = {r[1] for r in cur.fetchall()}
            if {"gid", "lang", "title", "url", "text"}.issubset(cols):
                table_found = t
                break
        except sqlite3.DatabaseError:
            continue
    _SQLITE_CONN = conn if table_found else None
    _SQLITE_TABLE = table_found


def _sqlite_get_qid_text(qid: str, lang: str = "en") -> str | None:
    """Return cached text for a QID from the SQLite store or None."""
    _sqlite_connect()
    if _SQLITE_CONN is None or _SQLITE_TABLE is None:
        return None
    cur = _SQLITE_CONN.cursor()
    try:
        cur.execute(f'SELECT text FROM "{_SQLITE_TABLE}" WHERE gid=? AND lang=? LIMIT 1', (qid, lang))
        row = cur.fetchone()
        if row and isinstance(row[0], str) and row[0].strip():
            return row[0]
    except sqlite3.DatabaseError:
        return None
    return None


def _open_wiki_search_db(path: str = WIKI_SEARCH_SQLITE_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except:
        pass

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wiki_search_cache (
            key TEXT PRIMARY KEY,
            raw_json TEXT,
            created_at REAL
        )
        """
    )
    conn.commit()
    return conn


def _wiki_search_cache_key(title: str, limit: int) -> str:
    # use normalized title + limit to be safe if you ever change limit
    norm = normalize_title_for_match(title)
    return f"{norm}||{limit}"


def _wiki_search_cache_get(conn, key: str):
    cur = conn.cursor()
    cur.execute(
        "SELECT raw_json FROM wiki_search_cache WHERE key = ? LIMIT 1",
        (key,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _wiki_search_cache_put(conn, key: str, data):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO wiki_search_cache (key, raw_json, created_at)
        VALUES (?, ?, ?)
        """,
        (key, json.dumps(data), time.time()),
    )
    conn.commit()


def _open_resolve_db(path: str = RESOLVE_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sentence_cache (
            key TEXT PRIMARY KEY,
            title TEXT,
            sentence TEXT,
            resolved_page_id TEXT,
            resolved_page_title TEXT,
            match_score REAL
        )
        """
    )
    conn.commit()
    return conn


def _load_cached_resolutions(conn, keys):
    """Return dict key -> (resolved_page_id, resolved_page_title, match_score)."""
    if not keys:
        return {}

    cur = conn.cursor()
    out = {}
    BATCH = 1000
    for i in range(0, len(keys), BATCH):
        batch = keys[i: i + BATCH]
        placeholders = ",".join(["?"] * len(batch))
        cur.execute(
            f"""
            SELECT key, resolved_page_id, resolved_page_title, match_score
            FROM sentence_cache
            WHERE key IN ({placeholders})
            """,
            batch,
        )
        for k, pid, title, score in cur.fetchall():
            out[k] = (pid, title, score)

    return out


def _save_resolve_batch(conn, rows):
    """rows: iterable of (key, title, sentence, pid, ptitle, score)."""
    if not rows:
        return
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO sentence_cache
        (key, title, sentence, resolved_page_id, resolved_page_title, match_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def wiki_search(title: str, limit: int = 5):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": title,
        "srlimit": limit,
        "srprop": "",
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    if r.status_code == 403:
        time.sleep(0.8)
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    return [{"pageid": str(i["pageid"]), "title": i["title"], "local": False}
            for i in data.get("query", {}).get("search", [])]


def wiki_search_cached(title: str, limit: int, conn):
    cache_key = _wiki_search_cache_key(title, limit)
    cached = _wiki_search_cache_get(conn, cache_key)
    if cached is not None:
        return cached
    # miss -> call API
    result = wiki_search(title, limit=limit)
    _wiki_search_cache_put(conn, cache_key, result)
    return result


def wiki_get_page_plaintext(pageid: str, max_chars: int = 200000):
    logger.info(f'Loading page {pageid} via API')
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "pageids": pageid,
        "format": "json",
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    if r.status_code == 403:
        time.sleep(1)
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})
    extract = page.get("extract", "") or ""
    return extract[:max_chars]


def _sqlite_get_qid_text_bulk(qids, lang: str = "en") -> dict:
    """
    Fetch many QID → text mappings in bulk.
    Returns dict: {qid: text}.
    """
    _sqlite_connect()
    if _SQLITE_CONN is None or _SQLITE_TABLE is None:
        return {}

    qids = list({str(q) for q in qids})  # dedupe & stringify
    if not qids:
        return {}

    cur = _SQLITE_CONN.cursor()
    out = {}

    BATCH = 1000
    for i in range(0, len(qids), BATCH):
        batch = qids[i: i + BATCH]
        placeholders = ",".join(["?"] * len(batch))

        # one full-table scan per batch instead of per QID
        cur.execute(
            f'SELECT gid, text FROM "{_SQLITE_TABLE}" '
            f'WHERE lang = ? AND gid IN ({placeholders})',
            (lang, *batch),
        )
        for gid, text in cur.fetchall():
            if isinstance(text, str) and text.strip():
                out[str(gid)] = text
    return out


def load_or_page_sents(pageid: str,
                       cache_dir: str,
                       local_pages_dir: str,
                       sample_limit: int = 400,
                       qid_texts: dict | None = None):
    """
    If pageid starts with 'Q' -> use SQLite cache (preferred) or pre-fetched dict,
    then local pages/Q*.txt (fallback).
    Else -> numeric Wikipedia pageid using API + on-disk cache.
    Returns normalized sentences.
    """
    text = ""

    # QIDs → use pre-fetched dict / SQLite / local file
    if isinstance(pageid, str) and pageid.startswith("Q"):
        if qid_texts is not None and pageid in qid_texts:
            text = qid_texts[pageid]
        else:
            # fall back to per-QID SQLite + local file if really needed
            text = _sqlite_get_qid_text(pageid) or ""
            if not text:
                path = os.path.join(local_pages_dir, f"{pageid}.txt")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
    else:
        # numeric pageid → API + file cache
        os.makedirs(cache_dir, exist_ok=True)
        page_cache = os.path.join(cache_dir, f"page_{pageid}.txt")
        if os.path.exists(page_cache):
            with open(page_cache, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            text = wiki_get_page_plaintext(pageid)
            with open(page_cache, "w", encoding="utf-8") as f:
                f.write(text)
            time.sleep(0.05)

    if not text:
        return []

    sents = split_into_sentences(text)
    if len(sents) > sample_limit:
        sents = sents[:sample_limit]

    return [normalize_text(s) for s in sents]


def _mp_init(page_sents, title_candidates, min_sentence_score):
    global G_PAGE_SENTS, G_TITLE_CANDIDATES, G_MIN_SENTENCE_SCORE
    G_PAGE_SENTS = page_sents
    G_TITLE_CANDIDATES = title_candidates
    G_MIN_SENTENCE_SCORE = min_sentence_score


def _resolve_one(args):
    """Sentence to Wikipedia page resolver (fast, using RapidFuzz)."""
    idx, title, sentence = args
    sent_norm = normalize_text(sentence)
    if not sent_norm:
        return idx, None, None, 0.0

    cands = G_TITLE_CANDIDATES.get(title, [])
    if not cands:
        return idx, None, None, 0.0

    best_pid = None
    best_title = None
    best_score = 0.0  # still in 0–1 range

    for cand in cands:
        pid = str(cand["pageid"])
        page_sents = G_PAGE_SENTS.get(pid)
        if not page_sents:
            continue

        # Find best matching sentence on this page in *C++*.
        # result: (matched_sentence, score_0_100, index)
        match = process.extractOne(
            sent_norm,
            page_sents,
            scorer=fuzz.token_set_ratio,
        )
        if match is None:
            continue

        _, score_0_100, _ = match
        local_best = score_0_100 / 100.0

        if local_best > best_score:
            best_score = local_best
            best_pid = pid
            best_title = cand.get("title", pid)

            if best_score > 0.98:
                break

    if best_pid is not None and best_score >= G_MIN_SENTENCE_SCORE:
        return idx, best_pid, best_title, best_score
    else:
        return idx, None, None, best_score


def attach_pageids(
        df,
        title_col="title",
        sentence_col="sentence",
        min_sentences_per_title=3,
        min_sentences_per_page=3,
        search_limit=5,
        min_sentence_score=0.7,
        local_pages_dir=LOCAL_PAGES_DIR,
        qid_index_path=QID_INDEX_PATH,
        cache_dir=CACHE_DIR,
):
    df = df.copy()

    # Filtering out less than min_sentences_per_title for unique titles set in LAGRANGE
    counts = df[title_col].value_counts()
    good_titles = set(counts[counts >= min_sentences_per_title].index)
    df = df[df[title_col].isin(good_titles)].reset_index(drop=True)

    # Trying to read subject/object wikidata qids after matching with T-Rex triplets (Optional; only if tried trex_resolver.py before).
    if qid_index_path:
        if not os.path.exists(qid_index_path):
            raise RuntimeError(f"Missing {qid_index_path}. Run build_local_index.py first or set qid_index_path=None.")
        qidx = pd.read_parquet(qid_index_path)
        qidx = qidx.dropna(subset=["en_label"])
        local_title_to_qids = qidx.groupby("en_label")["qid"].apply(list).to_dict()
    else:
        local_title_to_qids = dict()

    # Find top-5 (search_limit parameter) WikiData candidates for entity linking of each LAGRANGE['title']
    title_candidates = {}
    unique_titles = df[title_col].unique()

    search_conn = _open_wiki_search_db(WIKI_SEARCH_SQLITE_PATH)
    for title in tqdm(unique_titles, desc="[build candidates]"):
        norm_t = normalize_title_for_match(title)
        local_qids = local_title_to_qids.get(norm_t, [])
        local_cands = [{"pageid": qid, "title": title, "local": True} for qid in local_qids]

        # Get top-5 candidates
        remote_cands = wiki_search_cached(title, limit=search_limit, conn=search_conn)

        seen = set()
        all_cands = []
        for c in (local_cands + remote_cands):
            pid = str(c["pageid"])
            if pid not in seen:
                seen.add(pid)
                all_cands.append({**c, "pageid": pid})
        title_candidates[title] = all_cands

    search_conn.close()

    # Get wikipedia pages associated with each candidate qid
    candidate_pageids = {c["pageid"] for candlist in title_candidates.values() for c in candlist}

    page_sents = {}

    def _load_one_page(pid: str):
        try:
            sents = load_or_page_sents(pid, cache_dir, local_pages_dir)
            return pid, sents
        except Exception:
            return pid, []

    max_workers = min(64, max(4, (cpu_count() or 16)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_load_one_page, pid): pid
            for pid in candidate_pageids
        }
        for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[load pages]",
        ):
            pid, sents = fut.result()
            if sents:
                page_sents[pid] = sents

    df["resolved_page_id"] = None
    df["resolved_page_title"] = None
    df["match_score"] = 0.0
    df["_cache_key"] = [
        sentence_cache_key(t, s)
        for t, s in zip(df[title_col].astype(str), df[sentence_col].astype(str))
    ]

    # # Trying to load previously computed resolutions, if any
    conn = _open_resolve_db(RESOLVE_CACHE_PATH)
    all_keys = df["_cache_key"].tolist()

    mask_unresolved = df["resolved_page_id"].isna()
    tasks = [
        (i, row[title_col], row[sentence_col])
        for i, row in df[mask_unresolved].iterrows()
    ]

    # We'll resolve / map sentence-to-actual-page in multithread with periodic DB writes
    n_proc = min(16, max(1, (cpu_count() or 4) - 1))
    BATCH_SAVE = 1000
    pending_rows = []

    with Pool(processes=n_proc, initializer=_mp_init,
              initargs=(page_sents, title_candidates, min_sentence_score)) as pool:
        for idx, pid, rtitle, score in tqdm(
                pool.imap_unordered(_resolve_one, tasks, chunksize=64),
                total=len(tasks),
                desc="[resolve sentences]",
        ):
            df.at[idx, "resolved_page_id"] = pid
            df.at[idx, "resolved_page_title"] = rtitle
            df.at[idx, "match_score"] = score

            # add to cache batch
            key = df.at[idx, "_cache_key"]
            title = df.at[idx, title_col]
            sent = df.at[idx, sentence_col]
            pending_rows.append((key, title, sent, pid, rtitle, float(score)))

            if len(pending_rows) >= BATCH_SAVE:
                _save_resolve_batch(conn, pending_rows)
                pending_rows.clear()

    if pending_rows:
        _save_resolve_batch(conn, pending_rows)
        pending_rows.clear()

    conn.close()

    # CHANGED: drop helper column
    df = df.drop(columns=["_cache_key"])

    # same post-filtering as before
    df = df[df["resolved_page_id"].notnull()].copy()
    page_counts = df["resolved_page_id"].value_counts()
    keep_pages = set(page_counts[page_counts >= min_sentences_per_page].index)
    df = df[df["resolved_page_id"].isin(keep_pages)].copy()

    return df


def load_jsonl(path):
    with jsonlines.open(path) as reader:
        return [obj for obj in reader]


if __name__ == "__main__":

    train_filtered = load_jsonl("data/lagrange/lagrange_train_filtered.json")
    test_filtered = load_jsonl("data/lagrange/lagrange_test_filtered.json")

    df_train = pd.DataFrame(train_filtered)
    df_test = pd.DataFrame(test_filtered)

    df_train["split"] = "train"
    df_test["split"] = "test"

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    logger.info(f'{df_all.shape = }')

    cleaned_all = attach_pageids(
        df_all,
        title_col="title",
        sentence_col="sentence",
        min_sentences_per_title=3,
        min_sentences_per_page=3,
        search_limit=5,
        min_sentence_score=0.5,
        local_pages_dir=LOCAL_PAGES_DIR,
        qid_index_path=QID_INDEX_PATH,
        cache_dir=CACHE_DIR,
    )
    logger.info(f'{cleaned_all.shape = }')

    cleaned_train = cleaned_all[cleaned_all["split"] == "train"].drop(columns=["split"])
    cleaned_test = cleaned_all[cleaned_all["split"] == "test"].drop(columns=["split"])

    cleaned_train.to_json(
        "data/lagrange/lagrange_train_resolved.json",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    cleaned_test.to_json(
        "data/lagrange/lagrange_test_resolved.json",
        orient="records",
        lines=True,
        force_ascii=False,
    )
