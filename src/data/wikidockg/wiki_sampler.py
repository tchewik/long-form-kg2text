#!/usr/bin/env python3

import argparse
import bz2
import json
import random
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm


API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WikiHybridSampler/2.0 (your_email@example.com)"
}

MIN_PARAGRAPHS = 4
MAX_PARAGRAPHS = 30
MIN_WORDS_PER_PARAGRAPH = 8


def safe_request(params, max_retries=5):
    delay = 2

    for _ in range(max_retries):
        try:
            r = requests.get(
                API_URL,
                params=params,
                headers=HEADERS,
                timeout=(5, 30),
            )
        except requests.exceptions.RequestException as e:
            print(f"request error: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if r.status_code == 200:
            return r

        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", delay))
            print(f"rate limited, waiting {wait}s")
            if wait > 120:
                print("Retry-After too large, aborting cleanly.")
                return None
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        print(f"unexpected status: {r.status_code}")
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)

        time.sleep(delay)
        delay = min(delay * 2, 60)

    return None


def build_url(title):
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def strip_tag(tag):
    return tag.rsplit("}", 1)[-1]


def init_db(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS created_pages (
            page_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cutoff_iso TEXT,
            lecontinue TEXT,
            finished INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sampled_articles (
            page_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            paragraph_count INTEGER NOT NULL,
            url TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def save_created_page(conn, page_id, title, created_at):
    conn.execute("""
        INSERT OR REPLACE INTO created_pages (page_id, title, created_at)
        VALUES (?, ?, ?)
    """, (page_id, title, created_at))


def get_index_state(conn):
    row = conn.execute("""
        SELECT cutoff_iso, lecontinue, finished
        FROM index_state
        WHERE id = 1
    """).fetchone()

    if not row:
        return None

    return {
        "cutoff_iso": row[0],
        "lecontinue": row[1],
        "finished": bool(row[2]),
    }


def set_index_state(conn, cutoff_iso, lecontinue, finished):
    conn.execute("""
        INSERT INTO index_state (id, cutoff_iso, lecontinue, finished, updated_at)
        VALUES (1, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            cutoff_iso=excluded.cutoff_iso,
            lecontinue=excluded.lecontinue,
            finished=excluded.finished,
            updated_at=CURRENT_TIMESTAMP
    """, (cutoff_iso, lecontinue, int(finished)))
    conn.commit()


def count_created_pages(conn):
    row = conn.execute("SELECT COUNT(*) FROM created_pages").fetchone()
    return row[0]


def load_created_pages_map(conn):
    cur = conn.execute("SELECT page_id, title, created_at FROM created_pages")
    return {
        row[0]: {"title": row[1], "created_at": row[2]}
        for row in cur
    }


def clear_sampled_articles(conn):
    conn.execute("DELETE FROM sampled_articles")
    conn.commit()


def save_sampled_article(conn, article):
    conn.execute("""
        INSERT OR REPLACE INTO sampled_articles
        (page_id, title, created_at, paragraph_count, url, text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        article["page_id"],
        article["title"],
        article["created_at"],
        article["paragraph_count"],
        article["url"],
        article["text"],
    ))


def fetch_creation_events_batch(cutoff_iso, lecontinue=None, lelimit="max"):
    params = {
        "action": "query",
        "format": "json",
        "list": "logevents",
        "letype": "create",
        "lenamespace": 0,
        "leprop": "ids|title|timestamp",
        "ledir": "newer",
        "lestart": cutoff_iso,
        "lelimit": lelimit,
    }

    if lecontinue:
        params["lecontinue"] = lecontinue

    r = safe_request(params)
    if not r:
        return None, None

    data = r.json()

    if "error" in data:
        print("API ERROR:", data["error"])
        return None, None

    events = data.get("query", {}).get("logevents", [])
    next_lecontinue = data.get("continue", {}).get("lecontinue")

    return events, next_lecontinue


def build_creation_index(db_path, cutoff_iso, pause_seconds=1.0, resume=True):
    conn = init_db(db_path)

    state = get_index_state(conn)
    lecontinue = None

    if state and resume:
        if state["cutoff_iso"] != cutoff_iso:
            raise SystemExit(
                f"Existing index cutoff ({state['cutoff_iso']}) does not match requested cutoff ({cutoff_iso}). "
                f"Use a new DB path or delete/reset the DB."
            )

        if state["finished"]:
            print(f"Index already complete. created_pages={count_created_pages(conn)}")
            conn.close()
            return

        lecontinue = state["lecontinue"]
        print(f"Resuming index build from lecontinue={lecontinue!r}")
    else:
        set_index_state(conn, cutoff_iso=cutoff_iso, lecontinue=None, finished=False)

    total_before = count_created_pages(conn)
    print(f"Starting/continuing creation-index build. Existing rows: {total_before}")

    while True:
        events, next_lecontinue = fetch_creation_events_batch(
            cutoff_iso=cutoff_iso,
            lecontinue=lecontinue,
        )

        if events is None:
            print("Stopping due to API failure. You can resume later.")
            break

        added = 0
        with conn:
            for ev in events:
                page_id = ev.get("pageid")
                title = ev.get("title")
                created_at = ev.get("timestamp")

                if page_id is None or not title or not created_at:
                    continue

                save_created_page(conn, int(page_id), title, created_at)
                added += 1

        set_index_state(conn, cutoff_iso=cutoff_iso, lecontinue=next_lecontinue, finished=False)

        total_now = count_created_pages(conn)
        print(f"batch_added={added} total_created_pages={total_now} next_lecontinue={next_lecontinue!r}")

        if not next_lecontinue:
            set_index_state(conn, cutoff_iso=cutoff_iso, lecontinue=None, finished=True)
            print(f"Index build finished. Total created pages: {total_now}")
            break

        lecontinue = next_lecontinue
        time.sleep(pause_seconds)

    conn.close()


def remove_templates(text):
    """ Remove any remaining {{...}} templates, including nested ones. """
    out = []
    i = 0
    depth = 0
    n = len(text)

    while i < n:
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
            continue
        if text[i:i+2] == "}}":
            if depth > 0:
                depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1

    return "".join(out)


def split_template_parts(body):
    parts = []
    buf = []
    link_depth = 0
    i = 0
    n = len(body)

    while i < n:
        if body[i:i+2] == "[[":
            link_depth += 1
            buf.append("[[")
            i += 2
            continue
        if body[i:i+2] == "]]":
            if link_depth > 0:
                link_depth -= 1
            buf.append("]]")
            i += 2
            continue
        if body[i] == "|" and link_depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue

        buf.append(body[i])
        i += 1

    parts.append("".join(buf))
    return parts


def normalize_template_name(name):
    name = name.strip().lower().replace("_", " ")
    if ":" in name:
        name = name.split(":", 1)[1]
    return re.sub(r"\s+", " ", name).strip()


def render_inline_template(body):
    parts = [p.strip() for p in split_template_parts(body)]
    if not parts:
        return None

    name = normalize_template_name(parts[0])

    positional = []
    for part in parts[1:]:
        if "=" not in part and part:
            positional.append(part.strip())

    if name == "lang" or name.startswith("lang-"):
        return positional[-1] if positional else ""

    if name in {
        "transliteration",
        "translit",
        "ipa",
        "ipa-ko",
        "ko-ipa",
        "pron",
        "pronunciation",
        "hanja",
        "hangul",
        "rr",
        "mr",
    }:
        return positional[-1] if positional else ""

    if name in {"nowrap", "nobr", "small", "smaller", "sup", "sub"}:
        return " ".join(positional).strip()

    return None


def unwrap_inline_templates(text):
    pattern = re.compile(r"\{\{([^{}]*)\}\}")

    while True:
        changed = False

        def repl(match):
            nonlocal changed
            replacement = render_inline_template(match.group(1))
            if replacement is None:
                return match.group(0)
            changed = True
            return replacement

        new_text = pattern.sub(repl, text)
        if not changed:
            return text
        text = new_text


def strip_file_links(text):
    """
    Remove [[File:...]] and [[Image:...]] blocks entirely,
    including thumb|... captions.
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        if text[i:i+2] == "[[":
            prefix = text[i+2:i+18].lower()
            if prefix.startswith("file:") or prefix.startswith("image:"):
                depth = 1
                j = i + 2
                while j < n and depth > 0:
                    if text[j:j+2] == "[[":
                        depth += 1
                        j += 2
                    elif text[j:j+2] == "]]":
                        depth -= 1
                        j += 2
                    else:
                        j += 1
                i = j
                continue

        result.append(text[i])
        i += 1

    return "".join(result)


def clean_wikitext(text):
    if not text:
        return ""

    text = re.sub(r"__[^_\s]+__", " ", text)   # removes __NOTOC__ and similar magic words
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)

    text = re.sub(r"<ref\b[^>/]*>.*?</ref>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text, flags=re.S)

    text = strip_file_links(text)
    text = re.sub(r"\[\[(?:Category|Help|Special):[^\]]+\]\]", " ", text, flags=re.I)

    text = re.sub(r"\{\|.*?\|\}", "\n", text, flags=re.S)

    text = unwrap_inline_templates(text)
    text = remove_templates(text)

    text = re.sub(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]", r"\2", text)
    text = re.sub(r"\[(https?://[^\]]+)\]", " ", text)

    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"\[\d+\]", " ", text)  # removes citation markers like [1], [23]

    text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", "\n\n", text, flags=re.M)

    text = text.replace("'''", "").replace("''", "")
    text = (
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
    )

    text = re.sub(r"\(\s*([,;:/\-–—]|\s)*\)", " ", text)
    text = re.sub(r"\(\s*;\s*", "(", text)
    text = re.sub(r";\s*\)", ")", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            lines.append("")
            continue

        if line.startswith(("*", "#", ";", ":")):
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_paragraphs(cleaned_text):
    paragraphs = []

    for p in cleaned_text.split("\n\n"):
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if len(p.split()) < MIN_WORDS_PER_PARAGRAPH:
            continue
        paragraphs.append(p)

    return paragraphs


def parse_page(elem):
    page = {
        "title": None,
        "ns": None,
        "id": None,
        "redirect": False,
        "text": "",
    }

    revision_seen = False

    for child in elem:
        name = strip_tag(child.tag)

        if name == "title":
            page["title"] = child.text or ""

        elif name == "ns":
            page["ns"] = child.text or ""

        elif name == "id" and not revision_seen:
            page["id"] = child.text or ""

        elif name == "redirect":
            page["redirect"] = True

        elif name == "revision":
            revision_seen = True
            for rev_child in child:
                if strip_tag(rev_child.tag) == "text":
                    page["text"] = rev_child.text or ""

    return page


class Reservoir:
    def __init__(self, size, rng):
        self.size = size
        self.rng = rng
        self.items = []
        self.seen = 0

    def add(self, item):
        self.seen += 1

        if len(self.items) < self.size:
            self.items.append(item)
            return

        j = self.rng.randint(1, self.seen)
        if j <= self.size:
            self.items[j - 1] = item


def sample_from_dump(
    dump_path,
    db_path,
    target=1500,
    random_seed=42,
    thinning_prob=0.5,
    output_json=None,
    output_stats=None,
):
    conn = init_db(db_path)
    state = get_index_state(conn)

    if not state or not state["finished"]:
        raise SystemExit("Creation index is not complete. Run build-index first.")

    created_pages = load_created_pages_map(conn)
    created_ids = set(created_pages.keys())

    print(f"Loaded creation index: {len(created_ids)} page IDs")

    rng = random.Random(random_seed)
    reservoir = Reservoir(target, rng)

    stats = {
        "dump_path": dump_path,
        "target": target,
        "random_seed": random_seed,
        "thinning_prob": thinning_prob,
        "created_ids_loaded": len(created_ids),
        "pages_seen": 0,
        "namespace0_nonredirect_seen": 0,
        "id_matched": 0,
        "paragraph_qualified": 0,
        "reservoir_seen_qualified": 0,
        "final_sample_size": 0,
    }

    pbar = tqdm(total=target, desc="Reservoir fill")

    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for _, elem in context:
            if strip_tag(elem.tag) != "page":
                continue

            stats["pages_seen"] += 1
            page = parse_page(elem)

            if page["ns"] != "0" or page["redirect"] or not page["text"] or not page["id"]:
                elem.clear()
                continue

            stats["namespace0_nonredirect_seen"] += 1

            page_id = int(page["id"])
            meta = created_pages.get(page_id)

            if not meta:
                elem.clear()
                continue

            stats["id_matched"] += 1

            if rng.random() > thinning_prob:
                elem.clear()
                continue

            cleaned = clean_wikitext(page["text"])
            paragraphs = extract_paragraphs(cleaned)

            if not (MIN_PARAGRAPHS <= len(paragraphs) <= MAX_PARAGRAPHS):
                elem.clear()
                continue

            stats["paragraph_qualified"] += 1
            stats["reservoir_seen_qualified"] += 1

            article = {
                "page_id": page_id,
                "title": page["title"],
                "created_at": meta["created_at"],
                "paragraph_count": len(paragraphs),
                "url": build_url(page["title"]),
                "text": "\n\n".join(paragraphs),
            }

            reservoir.add(article)

            pbar.n = min(len(reservoir.items), target)
            pbar.set_postfix({
                "seen": stats["pages_seen"],
                "matched": stats["id_matched"],
                "qualified": stats["paragraph_qualified"],
                "sampled": len(reservoir.items),
                "last": page["title"][:22],
            })
            pbar.refresh()

            elem.clear()

    pbar.close()

    clear_sampled_articles(conn)
    with conn:
        for article in reservoir.items:
            save_sampled_article(conn, article)

    stats["final_sample_size"] = len(reservoir.items)

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                sorted(reservoir.items, key=lambda x: x["created_at"], reverse=True),
                f,
                ensure_ascii=False,
                indent=2,
            )

    if output_stats:
        Path(output_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(output_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    conn.close()

    print()
    print(f"Final sample size: {len(reservoir.items)}")
    if output_json:
        print(f"JSON:  {output_json}")
    if output_stats:
        print(f"Stats: {output_stats}")
    print(f"DB:    {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build cached Wikipedia creation index, then sample matching pages from a dump."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("build-index", help="Fetch and cache page creation metadata via API")
    p_index.add_argument("--db", required=True, help="SQLite DB path")
    p_index.add_argument("--cutoff", default="2026-02-01T00:00:00Z", help="Creation cutoff timestamp, UTC ISO format")
    p_index.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between API pages")
    p_index.add_argument("--no-resume", action="store_true", help="Do not resume from saved lecontinue")

    p_sample = subparsers.add_parser("sample-dump", help="Stream dump and reservoir-sample pages present in cached index")
    p_sample.add_argument("--dump", required=True, help="Path to Wikipedia .xml.bz2 dump")
    p_sample.add_argument("--db", required=True, help="SQLite DB path with completed created_pages index")
    p_sample.add_argument("--target", type=int, default=1500, help="Final sample size")
    p_sample.add_argument("--random-seed", type=int, default=42, help="Random seed")
    p_sample.add_argument("--thinning-prob", type=float, default=1.0, help="Optional thinning after ID match")
    p_sample.add_argument("--output-json", default="data/wikidockg/sampled_articles.json", help="Output JSON path")
    p_sample.add_argument("--output-stats", default="data/wikidockg/run_stats.json", help="Output stats JSON path")

    args = parser.parse_args()

    if args.command == "build-index":
        build_creation_index(
            db_path=args.db,
            cutoff_iso=args.cutoff,
            pause_seconds=args.pause_seconds,
            resume=not args.no_resume,
        )
    elif args.command == "sample-dump":
        sample_from_dump(
            dump_path=args.dump,
            db_path=args.db,
            target=args.target,
            random_seed=args.random_seed,
            thinning_prob=args.thinning_prob,
            output_json=args.output_json,
            output_stats=args.output_stats,
        )


if __name__ == "__main__":
    main()
