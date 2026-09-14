import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

DEFAULT_MODULE_PATH = "src/data/wikidockg/wikontic_local.py"
DEFAULT_TABLE_NAME = "wikontic_analysis"


def load_wikontic_module(module_path: str):
    spec = importlib.util.spec_from_file_location("wikontic_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["wikontic_local"] = mod
    spec.loader.exec_module(mod)
    return mod


def connect_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_analysis_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            page_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            url TEXT NOT NULL,
            paragraph_count_total INTEGER NOT NULL,
            paragraph_count_used INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            raw_output_json TEXT,
            triples_json TEXT,
            usage_json TEXT,
            triple_count INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(page_id) REFERENCES sampled_articles(page_id)
        )
    """)

    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_status
        ON {table_name}(status)
    """)

    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_processed_at
        ON {table_name}(processed_at)
    """)

    conn.commit()


def _json_fallback(obj: Any):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return repr(obj)


def json_dumps_safe(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=_json_fallback)


def normalize_output(output: Any) -> Tuple[Any, Any, Any]:
    """
    Returns (raw_output, triples, usage).
    """
    raw_output = output
    triples = None
    usage = None

    if isinstance(output, tuple):
        if len(output) >= 1:
            triples = output[0]
        if len(output) >= 2:
            usage = output[1]
    elif isinstance(output, dict):
        if "triplets" in output:
            triples = output.get("triplets")
            usage = output.get("usage")
        elif "usage" in output:
            usage = output.get("usage")
        else:
            triples = output
    else:
        triples = output

    return raw_output, triples, usage


def infer_triple_count(triples: Any) -> Optional[int]:
    if triples is None:
        return None

    if isinstance(triples, list):
        return len(triples)

    if isinstance(triples, dict):
        if isinstance(triples.get("triples"), list):
            return len(triples["triples"])
        if isinstance(triples.get("results"), list):
            return len(triples["results"])
        return None

    return None


def collect_paragraphs(text: str, remove_categories: bool):
    paragraphs = [par for par in text.splitlines() if par.strip()]

    if paragraphs and remove_categories:
        if "Category:" in paragraphs[-1]:
            return paragraphs[:-1]

    return paragraphs


def build_analysis_input(text: str, max_paragraphs: int) -> Tuple[str, int]:
    paragraphs = collect_paragraphs(text, remove_categories=True)
    selected = paragraphs[:max_paragraphs]
    return "\n".join(selected), len(selected)


def select_articles(
        conn: sqlite3.Connection,
        table_name: str,
        refresh: bool,
        limit: Optional[int],
) -> list[sqlite3.Row]:
    sql = """
        SELECT sa.page_id, sa.title, sa.created_at, sa.paragraph_count, sa.url, sa.text
        FROM sampled_articles sa
    """

    if not refresh:
        sql += f"""
            LEFT JOIN {table_name} wa
                ON wa.page_id = sa.page_id
            WHERE wa.page_id IS NULL
        """

    sql += " ORDER BY sa.created_at DESC, sa.page_id ASC"

    if limit is not None:
        sql += " LIMIT ?"
        return list(conn.execute(sql, (limit,)))

    return list(conn.execute(sql))


def save_success(
        conn: sqlite3.Connection,
        table_name: str,
        row: sqlite3.Row,
        input_text: str,
        paragraphs_used: int,
        raw_output: Any,
        triples: Any,
        usage: Any,
) -> None:
    conn.execute(f"""
        INSERT INTO {table_name} (
            page_id,
            title,
            created_at,
            url,
            paragraph_count_total,
            paragraph_count_used,
            input_text,
            raw_output_json,
            triples_json,
            usage_json,
            triple_count,
            status,
            error,
            processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', NULL, CURRENT_TIMESTAMP)
        ON CONFLICT(page_id) DO UPDATE SET
            title=excluded.title,
            created_at=excluded.created_at,
            url=excluded.url,
            paragraph_count_total=excluded.paragraph_count_total,
            paragraph_count_used=excluded.paragraph_count_used,
            input_text=excluded.input_text,
            raw_output_json=excluded.raw_output_json,
            triples_json=excluded.triples_json,
            usage_json=excluded.usage_json,
            triple_count=excluded.triple_count,
            status='ok',
            error=NULL,
            processed_at=CURRENT_TIMESTAMP
    """, (
        row["page_id"],
        row["title"],
        row["created_at"],
        row["url"],
        row["paragraph_count"],
        paragraphs_used,
        input_text,
        json_dumps_safe(raw_output),
        json_dumps_safe(triples),
        json_dumps_safe(usage),
        infer_triple_count(triples),
    ))


def save_error(
        conn: sqlite3.Connection,
        table_name: str,
        row: sqlite3.Row,
        input_text: str,
        paragraphs_used: int,
        error_message: str,
) -> None:
    conn.execute(f"""
        INSERT INTO {table_name} (
            page_id,
            title,
            created_at,
            url,
            paragraph_count_total,
            paragraph_count_used,
            input_text,
            raw_output_json,
            triples_json,
            usage_json,
            triple_count,
            status,
            error,
            processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 'error', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(page_id) DO UPDATE SET
            title=excluded.title,
            created_at=excluded.created_at,
            url=excluded.url,
            paragraph_count_total=excluded.paragraph_count_total,
            paragraph_count_used=excluded.paragraph_count_used,
            input_text=excluded.input_text,
            raw_output_json=NULL,
            triples_json=NULL,
            usage_json=NULL,
            triple_count=NULL,
            status='error',
            error=excluded.error,
            processed_at=CURRENT_TIMESTAMP
    """, (
        row["page_id"],
        row["title"],
        row["created_at"],
        row["url"],
        row["paragraph_count"],
        paragraphs_used,
        input_text,
        error_message,
    ))


def populate_with_wikontic(
        db_path: str,
        module_path: str,
        table_name: str,
        max_paragraphs: int,
        do_align: bool,
        do_refine: bool,
        refresh: bool,
        limit: Optional[int],
        sleep_seconds: float,
        commit_every: int,
) -> None:
    mod = load_wikontic_module(module_path)
    conn = connect_db(db_path)
    ensure_analysis_table(conn, table_name)

    missing_input = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sampled_articles'"
    ).fetchone()
    if missing_input is None:
        raise SystemExit("sampled_articles table does not exist in the target DB")

    rows = select_articles(conn, table_name=table_name, refresh=refresh, limit=limit)
    total = len(rows)
    print(f"articles_to_process={total}")

    ok = 0
    errors = 0
    since_commit = 0

    for idx, row in enumerate(rows, start=1):
        input_text, paragraphs_used = build_analysis_input(
            row["text"],
            max_paragraphs=max_paragraphs,
        )

        if not input_text:
            save_error(
                conn,
                table_name=table_name,
                row=row,
                input_text="",
                paragraphs_used=0,
                error_message="empty input after paragraph filtering",
            )
            errors += 1
            since_commit += 1
        else:
            try:
                output = mod.extract_triples(
                    input_text,
                    do_align=do_align,
                    do_refine=do_refine,
                    return_usage=True,
                )
                raw_output, triples, usage = normalize_output(output)
                save_success(
                    conn,
                    table_name=table_name,
                    row=row,
                    input_text=input_text,
                    paragraphs_used=paragraphs_used,
                    raw_output=raw_output,
                    triples=triples,
                    usage=usage,
                )
                ok += 1
                since_commit += 1
            except Exception as exc:
                save_error(
                    conn,
                    table_name=table_name,
                    row=row,
                    input_text=input_text,
                    paragraphs_used=paragraphs_used,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                errors += 1
                since_commit += 1

        if since_commit >= commit_every:
            conn.commit()
            since_commit = 0

        print(
            f"[{idx}/{total}] page_id={row['page_id']} title={row['title']!r} ok={ok} errors={errors}",
            flush=True,
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if since_commit:
        conn.commit()

    conn.close()
    print(f"done ok={ok} errors={errors} table={table_name} db={db_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the sampler SQLite DB with Wikontic analysis outputs."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite DB created by the sampler script",
    )
    parser.add_argument(
        "--module-path",
        default=DEFAULT_MODULE_PATH,
        help="Path to wikontic_local.py",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help="Destination table name",
    )
    parser.add_argument(
        "--max-paragraphs",
        type=int,
        default=10,
        help="Maximum paragraphs passed to extract_triples",
    )
    parser.add_argument(
        "--do-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass do_align to align with wikidata",
    )
    parser.add_argument(
        "--do-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass do_refine to normalize wikidata entities",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute rows even if already present in analysis table",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on rows processed",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between articles",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=1,
        help="Commit every N processed rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    populate_with_wikontic(
        db_path=args.db,
        module_path=args.module_path,
        table_name=args.table,
        max_paragraphs=args.max_paragraphs,
        do_align=args.do_align,
        do_refine=args.do_refine,
        refresh=args.refresh,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()
