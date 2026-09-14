# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv
import httpx
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import torch

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

os.environ["WIKONTIC_DO_REFINE"] = os.getenv("WIKONTIC_DO_REFINE", "true")
os.environ["WIKONTIC_CANDIDATE_BLEU_THRESHOLD"] = os.getenv("WIKONTIC_CANDIDATE_BLEU_THRESHOLD", "0.5")

LOGGER = logging.getLogger("wikontic_local_batched")

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ"])')
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_QID_RE = re.compile(r"^Q\d+$", re.I)
_PID_RE = re.compile(r"^P\d+$", re.I)
_DATEISH_RE = re.compile(
    r"""^(
        \d{4}([-/]\d{1,2}([-/]\d{1,2})?)? |
        \d{1,2}[./-]\d{1,2}[./-]\d{2,4} |
        \d{1,2}\s+[A-Za-z]+\s+\d{4} |
        [A-Za-z]+\s+\d{1,2},\s+\d{4}
    )$""",
    re.X,
)
_NUMERICISH_RE = re.compile(r"^[+-]?(\d+([.,]\d+)?)(\s*[%°]\s*)?$")

TRIPLET_EXTRACTION_PROMPT = r"""
You are an algorithm designed to extract structured knowledge from texts to build a Wikidata-like knowledge graph. A knowledge graph consists of triplets in the format (subject, relation, object), where:

- Subject: A named entity or a concept that describes a group of people, events, or any abstract objects that serves as the source of the relation.
- Relation: A Wikidata-style predicate that connects the subject and object. If unsure, use the closest valid Wikidata property rather than inventing a relation.
- Object: A named entity or a concept that describes a group of people, events, or any abstract objects that is related to the subject.

Additionally, some triplets may have qualifiers that provide more context (e.g., date, place, or other attributes). Qualifiers should have relations and object like triplets do, but instead of subject their relation connects an object and the triplet qualifier belongs to. Qualifiers must always be attached to a triplet and never exist as standalone triplets.

You will receive a text labeled "Text:". Your task is to extract meaningful triplets that represent factual relationships.

Return only triplets in JSON format as a list of dictionaries, where each dictionary contains:
- "subject"
- "relation"
- "object"
- "qualifiers": list of {"relation","object"}
- "subject_type"
- "object_type"

NO additional text. ONLY JSON.

<example>
{
 "triplets": [
  {
   "subject": "Marie Curie",
   "relation": "award received",
   "object": "Nobel Prize in Physics",
   "qualifiers": [{"relation": "point in time", "object": "1903"}],
   "subject_type": "human",
   "object_type": "award"
  }
 ]
}
</example>
""".strip()

RANK_SUBJECT_NAMES_PROMPT = r"""
In the previous step, there was extracted a triplet akin to one in Wikidata knowledge graph from the text.
Triplet contains two entities (subject and object) and one relation that connects these subject and object.
Using semantic similarity, we linked subject name with top similar exact names from the knowledge graph built from previously seen texts.

You will be provided with:
Text, Extracted Triplet, Original Subject, Candidate Subjects.

Task:
Select the most contextually appropriate subject name from Candidate Subjects.

- Return the corresponding name exactly as it appears in the list.
- If no suitable match exists, return "None".
- No explanations.
""".strip()

RANK_OBJECT_NAMES_PROMPT = r"""
In the previous step, there was extracted a triplet akin to one in Wikidata knowledge graph from the text.
Triplet contains two entities (subject and object) and one relation that connects these subject and object.
Using semantic similarity, we linked object name with top similar exact names from the knowledge graph built from previously seen texts.

You will be provided with:
Text, Extracted Triplet, Original Object, Candidate Objects.

Task:
Select the most contextually appropriate object name from Candidate Objects.

- If the original object represents date, number or quantity, return EXACTLY the original object name.
- If a suitable match exists, return the matching candidate exactly as it appears.
- If no suitable match exists, return "None".
- No explanations.
""".strip()

RANK_RELATION_NAMES_PROMPT = r"""
In the previous step, there was extracted a triplet akin to one in Wikidata knowledge graph from the text.
Triplet contains two entities (subject and object) and one relation that connects these subject and object.
Using semantic similarity, we linked relation name with top similar exact names from the knowledge graph built from previously seen texts.

You will be provided with:
Text, Extracted Triplet, Original relation, Candidate relations.

Task:
Select the most contextually appropriate relation name from Candidate relations.

- Return the matching candidate exactly as it appears.
- If no suitable match exists, return "None".
- No explanations.
""".strip()


def configure_logging(level: int = logging.INFO) -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def set_verbose(enabled: bool = True) -> None:
    configure_logging(logging.INFO if enabled else logging.WARNING)


configure_logging(logging.INFO if os.getenv("WIKONTIC_VERBOSE", "1") == "1" else logging.WARNING)


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_lang(text: str) -> str:
    return "en"


def sha256_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_string(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("\\u"):
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    return s


def is_none_string(s: str) -> bool:
    return re.sub(r"\W+", "", s or "").lower() == "none"


def json_extract_loose(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    patterns = [
        r"```json\s*(\{.*?\}|\[.*?\])\s*```",
        r"(\{.*?\}|\[.*?\])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                continue
    return text


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def chunk_text_sentence_aware(
        text: str,
        max_chars: int = 3500,
        overlap_sentences: int = 1,
) -> List[str]:
    text = normalize_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []

    for p in paragraphs:
        ps = split_sentences(p)
        if ps:
            sentences.extend(ps)
        else:
            sentences.append(p)

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + (1 if current else 0)

        if current and current_len + sent_len > max_chars:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap[:]
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

        if len(sent) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i:i + max_chars])
            continue

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def is_date_like(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if _DATEISH_RE.match(text):
        return True
    if re.fullmatch(r"\d{4}", text):
        return True
    return False


def is_numeric_like(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    return bool(_NUMERICISH_RE.match(text))


def should_skip_item_alignment(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    if _QID_RE.fullmatch(text):
        return False
    return is_date_like(text) or is_numeric_like(text)


def normalize_cache_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def iter_chunks(items: Iterable[str], size: int) -> Iterable[List[str]]:
    buf = []
    for x in items:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self, model: Optional[str] = None) -> Dict[str, Any]:
        out = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }
        if model is not None:
            out["model"] = model
        return out


class LLMClient:
    MODEL_PRICES = {
        "gpt-4o": {"input": 2.5e-6, "output": 10e-6},
        "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
        "gpt-4.1-mini": {"input": 0.4e-6, "output": 1.6e-6},
        "gpt-4.1": {"input": 2.0e-6, "output": 8.0e-6},
        "openai/gpt-5.1": {"input": 1.25e-6, "output": 10.0e-6},
    }

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4o",
            base_url: Optional[str] = None,
            proxy_url: Optional[str] = None,
            timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.usage = LLMUsage()

        LOGGER.info("Initializing LLM client: model=%s base_url=%s proxy=%s", model, bool(base_url), bool(proxy_url))

        timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        http_client = httpx.Client(timeout=timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def complete(self, system_prompt: str, user_prompt: str, expect_json: bool = True) -> Any:
        t0 = time.time()
        LOGGER.info("LLM request started | expect_json=%s | chars=%d", expect_json, len(user_prompt))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        dt = time.time() - t0
        content = (resp.choices[0].message.content or "").strip()

        try:
            pt = int(resp.usage.prompt_tokens or 0)
            ct = int(resp.usage.completion_tokens or 0)
        except Exception:
            pt, ct = 0, 0

        self.usage.prompt_tokens += pt
        self.usage.completion_tokens += ct

        if self.model in self.MODEL_PRICES:
            self.usage.cost_usd += (
                    pt * self.MODEL_PRICES[self.model]["input"]
                    + ct * self.MODEL_PRICES[self.model]["output"]
            )

        LOGGER.info(
            "LLM request finished | %.2fs | prompt_tokens=%s completion_tokens=%s total_cost=%.6f",
            dt, pt, ct, self.usage.cost_usd
        )
        return json_extract_loose(content) if expect_json else content


class EmbeddingModel:
    def __init__(self, model_name: str = "facebook/contriever", device: Optional[str] = None) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        LOGGER.info("Loading embedding tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        LOGGER.info("Loading embedding model: %s on %s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()
        LOGGER.info("Embedding model ready")

    @torch.no_grad()
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text or not isinstance(text, str):
            return None
        t0 = time.time()
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs.to(self.device))
        token_embeddings = outputs[0]
        mask = inputs["attention_mask"].to(self.device)
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sent = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        vec = sent.detach().cpu().numpy().astype(np.float32)[0]
        LOGGER.debug("Embedding computed | chars=%d | %.2fs", len(text), time.time() - t0)
        return vec


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    scores = (m @ q).astype(np.float32)
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64), scores[idx]


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        LOGGER.info("Opening SQLite DB: %s", self.path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        LOGGER.info("SQLite ready")

    def _init_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_aliases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              label TEXT NOT NULL,
              alias TEXT NOT NULL,
              sample_id TEXT NOT NULL,
              embedding BLOB NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(label, alias, sample_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS property_aliases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              label TEXT NOT NULL,
              alias TEXT NOT NULL,
              sample_id TEXT NOT NULL,
              embedding BLOB NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(label, alias, sample_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS triplets (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              subject TEXT NOT NULL,
              relation TEXT NOT NULL,
              object TEXT NOT NULL,
              qualifiers_json TEXT NOT NULL,
              subject_type TEXT NOT NULL DEFAULT '',
              object_type TEXT NOT NULL DEFAULT '',
              source_text_id TEXT NOT NULL DEFAULT '',
              sample_id TEXT NOT NULL,
              prompt_tokens INTEGER,
              completion_tokens INTEGER,
              llm_cost_usd REAL,
              alignment_json TEXT,
              confidence REAL,
              created_at TEXT NOT NULL,
              UNIQUE(subject, relation, object, subject_type, object_type, sample_id, source_text_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS wikidata_cache (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT NOT NULL,
              lang TEXT NOT NULL,
              entity_kind TEXT NOT NULL,
              response_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(query, lang, entity_kind)
            )
            """
        )

        self.conn.commit()

    @staticmethod
    def _vec_to_blob(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            return arr.astype(np.float32)
        return arr.astype(np.float32)

    def upsert_entity_alias(self, label: str, alias: str, sample_id: str, embedding: np.ndarray) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO entity_aliases(label, alias, sample_id, embedding, created_at)
            VALUES(?,?,?,?,?)
            """,
            (sanitize_string(label), sanitize_string(alias), sanitize_string(sample_id), self._vec_to_blob(embedding),
             _now_iso()),
        )
        self.conn.commit()

    def upsert_property_alias(self, label: str, alias: str, sample_id: str, embedding: np.ndarray) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO property_aliases(label, alias, sample_id, embedding, created_at)
            VALUES(?,?,?,?,?)
            """,
            (sanitize_string(label), sanitize_string(alias), sanitize_string(sample_id), self._vec_to_blob(embedding),
             _now_iso()),
        )
        self.conn.commit()

    def retrieve_entity_candidates(self, query_vec: np.ndarray, dim: int, k: int, sample_id: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT label, embedding FROM entity_aliases
            WHERE sample_id = ? OR sample_id = 'all'
            """,
            (sample_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        labels = [r[0] for r in rows]
        mat = np.vstack([self._blob_to_vec(r[1], dim) for r in rows]).astype(np.float32)
        idx, _scores = cosine_topk(query_vec, mat, k=min(k * 3, len(labels)))
        out, seen = [], set()
        for i in idx:
            lab = labels[int(i)]
            if lab not in seen:
                out.append(lab)
                seen.add(lab)
            if len(out) >= k:
                break
        return out

    def retrieve_property_candidates(self, query_vec: np.ndarray, dim: int, k: int, sample_id: str) -> List[str]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT label, embedding FROM property_aliases
            WHERE sample_id = ? OR sample_id = 'all'
            """,
            (sample_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        labels = [r[0] for r in rows]
        mat = np.vstack([self._blob_to_vec(r[1], dim) for r in rows]).astype(np.float32)
        idx, _scores = cosine_topk(query_vec, mat, k=min(k * 3, len(labels)))
        out, seen = [], set()
        for i in idx:
            lab = labels[int(i)]
            if lab not in seen:
                out.append(lab)
                seen.add(lab)
            if len(out) >= k:
                break
        return out

    def upsert_triplet(
            self,
            triplet: Dict[str, Any],
            sample_id: str,
            source_text_id: str,
            llm_usage: LLMUsage,
            alignment: Optional[Dict[str, Any]],
            confidence: Optional[float],
    ) -> None:
        subject = sanitize_string(triplet.get("subject", ""))
        relation = sanitize_string(triplet.get("relation", ""))
        obj = sanitize_string(triplet.get("object", ""))
        qualifiers = triplet.get("qualifiers", [])
        subject_type = sanitize_string(triplet.get("subject_type") or "")
        object_type = sanitize_string(triplet.get("object_type") or "")
        source_text_id = sanitize_string(source_text_id or "")

        self.conn.execute(
            """
            INSERT OR REPLACE INTO triplets(
              subject, relation, object, qualifiers_json,
              subject_type, object_type,
              source_text_id, sample_id,
              prompt_tokens, completion_tokens, llm_cost_usd,
              alignment_json, confidence, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                subject,
                relation,
                obj,
                json.dumps(qualifiers, ensure_ascii=False),
                subject_type,
                object_type,
                source_text_id,
                sample_id,
                int(llm_usage.prompt_tokens),
                int(llm_usage.completion_tokens),
                float(llm_usage.cost_usd),
                json.dumps(alignment, ensure_ascii=False) if alignment else None,
                float(confidence) if confidence is not None else None,
                _now_iso(),
            ),
        )
        self.conn.commit()

    def cache_get(self, query: str, lang: str, entity_kind: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT response_json FROM wikidata_cache
            WHERE query=? AND lang=? AND entity_kind=?
            """,
            (normalize_cache_key(query), lang, entity_kind),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def cache_put(self, query: str, lang: str, entity_kind: str, payload: dict) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO wikidata_cache(query, lang, entity_kind, response_json, created_at)
            VALUES(?,?,?,?,?)
            """,
            (normalize_cache_key(query), lang, entity_kind, json.dumps(payload, ensure_ascii=False), _now_iso()),
        )
        self.conn.commit()


class NameRefiner:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def refine_subject(self, text: str, triplet: Dict[str, Any], candidates: List[str]) -> str:
        triplet_filtered = {k: triplet.get(k) for k in ["subject", "relation", "object"]}
        original = sanitize_string(triplet_filtered.get("subject", ""))
        user_prompt = (
            f'Text: "{text}\n'
            f'Extracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
            f"Original Subject: {original}\n"
            f"Candidate Subjects: {json.dumps(candidates, ensure_ascii=False)}"
            f'"'
        )
        return sanitize_string(self.llm.complete(RANK_SUBJECT_NAMES_PROMPT, user_prompt, expect_json=False))

    def refine_object(self, text: str, triplet: Dict[str, Any], candidates: List[str]) -> str:
        triplet_filtered = {k: triplet.get(k) for k in ["subject", "relation", "object"]}
        original = sanitize_string(triplet_filtered.get("object", ""))
        user_prompt = (
            f'Text: "{text}\n'
            f'Extracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
            f"Original Object: {original}\n"
            f"Candidate Objects: {json.dumps(candidates, ensure_ascii=False)}"
            f'"'
        )
        return sanitize_string(self.llm.complete(RANK_OBJECT_NAMES_PROMPT, user_prompt, expect_json=False))

    def refine_relation(self, text: str, triplet: Dict[str, Any], candidates: List[str]) -> str:
        triplet_filtered = {k: triplet.get(k) for k in ["subject", "relation", "object"]}
        original = sanitize_string(triplet_filtered.get("relation", ""))
        user_prompt = (
            f'Text: "{text}\n'
            f'Extracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
            f"Original relation: {original}\n"
            f"Candidate relations: {json.dumps(candidates, ensure_ascii=False)}"
            f'"'
        )
        return sanitize_string(self.llm.complete(RANK_RELATION_NAMES_PROMPT, user_prompt, expect_json=False))


class RateLimitDeferred(Exception):
    def __init__(self, retry_after: Optional[float] = None) -> None:
        self.retry_after = retry_after
        super().__init__(f"Wikidata rate-limited, retry_after={retry_after}")


class WikidataAligner:
    API_URL = "https://www.wikidata.org/w/api.php"
    SPARQL_URL = "https://query.wikidata.org/sparql"

    def __init__(self, store: SQLiteStore, user_agent: str, min_delay_s: float = 1.0) -> None:
        self.store = store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip,deflate",
        })
        self.sparql_session = requests.Session()
        self.sparql_session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/sparql-results+json",
            "Accept-Encoding": "gzip,deflate",
        })

        self.min_delay_s = float(min_delay_s)
        self.max_retry_after_s = float(os.getenv("WIKONTIC_WIKIDATA_MAX_RETRY_AFTER_S", "30.0"))
        self.sparql_batch_size = max(1, int(os.getenv("WIKONTIC_SPARQL_BATCH_SIZE", "64")))
        self.fallback_limit = max(1, min(int(os.getenv("WIKONTIC_WIKIDATA_FALLBACK_LIMIT", "10")), 50))

        self._last_call = 0.0
        self.blocked_until = 0.0
        self.mem_cache: Dict[Tuple[str, str, str], dict] = {}

    def _sleep_if_needed(self) -> None:
        now = time.time()
        if now < self.blocked_until:
            raise RateLimitDeferred(self.blocked_until - now)
        dt = now - self._last_call
        if dt < self.min_delay_s:
            time.sleep(self.min_delay_s - dt)

    def _cache_key(self, query: str, lang: str, entity_kind: str) -> Tuple[str, str, str]:
        return (normalize_cache_key(query), lang, entity_kind)

    def _escape_sparql_string(self, s: str) -> str:
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return s

    def _string_score(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        if fuzz is not None:
            return float(fuzz.token_set_ratio(a, b)) / 100.0
        import difflib
        return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

    def _get_cached(self, query: str, lang: str, entity_kind: str) -> Optional[dict]:
        key = self._cache_key(query, lang, entity_kind)
        if key in self.mem_cache:
            return self.mem_cache[key]
        payload = self.store.cache_get(query=key[0], lang=lang, entity_kind=entity_kind)
        if payload is not None:
            self.mem_cache[key] = payload
        return payload

    def _put_cached(self, query: str, lang: str, entity_kind: str, payload: dict) -> None:
        key = self._cache_key(query, lang, entity_kind)
        self.mem_cache[key] = payload
        self.store.cache_put(query=key[0], lang=lang, entity_kind=entity_kind, payload=payload)

    def wbsearchentities(self, query: str, lang: str, entity_kind: str = "item", limit: int = 10) -> dict:
        query = normalize_cache_key(query)
        if not query:
            return {"search": []}

        cached = self._get_cached(query=query, lang=lang, entity_kind=entity_kind)
        if cached is not None:
            LOGGER.info("Wikidata cache hit | kind=%s | query=%s", entity_kind, query)
            return cached

        self._sleep_if_needed()
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": lang,
            "uselang": lang,
            "type": entity_kind,
            "limit": max(1, min(int(limit), 50)),
            "format": "json",
        }

        LOGGER.info("Wikidata fallback query | kind=%s | query=%s", entity_kind, query)
        resp = self.session.get(self.API_URL, params=params, timeout=20)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_after_s = float(retry_after) if retry_after else None
            except Exception:
                retry_after_s = None

            LOGGER.warning("Wikidata 429 | query=%s | retry_after=%s", query, retry_after_s)

            if retry_after_s is None:
                retry_after_s = 2.0 + random.random()

            self.blocked_until = time.time() + retry_after_s

            if retry_after_s > self.max_retry_after_s:
                raise RateLimitDeferred(retry_after_s)

            time.sleep(retry_after_s)
            resp = self.session.get(self.API_URL, params=params, timeout=20)

        resp.raise_for_status()
        payload = resp.json()
        self._last_call = time.time()
        self._put_cached(query=query, lang=lang, entity_kind=entity_kind, payload=payload)
        return payload

    def align_item(self, label: str, lang: str) -> Dict[str, Any]:
        label = normalize_cache_key(label)
        if not label:
            return {"qid": None, "best_label": None, "score": 0.0, "candidates": []}
        if _QID_RE.fullmatch(label):
            return {"qid": label, "best_label": None, "score": 1.0, "candidates": []}
        if should_skip_item_alignment(label):
            return {"qid": None, "best_label": None, "score": 0.0, "candidates": [], "skipped": True}

        payload = self.wbsearchentities(label, lang=lang, entity_kind="item", limit=self.fallback_limit)
        candidates = []
        best = {"qid": None, "best_label": None, "score": 0.0, "candidates": []}
        for item in payload.get("search", [])[: self.fallback_limit]:
            qid = item.get("id")
            lab = item.get("label")
            sc = self._string_score(label, lab or "")
            candidates.append({
                "qid": qid,
                "label": lab,
                "description": item.get("description"),
                "url": item.get("url"),
                "score": sc,
            })
            if sc > best["score"]:
                best = {"qid": qid, "best_label": lab, "score": sc, "candidates": candidates}
        best["candidates"] = candidates
        return best

    def align_property(self, label: str, lang: str) -> Dict[str, Any]:
        label = normalize_cache_key(label)
        if not label:
            return {"pid": None, "best_label": None, "score": 0.0, "candidates": []}
        if _PID_RE.fullmatch(label):
            return {"pid": label, "best_label": None, "score": 1.0, "candidates": []}

        payload = self.wbsearchentities(label, lang=lang, entity_kind="property", limit=self.fallback_limit)
        candidates = []
        best = {"pid": None, "best_label": None, "score": 0.0, "candidates": []}
        for item in payload.get("search", [])[: self.fallback_limit]:
            pid = item.get("id")
            lab = item.get("label")
            sc = self._string_score(label, lab or "")
            candidates.append({
                "pid": pid,
                "label": lab,
                "description": item.get("description"),
                "url": item.get("url"),
                "score": sc,
            })
            if sc > best["score"]:
                best = {"pid": pid, "best_label": lab, "score": sc, "candidates": candidates}
        best["candidates"] = candidates
        return best

    def _sparql_batch_exact_items(self, labels: List[str], lang: str) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        labels = [normalize_cache_key(x) for x in labels if
                  normalize_cache_key(x) and not should_skip_item_alignment(x)]

        for batch in iter_chunks(labels, self.sparql_batch_size):
            values = " ".join(f'"{self._escape_sparql_string(x)}"@{lang}' for x in batch)
            query = f"""
                SELECT ?inputLabel ?item ?itemLabel ?itemDescription WHERE {{
                  VALUES ?inputLabel {{ {values} }}
                  ?item rdfs:label ?inputLabel .

                  OPTIONAL {{
                    ?item rdfs:label ?itemLabel .
                    FILTER(LANG(?itemLabel) = "{lang}")
                  }}
                  OPTIONAL {{
                    ?item schema:description ?itemDescription .
                    FILTER(LANG(?itemDescription) = "{lang}")
                  }}
                }}
                """

            LOGGER.info("SPARQL item batch | lang=%s | batch_size=%d", lang, len(batch))
            resp = self.sparql_session.get(
                self.SPARQL_URL,
                params={"query": query, "format": "json"},
                timeout=45,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                LOGGER.warning("SPARQL 429 | retry_after=%s", retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("results", {}).get("bindings", [])
            LOGGER.info("SPARQL item batch returned %d rows", len(rows))

            for row in data.get("results", {}).get("bindings", []):
                input_label = row.get("inputLabel", {}).get("value", "")
                item_uri = row.get("item", {}).get("value", "")
                item_label = row.get("itemLabel", {}).get("value", input_label)
                item_desc = row.get("itemDescription", {}).get("value")
                qid = item_uri.rsplit("/", 1)[-1] if item_uri else None
                if not input_label or not qid:
                    continue

                score = self._string_score(input_label, item_label or input_label)
                candidate = {
                    "qid": qid,
                    "label": item_label,
                    "description": item_desc,
                    "url": item_uri,
                    "score": score,
                }

                prev = results.get(input_label)
                if prev is None or candidate["score"] > prev.get("score", 0.0):
                    results[input_label] = {
                        "qid": qid,
                        "best_label": item_label,
                        "score": score,
                        "candidates": [candidate],
                        "source": "sparql-exact",
                    }
                else:
                    prev.setdefault("candidates", []).append(candidate)

        return results

    def _sparql_batch_exact_properties(self, labels: List[str], lang: str) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        labels = [normalize_cache_key(x) for x in labels if normalize_cache_key(x)]

        for batch in iter_chunks(labels, self.sparql_batch_size):
            values = " ".join(f'"{self._escape_sparql_string(x)}"@{lang}' for x in batch)
            query = f"""
                SELECT ?inputLabel ?prop ?propLabel ?propDescription WHERE {{
                  VALUES ?inputLabel {{ {values} }}
                  ?prop a wikibase:Property .
                  ?prop rdfs:label ?inputLabel .

                  OPTIONAL {{
                    ?prop rdfs:label ?propLabel .
                    FILTER(LANG(?propLabel) = "{lang}")
                  }}
                  OPTIONAL {{
                    ?prop schema:description ?propDescription .
                    FILTER(LANG(?propDescription) = "{lang}")
                  }}
                }}
                """

            LOGGER.info("SPARQL property batch | lang=%s | batch_size=%d", lang, len(batch))
            resp = self.sparql_session.get(
                self.SPARQL_URL,
                params={"query": query, "format": "json"},
                timeout=45,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                LOGGER.warning("SPARQL 429 | retry_after=%s", retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("results", {}).get("bindings", [])
            LOGGER.info("SPARQL item batch returned %d rows", len(rows))

            for row in data.get("results", {}).get("bindings", []):
                input_label = row.get("inputLabel", {}).get("value", "")
                prop_uri = row.get("prop", {}).get("value", "")
                prop_label = row.get("propLabel", {}).get("value", input_label)
                prop_desc = row.get("propDescription", {}).get("value")
                pid = prop_uri.rsplit("/", 1)[-1] if prop_uri else None
                if not input_label or not pid:
                    continue

                score = self._string_score(input_label, prop_label or input_label)
                candidate = {
                    "pid": pid,
                    "label": prop_label,
                    "description": prop_desc,
                    "url": prop_uri,
                    "score": score,
                }

                prev = results.get(input_label)
                if prev is None or candidate["score"] > prev.get("score", 0.0):
                    results[input_label] = {
                        "pid": pid,
                        "best_label": prop_label,
                        "score": score,
                        "candidates": [candidate],
                        "source": "sparql-exact",
                    }
                else:
                    prev.setdefault("candidates", []).append(candidate)

        return results

    def batch_align_items(self, labels: Iterable[str], lang: str) -> Dict[str, Dict[str, Any]]:
        labels_norm = sorted({normalize_cache_key(x) for x in labels if normalize_cache_key(x)})
        results: Dict[str, Dict[str, Any]] = {}
        unresolved: List[str] = []

        for label in labels_norm:
            if should_skip_item_alignment(label):
                results[label] = {"qid": None, "best_label": None, "score": 0.0, "candidates": [], "skipped": True}
                continue
            if _QID_RE.fullmatch(label):
                results[label] = {"qid": label, "best_label": None, "score": 1.0, "candidates": []}
                continue

            cached = self._get_cached(label, lang, "item")
            if cached is not None:
                search = cached.get("search", [])
                if search:
                    top = search[0]
                    results[label] = {
                        "qid": top.get("id"),
                        "best_label": top.get("label"),
                        "score": self._string_score(label, top.get("label") or ""),
                        "candidates": [
                            {
                                "qid": x.get("id"),
                                "label": x.get("label"),
                                "description": x.get("description"),
                                "url": x.get("url"),
                                "score": self._string_score(label, x.get("label") or ""),
                            }
                            for x in search
                        ],
                        "source": "cache-search",
                    }
                else:
                    results[label] = {"qid": None, "best_label": None, "score": 0.0, "candidates": [],
                                      "source": "cache-empty"}
                continue

            unresolved.append(label)

        if unresolved:
            LOGGER.info("Batch item align: unresolved before SPARQL = %d", len(unresolved))
            sparql_res = self._sparql_batch_exact_items(unresolved, lang=lang)
            for label, payload in sparql_res.items():
                results[label] = payload
                self._put_cached(
                    query=label,
                    lang=lang,
                    entity_kind="item",
                    payload={"search": [{
                        "id": payload.get("qid"),
                        "label": payload.get("best_label"),
                        "description": payload.get("candidates", [{}])[0].get("description"),
                        "url": payload.get("candidates", [{}])[0].get("url"),
                    }]}
                )

        still_unresolved = [x for x in unresolved if x not in results]
        LOGGER.info("Batch item align: unresolved after SPARQL = %d", len(still_unresolved))

        for label in still_unresolved:
            try:
                results[label] = self.align_item(label, lang)
            except RateLimitDeferred as e:
                LOGGER.warning("Deferred item lookup | label=%s | retry_after=%s", label, e.retry_after)
                results[label] = {"qid": None, "best_label": None, "score": 0.0, "candidates": [], "deferred": True}

        return results

    def batch_align_properties(self, labels: Iterable[str], lang: str) -> Dict[str, Dict[str, Any]]:
        labels_norm = sorted({normalize_cache_key(x) for x in labels if normalize_cache_key(x)})
        results: Dict[str, Dict[str, Any]] = {}
        unresolved: List[str] = []

        for label in labels_norm:
            if _PID_RE.fullmatch(label):
                results[label] = {"pid": label, "best_label": None, "score": 1.0, "candidates": []}
                continue

            cached = self._get_cached(label, lang, "property")
            if cached is not None:
                search = cached.get("search", [])
                if search:
                    top = search[0]
                    results[label] = {
                        "pid": top.get("id"),
                        "best_label": top.get("label"),
                        "score": self._string_score(label, top.get("label") or ""),
                        "candidates": [
                            {
                                "pid": x.get("id"),
                                "label": x.get("label"),
                                "description": x.get("description"),
                                "url": x.get("url"),
                                "score": self._string_score(label, x.get("label") or ""),
                            }
                            for x in search
                        ],
                        "source": "cache-search",
                    }
                else:
                    results[label] = {"pid": None, "best_label": None, "score": 0.0, "candidates": [],
                                      "source": "cache-empty"}
                continue

            unresolved.append(label)

        if unresolved:
            LOGGER.info("Batch property align: unresolved before SPARQL = %d", len(unresolved))
            sparql_res = self._sparql_batch_exact_properties(unresolved, lang=lang)
            for label, payload in sparql_res.items():
                results[label] = payload
                self._put_cached(
                    query=label,
                    lang=lang,
                    entity_kind="property",
                    payload={"search": [{
                        "id": payload.get("pid"),
                        "label": payload.get("best_label"),
                        "description": payload.get("candidates", [{}])[0].get("description"),
                        "url": payload.get("candidates", [{}])[0].get("url"),
                    }]}
                )

        still_unresolved = [x for x in unresolved if x not in results]
        LOGGER.info("Batch property align: unresolved after SPARQL = %d", len(still_unresolved))

        for label in still_unresolved:
            try:
                results[label] = self.align_property(label, lang)
            except RateLimitDeferred as e:
                LOGGER.warning("Deferred property lookup | label=%s | retry_after=%s", label, e.retry_after)
                results[label] = {"pid": None, "best_label": None, "score": 0.0, "candidates": [], "deferred": True}

        return results

    def batch_align_triplets(self, triplets: List[Dict[str, Any]], lang: str) -> Tuple[
        List[Optional[Dict[str, Any]]], List[Optional[float]]]:
        unique_items: set[str] = set()
        unique_props: set[str] = set()

        for t in triplets:
            unique_items.add(normalize_cache_key(t.get("subject", "")))
            unique_items.add(normalize_cache_key(t.get("object", "")))
            unique_props.add(normalize_cache_key(t.get("relation", "")))
            for q in t.get("qualifiers", []) or []:
                unique_props.add(normalize_cache_key(q.get("relation", "")))
                unique_items.add(normalize_cache_key(q.get("object", "")))

        unique_items.discard("")
        unique_props.discard("")

        LOGGER.info("Batch align collection | unique_items=%d | unique_props=%d", len(unique_items), len(unique_props))

        item_map = self.batch_align_items(unique_items, lang=lang)
        prop_map = self.batch_align_properties(unique_props, lang=lang)

        alignments: List[Optional[Dict[str, Any]]] = []
        confidences: List[Optional[float]] = []

        for t in triplets:
            subj = normalize_cache_key(t.get("subject", ""))
            obj = normalize_cache_key(t.get("object", ""))
            rel = normalize_cache_key(t.get("relation", ""))

            subj_al = item_map.get(subj, {"qid": None, "best_label": None, "score": 0.0, "candidates": []})
            obj_al = item_map.get(obj, {"qid": None, "best_label": None, "score": 0.0, "candidates": []})
            rel_al = prop_map.get(rel, {"pid": None, "best_label": None, "score": 0.0, "candidates": []})

            qual_als = []
            for q in t.get("qualifiers", []) or []:
                qr = normalize_cache_key(q.get("relation", ""))
                qo = normalize_cache_key(q.get("object", ""))
                qual_als.append({
                    "relation": qr,
                    "relation_alignment": prop_map.get(qr, {"pid": None, "best_label": None, "score": 0.0,
                                                            "candidates": []}),
                    "object": qo,
                    "object_alignment": item_map.get(qo,
                                                     {"qid": None, "best_label": None, "score": 0.0, "candidates": []}),
                })

            alignment = {
                "lang": lang,
                "subject": subj_al,
                "object": obj_al,
                "relation": rel_al,
                "qualifiers": qual_als,
            }
            alignments.append(alignment)

            base = 0.35
            s1 = float(subj_al.get("score", 0.0) or 0.0)
            s2 = float(obj_al.get("score", 0.0) or 0.0)
            conf = max(0.0, min(1.0, base + 0.325 * s1 + 0.325 * s2))
            confidences.append(conf)

            t["alignment"] = {
                "subject_qid": subj_al.get("qid"),
                "object_qid": obj_al.get("qid"),
                "relation_pid": rel_al.get("pid"),
                "subject_best_label": subj_al.get("best_label"),
                "object_best_label": obj_al.get("best_label"),
                "relation_best_label": rel_al.get("best_label"),
            }
            t["confidence"] = conf

        return alignments, confidences


class WikonticLocalPipeline:
    def __init__(
            self,
            store: SQLiteStore,
            llm: LLMClient,
            embedder: EmbeddingModel,
            sample_id: str = "default",
            k_candidates: int = 10,
            do_refine_default: bool = False,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.sample_id = sample_id
        self.k_candidates = k_candidates
        self.do_refine_default = do_refine_default
        self.refiner = NameRefiner(llm)

    def extract_raw_triplets(self, text: str) -> List[Dict[str, Any]]:
        LOGGER.info("Extracting raw triplets | chars=%d", len(text))
        result = self.llm.complete(TRIPLET_EXTRACTION_PROMPT, f'Text: "{text}"', expect_json=True)
        if isinstance(result, dict) and "triplets" in result and isinstance(result["triplets"], list):
            LOGGER.info("Raw triplet extraction returned %d triplets", len(result["triplets"]))
            return result["triplets"]
        if isinstance(result, list):
            LOGGER.info("Raw triplet extraction returned %d triplets", len(result))
            return result
        raise ValueError(f"Unexpected LLM output shape: {type(result)}: {str(result)[:200]}")

    def _normalize_triplet(self, t: Dict[str, Any]) -> Dict[str, Any]:
        t = dict(t)
        t["subject"] = sanitize_string(t.get("subject", ""))
        t["relation"] = sanitize_string(t.get("relation", ""))
        t["object"] = sanitize_string(t.get("object", ""))
        if "subject_type" in t:
            t["subject_type"] = sanitize_string(t.get("subject_type"))
        if "object_type" in t:
            t["object_type"] = sanitize_string(t.get("object_type"))
        quals = t.get("qualifiers", [])
        if not isinstance(quals, list):
            quals = []
        norm_quals = []
        for q in quals:
            if not isinstance(q, dict):
                continue
            qr = sanitize_string(q.get("relation", ""))
            qo = sanitize_string(q.get("object", ""))
            if qr and qo:
                norm_quals.append({"relation": qr, "object": qo})
        t["qualifiers"] = norm_quals
        return t

    def _dedupe(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for t in triplets:
            key = (
                t.get("subject", ""),
                t.get("relation", ""),
                t.get("object", ""),
                t.get("subject_type", ""),
                t.get("object_type", ""),
                json.dumps(t.get("qualifiers", []), ensure_ascii=False, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        LOGGER.info("Deduped triplets: %d -> %d", len(triplets), len(out))
        return out

    def _refine_entity_name(self, text: str, triplet: Dict[str, Any], is_object: bool) -> str:
        key = "object" if is_object else "subject"
        original = sanitize_string(triplet.get(key, ""))
        if not original:
            return original

        qvec = self.embedder.get_embedding(original)
        if qvec is None:
            return original

        candidates = self.store.retrieve_entity_candidates(
            query_vec=qvec,
            dim=int(qvec.shape[0]),
            k=self.k_candidates,
            sample_id=self.sample_id,
        )

        if not candidates or original in candidates:
            refined = original
        else:
            refined = self.refiner.refine_object(text, triplet,
                                                 candidates) if is_object else self.refiner.refine_subject(text,
                                                                                                           triplet,
                                                                                                           candidates)
            if is_none_string(refined):
                refined = original

        self.store.upsert_entity_alias(label=refined, alias=original, sample_id=self.sample_id, embedding=qvec)
        return refined

    def _refine_relation_name(self, text: str, triplet: Dict[str, Any]) -> str:
        original = sanitize_string(triplet.get("relation", ""))
        if not original:
            return original

        qvec = self.embedder.get_embedding(original)
        if qvec is None:
            return original

        candidates = self.store.retrieve_property_candidates(
            query_vec=qvec,
            dim=int(qvec.shape[0]),
            k=self.k_candidates,
            sample_id=self.sample_id,
        )

        if not candidates or original in candidates:
            refined = original
        else:
            refined = self.refiner.refine_relation(text, triplet, candidates)
            if is_none_string(refined):
                refined = original

        self.store.upsert_property_alias(label=refined, alias=original, sample_id=self.sample_id, embedding=qvec)
        return refined

    def _run_optional_refinement(self, chunks: List[str], triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not triplets:
            return triplets

        chunk_context = " ".join(chunks[:2]) if chunks else ""
        refined = []
        for t in triplets:
            t = dict(t)
            subj = self._refine_entity_name(chunk_context, t, is_object=False)
            obj = self._refine_entity_name(chunk_context, t, is_object=True)
            rel = self._refine_relation_name(chunk_context, t)
            t["subject"] = subj
            t["object"] = obj
            t["relation"] = rel
            refined.append(t)
        return refined

    def extract_triples(
            self,
            text: str,
            do_align: bool = False,
            do_refine: Optional[bool] = None,
            return_usage: bool = False,
    ) -> Any:
        t0 = time.time()
        text = normalize_text(text)
        lang = detect_lang(text)
        source_text_id = sha256_id(text)
        if do_refine is None:
            do_refine = self.do_refine_default

        LOGGER.info(
            "extract_triples started | chars=%d | do_align=%s | do_refine=%s | lang=%s",
            len(text), do_align, do_refine, lang
        )

        chunks = chunk_text_sentence_aware(text)
        LOGGER.info("Text split into %d chunk(s)", len(chunks))
        all_triplets: List[Dict[str, Any]] = []

        start_usage = LLMUsage(
            self.llm.usage.prompt_tokens,
            self.llm.usage.completion_tokens,
            self.llm.usage.cost_usd,
        )

        for chunk_idx, ch in enumerate(chunks, start=1):
            LOGGER.info("Processing chunk %d/%d | chars=%d", chunk_idx, len(chunks), len(ch))
            raw = self.extract_raw_triplets(ch)
            for t in raw:
                all_triplets.append(self._normalize_triplet(t))

        all_triplets = self._dedupe(all_triplets)

        alignment_results: List[Optional[Dict[str, Any]]] = [None] * len(all_triplets)
        confidences: List[Optional[float]] = [None] * len(all_triplets)

        if do_align and all_triplets:
            ua = os.getenv("WIKIDATA_USER_AGENT") or "wikontic-local-batched/0.1 (set WIKIDATA_USER_AGENT)"
            min_delay_s = float(os.getenv("WIKONTIC_WIKIDATA_MIN_DELAY_S", "1.0"))
            wd = WikidataAligner(self.store, user_agent=ua, min_delay_s=min_delay_s)
            LOGGER.info("Running batched Wikidata alignment for %d triplets", len(all_triplets))
            alignment_results, confidences = wd.batch_align_triplets(all_triplets, lang=lang)

        if do_refine and all_triplets:
            LOGGER.info("Running optional refinement after batched alignment")
            all_triplets = self._run_optional_refinement(chunks, all_triplets)

        delta_usage = LLMUsage(
            prompt_tokens=self.llm.usage.prompt_tokens - start_usage.prompt_tokens,
            completion_tokens=self.llm.usage.completion_tokens - start_usage.completion_tokens,
            cost_usd=self.llm.usage.cost_usd - start_usage.cost_usd,
        )

        LOGGER.info(
            "Persisting %d triplets | prompt=%d completion=%d total=%d cost=%.6f",
            len(all_triplets),
            delta_usage.prompt_tokens,
            delta_usage.completion_tokens,
            delta_usage.total_tokens,
            delta_usage.cost_usd,
        )

        for i, t in enumerate(all_triplets):
            self.store.upsert_triplet(
                triplet=t,
                sample_id=self.sample_id,
                source_text_id=source_text_id,
                llm_usage=delta_usage,
                alignment=alignment_results[i] if i < len(alignment_results) else None,
                confidence=confidences[i] if i < len(confidences) else None,
            )

        LOGGER.info("extract_triples finished | triplets=%d | elapsed=%.2fs", len(all_triplets), time.time() - t0)

        if return_usage:
            return {
                "triplets": all_triplets,
                "usage": delta_usage.to_dict(model=self.llm.model),
            }

        return all_triplets


_PIPELINE: Optional[WikonticLocalPipeline] = None


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None
    LOGGER.info("Pipeline reset")


def _build_pipeline() -> WikonticLocalPipeline:
    LOGGER.info("Building pipeline")
    load_dotenv(find_dotenv())

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_KEY") or ""
    if not api_key:
        raise RuntimeError("No API key found. Set OPENAI_API_KEY or OPENROUTER_KEY in your environment/.env")

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    proxy_url = os.getenv("PROXY_URL")
    model = os.getenv("WIKONTIC_LLM_MODEL", "gpt-4o")
    db_path = os.getenv("WIKONTIC_DB_PATH", os.path.abspath("wikontic_local.sqlite"))
    sample_id = os.getenv("WIKONTIC_SAMPLE_ID", "default")
    k_candidates = int(os.getenv("WIKONTIC_K_CANDIDATES", "10"))
    do_refine_default = os.getenv("WIKONTIC_DO_REFINE", "false").strip().lower() in {"1", "true", "yes", "y", "on"}

    llm = LLMClient(api_key=api_key, model=model, base_url=base_url, proxy_url=proxy_url)
    store = SQLiteStore(db_path)
    embed_model_name = os.getenv("WIKONTIC_EMBED_MODEL", "facebook/contriever")
    device = os.getenv("WIKONTIC_DEVICE")
    embedder = EmbeddingModel(model_name=embed_model_name, device=device)

    return WikonticLocalPipeline(
        store=store,
        llm=llm,
        embedder=embedder,
        sample_id=sample_id,
        k_candidates=k_candidates,
        do_refine_default=do_refine_default,
    )


def extract_triples(
        text: str,
        do_align: bool = False,
        do_refine: Optional[bool] = None,
        return_usage: bool = False,
) -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = _build_pipeline()
    return _PIPELINE.extract_triples(
        text=text,
        do_align=do_align,
        do_refine=do_refine,
        return_usage=return_usage,
    )


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Wikontic triple extraction with local SQLite storage and batched Wikidata alignment.")
    p.add_argument("--text", type=str, required=True, help="Input text")
    p.add_argument("--do-align", action="store_true", help="Batched align entities/properties to Wikidata")
    p.add_argument("--do-refine", action="store_true", help="Enable refinement")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = p.parse_args()

    if args.quiet:
        set_verbose(False)

    out = extract_triples(
        args.text,
        do_align=args.do_align,
        do_refine=args.do_refine,
        return_usage=True,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
