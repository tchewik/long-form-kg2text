import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


import gc
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


_FILTER_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their",
    "his", "her", "they", "them", "he", "she", "we", "you",
}


def configure_reranker_for_filtering(
        reranker: Any,
        score_batch_size: int = 4,
) -> Any:
    """
    Best-effort configuration for safer long inference runs.
    """
    for obj in (reranker, getattr(reranker, "model", None)):
        if obj is not None and hasattr(obj, "eval"):
            obj.eval()

    for attr in (
        "batch_size",
        "eval_batch_size",
        "inference_batch_size",
        "score_batch_size",
    ):
        if hasattr(reranker, attr):
            setattr(reranker, attr, score_batch_size)

    return reranker


def cleanup_accelerator_memory() -> None:
    gc.collect()

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()

    return any(
        marker in msg
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "hip out of memory",
        )
    )


def to_plain_python(value: Any) -> Any:
    """
    Avoid retaining tensors, computation graphs, or GPU buffers in filter state.
    """
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()

        if value.ndim == 0:
            return value.item()

        return value.tolist()

    if isinstance(value, dict):
        return {k: to_plain_python(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_plain_python(v) for v in value]

    if isinstance(value, tuple):
        return tuple(to_plain_python(v) for v in value)

    return value


def score_text_detailed_safe(
        reranker: Any,
        triples: List[Dict[str, str]],
        text: str,
        score_batch_size: Optional[int] = 4,
) -> Dict[str, Any]:
    """
    Score under inference mode.

    Uses batch_size when the reranker exposes it, but falls back cleanly
    for rerankers whose score_text_detailed signature does not accept it.
    """
    inference_context = (
        torch.inference_mode()
        if torch is not None
        else nullcontext()
    )

    try:
        with inference_context:
            try:
                result = reranker.score_text_detailed(
                    triples,
                    text,
                    batch_size=score_batch_size,
                )
            except TypeError:
                result = reranker.score_text_detailed(triples, text)

        return to_plain_python(result)

    except RuntimeError as exc:
        if is_oom_error(exc):
            cleanup_accelerator_memory()
        raise


def tokenize_for_filtering(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())

    return {
        tok
        for tok in toks
        if len(tok) >= 3 and tok not in _FILTER_STOPWORDS
    }


def triple_token_set(triples: List[Dict[str, str]]) -> set[str]:
    joined = []

    for triple in triples:
        joined.append(triple.get("subject", ""))
        joined.append(triple.get("predicate", ""))
        joined.append(
            triple.get("object", "")
            .replace("<Q>", " ")
            .replace("<T>", " ")
        )

    return tokenize_for_filtering(" ".join(joined))


from collections import Counter
from typing import Optional


def build_kg_token_profile(
        triples: List[Dict[str, str]],
) -> Dict[str, set[str]]:
    """
    Build token groups for cheap deletion-candidate selection.

    Key idea:
      - subject tokens are weak evidence because they often just identify
        the article/document topic
      - predicate/object tokens are stronger evidence that a sentence may
        support a KG triple
    """
    subject_tokens_all: list[str] = []
    predicate_tokens: set[str] = set()
    object_tokens: set[str] = set()

    for triple in triples:
        subject_tokens = tokenize_for_filtering(triple.get("subject", ""))
        pred_tokens = tokenize_for_filtering(triple.get("predicate", ""))

        object_text = (
            triple.get("object", "")
            .replace("<Q>", " ")
            .replace("<T>", " ")
        )
        obj_tokens = tokenize_for_filtering(object_text)

        subject_tokens_all.extend(subject_tokens)
        predicate_tokens.update(pred_tokens)
        object_tokens.update(obj_tokens)

    subject_counts = Counter(subject_tokens_all)

    # Tokens appearing as subjects in many triples are usually the main topic.
    # They should not protect a sentence from being tested for deletion.
    if triples:
        repeated_subject_threshold = max(2, int(0.5 * len(triples)))
    else:
        repeated_subject_threshold = 2

    repeated_subject_tokens = {
        tok
        for tok, count in subject_counts.items()
        if count >= repeated_subject_threshold
    }

    all_subject_tokens = set(subject_counts)

    evidence_tokens = predicate_tokens | object_tokens

    # Remove repeated main-subject tokens from the evidence set.
    # Example: if every triple is about "Marie Curie", the tokens
    # "marie" and "curie" should not make a sentence look KG-supported.
    evidence_tokens = evidence_tokens - repeated_subject_tokens

    return {
        "all_subject_tokens": all_subject_tokens,
        "repeated_subject_tokens": repeated_subject_tokens,
        "predicate_tokens": predicate_tokens,
        "object_tokens": object_tokens,
        "evidence_tokens": evidence_tokens,
    }


def select_deletion_candidates(
        sentences: List[str],
        triples: List[Dict[str, str]],
        protect_first_sentence: bool = True,
        max_overlap_tokens: Optional[int] = 1,
        max_overlap_ratio: Optional[float] = 0.15,
        max_candidate_sentences: Optional[int] = 16,
) -> List[int]:
    """
    Rank deletion candidates, but do not hard-exclude sentences solely because
    they overlap with KG tokens.

    This is safer than strict lexical pruning:
      - subject-only overlap is weak evidence
      - predicate/object overlap is stronger evidence
      - but all non-protected sentences remain eligible if the candidate budget
        is large enough or None

    Set max_candidate_sentences=None to recover the original all-sentences
    search order, except for protect_first_sentence and prob-drop constraints.
    """
    profile = build_kg_token_profile(triples)

    evidence_tokens = profile["evidence_tokens"]
    repeated_subject_tokens = profile["repeated_subject_tokens"]
    all_subject_tokens = profile["all_subject_tokens"]

    ranked: List[Tuple[int, int, float, int, int, int]] = []

    for i, sentence in enumerate(sentences):
        if protect_first_sentence and i == 0:
            continue

        sent_tokens = tokenize_for_filtering(sentence)

        if not sent_tokens:
            evidence_overlap_count = 0
            evidence_overlap_ratio = 0.0
            repeated_subject_overlap_count = 0
            any_subject_overlap_count = 0
        else:
            evidence_overlap = sent_tokens & evidence_tokens
            repeated_subject_overlap = sent_tokens & repeated_subject_tokens
            any_subject_overlap = sent_tokens & all_subject_tokens

            evidence_overlap_count = len(evidence_overlap)
            evidence_overlap_ratio = evidence_overlap_count / max(len(sent_tokens), 1)
            repeated_subject_overlap_count = len(repeated_subject_overlap)
            any_subject_overlap_count = len(any_subject_overlap)

        # Soft pass/fail only affects ranking, not eligibility.
        token_ok = (
            max_overlap_tokens is None
            or evidence_overlap_count <= max_overlap_tokens
        )
        ratio_ok = (
            max_overlap_ratio is None
            or evidence_overlap_ratio <= max_overlap_ratio
        )

        lexical_priority_bucket = 0 if token_ok and ratio_ok else 1

        ranked.append(
            (
                lexical_priority_bucket,
                evidence_overlap_count,
                evidence_overlap_ratio,
                -repeated_subject_overlap_count,
                -any_subject_overlap_count,
                -i,  # later sentences first, matching original backward tendency
            )
        )

    ranked.sort()

    selected = [-item[-1] for item in ranked]

    if max_candidate_sentences is not None:
        selected = selected[:max_candidate_sentences]

    return sorted(selected, reverse=True)


def default_sentence_splitter(text: str) -> List[str]:
    """
    Lightweight sentence splitter.

    For more precise splitting, replace this with spaCy/NLTK externally.
    """
    text = text.strip()
    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[\"'“”‘’\(\[]*[A-Z0-9])",
        text,
    )

    return [p.strip() for p in parts if p and p.strip()]


def parse_wikidockg_triples(triples_dict: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Same triple parsing logic as WikiDocKGLoader.
    Kept here so the filtering script can run without instantiating the loader.
    """

    def enrich_wqualifiers(obj: str, qualifiers: list) -> str:
        if not qualifiers:
            return obj

        res = obj.strip()
        for q in qualifiers:
            res += (
                "<Q>"
                + q.get("relation", "").strip()
                + "<T>"
                + q.get("object", "").strip()
            )
        return res

    parsed_triples: List[Dict[str, str]] = []

    for triple in triples_dict:
        obj = triple["object"].strip()
        qualifiers = triple.get("qualifiers", [])
        object_with_q = enrich_wqualifiers(obj, qualifiers)

        parsed_triples.append(
            {
                "subject": triple["subject"].strip(),
                "predicate": triple["relation"].strip(),
                "object": object_with_q,
            }
        )

    return parsed_triples


def filter_text_by_factspotter(
        text: str,
        triples: List[Dict[str, str]],
        reranker: Any,
        min_sentences_to_keep: int = 1,
        max_mean_entail_prob_drop: float = 0.02,
        protect_first_sentence: bool = True,
        score_batch_size: Optional[int] = 4,
        cleanup_every_scores: int = 10,
        max_overlap_tokens: int = 1,
        max_overlap_ratio: float = 0.99,
        max_candidate_sentences: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """
    Sparse-deletion optimized greedy filtering.

    This version assumes most rows will remove zero or one sentence.
    Therefore it does NOT test every sentence. It only tests sentences that
    have weak lexical overlap with the KG triples.

    Deletion is accepted only if:
      1. sentence i=0 is not removed;
      2. entailed_mask stays unchanged;
      3. mean_entail_prob drops by no more than 0.02 by default.
    """
    sentences = default_sentence_splitter(text)

    stats: Dict[str, Any] = {
        "original_num_sentences": len(sentences),
        "filtered_num_sentences": len(sentences),
        "removed_sentence_indices": [],
        "candidate_sentence_indices": [],
        "original_recall": None,
        "filtered_recall": None,
        "original_mean_entail_prob": None,
        "filtered_mean_entail_prob": None,
        "max_mean_entail_prob_drop": max_mean_entail_prob_drop,
        "protect_first_sentence": protect_first_sentence,
        "score_batch_size": score_batch_size,
        "num_score_calls": 0,
        "num_candidates_tested": 0,
        "rejected_due_to_mask_change": 0,
        "rejected_due_to_prob_drop": 0,
        "rejected_due_to_oom": 0,
        "filter_error": None,
        "candidate_filter": {
            "max_overlap_tokens": max_overlap_tokens,
            "max_overlap_ratio": max_overlap_ratio,
            "max_candidate_sentences": max_candidate_sentences,
        },
    }

    if len(sentences) <= min_sentences_to_keep:
        return text, stats

    def build_text(mask: List[bool]) -> str:
        return " ".join(
            sent for sent, is_kept in zip(sentences, mask) if is_kept
        ).strip()

    try:
        current_result = score_text_detailed_safe(
            reranker=reranker,
            triples=triples,
            text=text,
            score_batch_size=score_batch_size,
        )
        stats["num_score_calls"] += 1
    except RuntimeError as exc:
        if is_oom_error(exc):
            stats["filter_error"] = "oom_scoring_original_text"
            return text, stats
        raise

    stats["original_recall"] = current_result["recall"]
    stats["original_mean_entail_prob"] = current_result["mean_entail_prob"]

    kept = [True] * len(sentences)

    candidate_indices = select_deletion_candidates(
        sentences=sentences,
        triples=triples,
        protect_first_sentence=protect_first_sentence,
        max_overlap_tokens=max_overlap_tokens,
        max_overlap_ratio=max_overlap_ratio,
        max_candidate_sentences=max_candidate_sentences,
    )

    stats["candidate_sentence_indices"] = candidate_indices

    for i in candidate_indices:
        if protect_first_sentence and i == 0:
            continue

        if not kept[i]:
            continue

        if sum(kept) <= min_sentences_to_keep:
            break

        candidate_kept = kept.copy()
        candidate_kept[i] = False

        candidate_text = build_text(candidate_kept)

        if not candidate_text:
            continue

        try:
            candidate_result = score_text_detailed_safe(
                reranker=reranker,
                triples=triples,
                text=candidate_text,
                score_batch_size=score_batch_size,
            )
            stats["num_score_calls"] += 1
            stats["num_candidates_tested"] += 1

            if (
                cleanup_every_scores > 0
                and stats["num_score_calls"] % cleanup_every_scores == 0
            ):
                cleanup_accelerator_memory()

        except RuntimeError as exc:
            if is_oom_error(exc):
                stats["rejected_due_to_oom"] += 1
                continue
            raise

        same_entailment_mask = (
            candidate_result["entailed_mask"] == current_result["entailed_mask"]
        )

        current_prob = current_result.get("mean_entail_prob")
        candidate_prob = candidate_result.get("mean_entail_prob")

        if current_prob is None or candidate_prob is None:
            mean_entail_prob_drop = 0.0
        else:
            mean_entail_prob_drop = current_prob - candidate_prob

        probability_ok = mean_entail_prob_drop <= max_mean_entail_prob_drop

        if same_entailment_mask and probability_ok:
            kept = candidate_kept
            current_result = candidate_result
            stats["removed_sentence_indices"].append(i)
        else:
            if not same_entailment_mask:
                stats["rejected_due_to_mask_change"] += 1
            elif not probability_ok:
                stats["rejected_due_to_prob_drop"] += 1

        # Do not keep candidate_result alive longer than necessary.
        del candidate_result

    filtered_text = build_text(kept)

    stats["filtered_num_sentences"] = sum(kept)
    stats["filtered_recall"] = current_result["recall"]
    stats["filtered_mean_entail_prob"] = current_result["mean_entail_prob"]
    stats["removed_sentence_indices"] = sorted(stats["removed_sentence_indices"])

    cleanup_accelerator_memory()

    return filtered_text, stats


def count_jsonl_rows(path: str | Path) -> int:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_valid_jsonl_rows_and_repair_tail(path: str | Path) -> int:
    """
    Count valid JSONL rows in an existing output file.

    If the previous run was killed mid-write, the final line may be partial JSON.
    This function preserves the valid prefix and removes the corrupt tail.
    """
    path = Path(path)

    if not path.exists():
        return 0

    tmp_path = path.with_suffix(path.suffix + ".repairing")

    n_valid = 0
    needs_repair = False

    with open(path, "r", encoding="utf-8") as fin, open(
        tmp_path, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            stripped = line.strip()

            if not stripped:
                needs_repair = True
                continue

            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "Stopping resume scan at invalid JSONL line %s in %s",
                    line_no,
                    path,
                )
                needs_repair = True
                break

            fout.write(stripped + "\n")
            n_valid += 1

    if needs_repair:
        tmp_path.replace(path)
        logger.warning("Repaired %s; kept %s valid rows", path, n_valid)
    else:
        tmp_path.unlink(missing_ok=True)

    return n_valid


def filter_wikidockg_jsonl_file(
        input_path: str | Path,
        output_path: str | Path,
        reranker: Any,
        min_sentences_to_keep: int = 1,
        overwrite: bool = False,
        resume: bool = True,
        max_mean_entail_prob_drop: float = 0.02,
        protect_first_sentence: bool = True,
        score_batch_size: Optional[int] = 4,
        cleanup_every_scores: int = 10,
        max_overlap_tokens: int = 1,
        max_overlap_ratio: float = 0.15,
        max_candidate_sentences: int = 8,
        flush_every: int = 10,
) -> None:
    """
    Reads one WikiDocKG JSONL file and writes/resumes a filtered JSONL file.

    Output row format:
        - keeps all original fields
        - replaces `input_text` with filtered text
        - adds `input_text_original`
        - adds `filter_stats`

    Resume behavior:
        - if output_path exists and overwrite=False and resume=True:
            * scan existing output
            * keep only valid JSONL prefix
            * skip the corresponding number of processable input rows
            * append remaining filtered rows

    Important:
        Existing filtered rows correspond only to processable rows, because invalid
        input rows are skipped and not written. Therefore resume skips by
        processable-row count, not raw input-line count.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def count_valid_existing_output_rows(path: Path) -> int:
        """
        Count valid JSONL rows in an existing output file.

        If the previous run died mid-write, preserve the valid prefix and remove
        the corrupt tail.
        """
        if not path.exists():
            return 0

        tmp_path = path.with_suffix(path.suffix + ".repairing")

        n_valid = 0
        needs_repair = False

        with open(path, "r", encoding="utf-8") as fin, open(
                tmp_path,
                "w",
                encoding="utf-8",
        ) as fout:
            for line_no, line in enumerate(fin, start=1):
                stripped = line.strip()

                if not stripped:
                    needs_repair = True
                    continue

                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        "Stopping resume scan at invalid JSONL line %s in %s",
                        line_no,
                        path,
                    )
                    needs_repair = True
                    break

                fout.write(stripped + "\n")
                n_valid += 1

        if needs_repair:
            tmp_path.replace(path)
            logger.warning("Repaired %s; kept %s valid rows", path, n_valid)
        else:
            tmp_path.unlink(missing_ok=True)

        return n_valid

    if overwrite:
        existing_output_rows = 0
        output_mode = "w"
    elif output_path.exists():
        if not resume:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to replace it "
                f"or resume=True to continue from it."
            )

        existing_output_rows = count_valid_existing_output_rows(output_path)
        output_mode = "a"

        logger.info(
            "Resuming %s from %s existing filtered rows",
            output_path,
            existing_output_rows,
        )
    else:
        existing_output_rows = 0
        output_mode = "w"

    # Best-effort inference/small-batch setup if you added this helper.
    configure_fn = globals().get("configure_reranker_for_filtering")
    if callable(configure_fn):
        reranker = configure_fn(
            reranker,
            score_batch_size=score_batch_size or 1,
        )

    total_rows = count_jsonl_rows(input_path)

    n_total = 0
    n_processable_seen = 0
    n_resume_skipped = 0
    n_written = 0
    n_skipped = 0
    n_errors = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
            output_path,
            output_mode,
            encoding="utf-8",
    ) as fout:
        pbar = tqdm(
            fin,
            total=total_rows,
            desc=f"Filtering {input_path.name}",
        )

        for line in pbar:
            n_total += 1
            raw_line = line
            line = line.strip()

            if not line:
                n_skipped += 1
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSONL input line %s in %s",
                    n_total,
                    input_path,
                )
                n_skipped += 1
                continue

            triples_raw = row.get("triples")
            text = row.get("input_text")

            if not isinstance(triples_raw, list) or not isinstance(text, str):
                n_skipped += 1
                continue

            triples = parse_wikidockg_triples(triples_raw)

            if not triples:
                n_skipped += 1
                continue

            # Resume by processable row count, because only processable rows are
            # written to the filtered output.
            if n_processable_seen < existing_output_rows:
                n_processable_seen += 1
                n_resume_skipped += 1
                continue

            n_processable_seen += 1

            try:
                filtered_text, filter_stats = filter_text_by_factspotter(
                    text=text,
                    triples=triples,
                    reranker=reranker,
                    min_sentences_to_keep=min_sentences_to_keep,
                    max_mean_entail_prob_drop=max_mean_entail_prob_drop,
                    protect_first_sentence=protect_first_sentence,
                    score_batch_size=score_batch_size,
                    cleanup_every_scores=cleanup_every_scores,
                    max_overlap_tokens=max_overlap_tokens,
                    max_overlap_ratio=max_overlap_ratio,
                    max_candidate_sentences=max_candidate_sentences,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.exception(
                        "OOM while filtering line %s in %s; writing original text",
                        n_total,
                        input_path,
                    )

                    cleanup_fn = globals().get("cleanup_accelerator_memory")
                    if callable(cleanup_fn):
                        cleanup_fn()

                    filtered_text = text
                    filter_stats = {
                        "filter_error": "oom",
                        "original_num_sentences": None,
                        "filtered_num_sentences": None,
                        "removed_sentence_indices": [],
                    }
                    n_errors += 1
                else:
                    raise
            except Exception:
                logger.exception(
                    "Error while filtering line %s in %s; writing original text",
                    n_total,
                    input_path,
                )

                filtered_text = text
                filter_stats = {
                    "filter_error": "exception",
                    "original_num_sentences": None,
                    "filtered_num_sentences": None,
                    "removed_sentence_indices": [],
                }
                n_errors += 1

            row["input_text_original"] = text
            row["input_text"] = filtered_text
            row["filter_stats"] = filter_stats
            row["filter_source_line_idx"] = n_total

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

            if flush_every > 0 and n_written % flush_every == 0:
                fout.flush()

            pbar.set_postfix(
                existing=existing_output_rows,
                resumed=n_resume_skipped,
                written=n_written,
                skipped=n_skipped,
                errors=n_errors,
            )

    logger.info(f"Finished filtering {input_path}")
    logger.info(f"Total input rows read:       {n_total}")
    logger.info(f"Processable rows seen:       {n_processable_seen}")
    logger.info(f"Existing filtered rows used: {existing_output_rows}")
    logger.info(f"Resume-skipped rows:         {n_resume_skipped}")
    logger.info(f"New rows written:            {n_written}")
    logger.info(f"Skipped invalid rows:        {n_skipped}")
    logger.info(f"Rows written with errors:    {n_errors}")
    logger.info(f"Output:                      {output_path}")


def build_filtered_wikidockg_files(
        data_dir: str | Path,
        reranker: Any,
        min_sentences_to_keep: int = 1,
        overwrite: bool = False,
        resume: bool = True,
        max_mean_entail_prob_drop: float = 0.02,
        protect_first_sentence: bool = True,
) -> None:
    """
    Creates or resumes:
        wikidockg_train.filtered.jsonl
        wikidockg_test.filtered.jsonl
    from:
        wikidockg_train.jsonl
        wikidockg_test.jsonl
    """
    data_dir = Path(data_dir)

    file_pairs = [
        (
            data_dir / "wikidockg_train.jsonl",
            data_dir / "wikidockg_train.filtered.jsonl",
        ),
        (
            data_dir / "wikidockg_test.jsonl",
            data_dir / "wikidockg_test.filtered.jsonl",
        ),
    ]

    for input_path, output_path in file_pairs:
        filter_wikidockg_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            reranker=reranker,
            min_sentences_to_keep=min_sentences_to_keep,
            overwrite=overwrite,
            resume=resume,
            max_mean_entail_prob_drop=max_mean_entail_prob_drop,
            protect_first_sentence=protect_first_sentence,
        )
