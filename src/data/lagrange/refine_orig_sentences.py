from pathlib import Path
import json
import logging
import re

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

tqdm.pandas()

DATA_DIR = Path("data/lagrange")
TRAIN_PATH = DATA_DIR / "lagrange_doc_train.jsonl"
TEST_PATH = DATA_DIR / "lagrange_doc_test.jsonl"
TRAIN_OUT_REFINED = DATA_DIR / "lagrange_doc_train_final.jsonl"
TEST_OUT_REFINED = DATA_DIR / "lagrange_doc_test_final.jsonl"

EDIT_DISTANCE_LOWER_THRESHOLD = 0.9
CONTRADICTION_HIGHER_THRESHOLD = 0.2  # probability in [0, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# Resolve label ids from the model config instead of assuming order.
id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
contradiction_idx = next(idx for idx, label in id2label.items() if label == "contradiction")

# MediaWiki / Wikipedia markup cleanup
WIKI_HEADING_RE = re.compile(r"(?:(?<=^)|(?<=\n))\s*={2,}\s*[^=\n]+?\s*={2,}\s*(?=\n|$)")
INLINE_WIKI_HEADING_RE = re.compile(r"\s*={2,}\s*[^=\n]+?\s*={2,}\s*")
WIKI_MAGIC_WORD_RE = re.compile(r"__(?:TOC|NOTOC|FORCETOC|NOEDITSECTION)__", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def clean_wiki_sentence(text: str) -> str:
    """
    Remove common Wikipedia / MediaWiki artifacts from a sentence-like field.
    Examples removed:
      - == Biography ==
      - === Later career and death ===
      - __TOC__
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = WIKI_MAGIC_WORD_RE.sub(" ", text)
    text = WIKI_HEADING_RE.sub(" ", text)
    text = INLINE_WIKI_HEADING_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


@torch.no_grad()
def check_contradiction(row: pd.Series) -> float:
    """
    Return contradiction probability in [0, 1].
    """
    if row["match_score"] == 1.0:
        return 0.0

    if row["match_score"] < EDIT_DISTANCE_LOWER_THRESHOLD:
        return 1.0

    encoded = tokenizer(
        row["sentence"],
        row["wiki_sentence"],
        truncation=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[contradiction_idx].item())


def define_final_sentence(row: pd.Series) -> str:
    wiki_sentence = row["wiki_sentence"]

    if not wiki_sentence:
        return row["sentence"]

    if row["match_score"] == 1.0:
        return row["sentence"]

    if (
        row["match_score"] >= EDIT_DISTANCE_LOWER_THRESHOLD
        and row["contradiction"] < CONTRADICTION_HIGHER_THRESHOLD
    ):
        return wiki_sentence

    return row["sentence"]


def process_split(input_path: Path, output_path: Path, split_name: str) -> None:
    logger.info("Loading %s data from %s", split_name, input_path)
    df = pd.read_json(input_path, orient="records", lines=True)

    original_wiki = df["wiki_sentence"].copy()
    df["wiki_sentence"] = df["wiki_sentence"].map(clean_wiki_sentence)

    cleaned_rows = (original_wiki.fillna("") != df["wiki_sentence"].fillna("")).sum()
    logger.info("Cleaned Wikipedia artifacts in %s %s rows", cleaned_rows, split_name)

    df["contradiction"] = df.progress_apply(check_contradiction, axis=1)
    df["final_sentence"] = df.apply(define_final_sentence, axis=1)

    logger.info("Writing refined %s data → %s", split_name, output_path)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)


def main() -> None:
    process_split(TRAIN_PATH, TRAIN_OUT_REFINED, "TRAIN")
    process_split(TEST_PATH, TEST_OUT_REFINED, "TEST")


if __name__ == "__main__":
    main()