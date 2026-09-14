from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


INPUT_PATH = "data/wikidockg/wikidockg_df_filtered2.parquet"
OUTPUT_PATH = "data/wikidockg/wikidockg_df_topics.parquet"


TEXT_COL = "text"
CLASSIFY_TEXT_COL = "_classification_text"


# Short label for plotting / distributional figure
TOPIC_COL = "topic_label"
TOPIC_NLI_COL = "topic_nli_label"
TOPIC_SOURCE_COL = "topic_source"
SCORE_COL = "topic_score"


MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# For English-only Wikipedia, this may classify better but is heavier:
# MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

MAX_CHARS = 2048
BATCH_SIZE = 16

# Important:
# True = score each label independently against contradiction.
# Usually better for overlapping labels.
MULTI_LABEL = True

HYPOTHESIS_TEMPLATE = "This Wikipedia article is {}."

LABEL_SPECS = [
    ("about a person's life or biography", "Biography/person"),

    (
        "about an organization, institution, company, society, association, agency, business, or economic organization",
        "Organization/institution",
    ),

    (
        "about a geographic place, region, settlement, landmark, or building",
        "Place/building",
    ),

    (
        "about an animal, plant, fungus, microorganism, or other biological species",
        "Biology/species",
    ),

    (
        "about a historical event, period, empire, dynasty, civilization, war, battle, or armed conflict",
        "History/event",
    ),

    (
        "about politics, government, law, society, culture, customs, education, or social issues",
        "Society/politics/culture",
    ),

    (
        "about religion, religious texts, religious organizations, mythology, theology, or philosophy",
        "Religion/philosophy",
    ),

    (
        "about a language, ethnic group, nationality, or demographic group",
        "Ethnicity/language",
    ),

    (
        "about a book, literary work, poem, journal, newspaper, magazine, or literature",
        "Books/literature",
    ),

    (
        "about music, film, television, theatre, visual art, video games, or entertainment",
        "Arts/entertainment",
    ),

    (
        "about a sport, game, athlete, sports team, tournament, or recreational activity",
        "Sports/games",
    ),

    (
        "about science, mathematics, physics, chemistry, astronomy, technology, engineering, computers, or software",
        "Science/technology",
    ),
]

TOPIC_LABELS = [nli_label for nli_label, _ in LABEL_SPECS]
DISPLAY_LABELS = dict(LABEL_SPECS)
DISPLAY_TO_NLI = {
    display_label: nli_label
    for nli_label, display_label in LABEL_SPECS
}

RULES = [
    # Biological taxa / species
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(species|subspecies|genus|family|order|clade|taxon|plant|animal|fungus|bacterium|virus)\b",
            re.I,
        ),
        "Biology/species",
    ),

    # Organizations / institutions / businesses
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(non-profit|nonprofit|organization|institution|association|society|agency|company|corporation|"
            r"university|college|school|museum|foundation|charity|institute|committee|council|club|union|"
            r"business|enterprise|bank|publisher)\b",
            re.I,
        ),
        "Organization/institution",
    ),

    # History
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(battle|war|siege|revolt|rebellion|revolution|campaign|conflict|"
            r"invasion|uprising|massacre|military operation|armed conflict)\b",
            re.I,
        ),
        "History/event",
    ),

    # Politics / society / culture
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(law|election|government|political party|policy|constitution|court|parliament|senate|"
            r"culture|custom|education|social movement|social issue|civil rights)\b",
            re.I,
        ),
        "Society/politics/culture",
    ),

    # Books / literature / periodicals
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(book|novel|poem|short story|literary work|journal|newspaper|magazine|periodical|publication)\b",
            re.I,
        ),
        "Books/literature",
    ),

    # Arts / entertainment
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(song|album|film|movie|television series|tv series|painting|opera|play|musical|"
            r"band|orchestra|video game|artwork)\b",
            re.I,
        ),
        "Arts/entertainment",
    ),

    # Sports
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(sport|game|team|club|league|tournament|championship|footballer|cricketer|basketball player|"
            r"baseball player|tennis player|athlete)\b",
            re.I,
        ),
        "Sports/games",
    ),

    # Languages / ethnic groups
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(language|dialect|ethnic group|people|tribe|nationality)\b",
            re.I,
        ),
        "Ethnicity/language",
    ),

    # Religion / philosophy
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(religion|religious|mythology|mythological|philosophy|philosophical|theology|"
            r"denomination|sect|scripture|prayer book|bible)\b",
            re.I,
        ),
        "Religion/philosophy",
    ),

    # Science / technology
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(science|scientific|mathematics|physics|chemistry|astronomy|engineering|technology|"
            r"software|computer|algorithm|programming language|machine|device|invention)\b",
            re.I,
        ),
        "Science/technology",
    ),

    # Places / buildings
    (
        re.compile(
            r"\b(is|was|are|were)\s+(an?|the)?\s*.*\b"
            r"(village|town|city|municipality|province|district|county|region|state|country|"
            r"river|mountain|island|lake|valley|building|church|cathedral|castle|temple|mosque|synagogue|monument)\b",
            re.I,
        ),
        "Place/building",
    ),
]


def load_dataframe(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path_obj)
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path_obj)

    raise ValueError(f"Unsupported input format: {path}")


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    suffix = path_obj.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path_obj, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path_obj, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path_obj)
        return

    raise ValueError(f"Unsupported output format: {path}")


def first_sentences(text: str, max_sentences: int = 3) -> str:
    text = " ".join(str(text).split())
    if not text:
        return ""

    # Simple sentence splitter good enough for Wikipedia leads.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:max_sentences])


def build_classification_text(row) -> str:
    title = str(row.get("title", "")).strip()
    text = first_sentences(row.get(TEXT_COL, ""), max_sentences=3)

    if title and text:
        return f"Title: {title}. Article lead: {text}"
    if text:
        return text
    return title

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""

    text = str(x)
    text = " ".join(text.split())
    return text[:MAX_CHARS]


def rule_based_label(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None

    # Important: rules should inspect the definitional sentence,
    # not all contextual details in the lead.
    sent = first_sentences(text, max_sentences=2)

    for pattern, display_label in RULES:
        if pattern.search(sent):
            return display_label

    return None


def build_classifier():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )


def classify_topics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Build the shorter classification text: title + first lead sentences.
    df[CLASSIFY_TEXT_COL] = df.apply(build_classification_text, axis=1)

    df[TOPIC_COL] = pd.NA
    df[TOPIC_NLI_COL] = pd.NA
    df[SCORE_COL] = pd.NA
    df[TOPIC_SOURCE_COL] = pd.NA

    nonempty_mask = df[CLASSIFY_TEXT_COL].fillna("").str.len().gt(0)

    if not nonempty_mask.any():
        return df

    rule_labels = df.loc[nonempty_mask, CLASSIFY_TEXT_COL].map(rule_based_label)
    rule_mask = nonempty_mask.copy()
    rule_mask.loc[nonempty_mask] = rule_labels.notna()

    if rule_mask.any():
        df.loc[rule_mask, TOPIC_COL] = rule_labels.loc[rule_mask]
        df.loc[rule_mask, TOPIC_NLI_COL] = df.loc[rule_mask, TOPIC_COL].map(DISPLAY_TO_NLI)
        df.loc[rule_mask, SCORE_COL] = pd.NA
        df.loc[rule_mask, TOPIC_SOURCE_COL] = "rule"

    zs_mask = nonempty_mask & df[TOPIC_COL].isna()

    if not zs_mask.any():
        return df

    work_df = df.loc[zs_mask, [CLASSIFY_TEXT_COL]].copy()
    ds = Dataset.from_pandas(work_df, preserve_index=False)

    clf = build_classifier()

    outputs_iter = clf(
        KeyDataset(ds, CLASSIFY_TEXT_COL),
        candidate_labels=TOPIC_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=MULTI_LABEL,
        batch_size=BATCH_SIZE,
    )

    top_nli_labels: List[str] = []
    top_display_labels: List[str] = []
    top_scores: List[float] = []

    for out in tqdm(outputs_iter, total=len(work_df), desc="Zero-shot topic fallback"):
        nli_label = out["labels"][0]
        score = float(out["scores"][0])

        top_nli_labels.append(nli_label)
        top_display_labels.append(DISPLAY_LABELS[nli_label])
        top_scores.append(score)

    if len(top_nli_labels) != len(work_df):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(top_nli_labels):,}, "
            f"expected {len(work_df):,}."
        )

    df.loc[zs_mask, TOPIC_NLI_COL] = top_nli_labels
    df.loc[zs_mask, TOPIC_COL] = top_display_labels
    df.loc[zs_mask, SCORE_COL] = top_scores
    df.loc[zs_mask, TOPIC_SOURCE_COL] = "zero_shot"

    return df


def print_distribution(df: pd.DataFrame) -> None:
    dist = (
        df[TOPIC_COL]
        .value_counts(dropna=False)
        .rename_axis("topic")
        .reset_index(name="n")
    )

    dist["pct"] = 100 * dist["n"] / dist["n"].sum()

    print("\nTopic distribution:")
    print(dist.to_string(index=False, formatters={"pct": "{:.2f}%".format}))


def print_sample(df: pd.DataFrame, n: int = 10) -> None:
    cols = [col for col in ["title", "url", TOPIC_COL, TOPIC_NLI_COL, SCORE_COL] if col in df.columns]

    print("\nSample predictions:")
    print(df[cols].sample(min(n, len(df)), random_state=42).to_string(index=False))


def print_low_confidence_sample(df: pd.DataFrame, n: int = 10) -> None:
    if SCORE_COL not in df.columns:
        return

    scored = df.dropna(subset=[SCORE_COL]).copy()
    if scored.empty:
        return

    scored[SCORE_COL] = scored[SCORE_COL].astype(float)

    cols = [col for col in ["title", "url", TOPIC_COL, TOPIC_NLI_COL, SCORE_COL] if col in scored.columns]

    print("\nLowest-confidence predictions:")
    print(
        scored
        .sort_values(SCORE_COL, ascending=True)
        .head(n)[cols]
        .to_string(index=False)
    )


def main() -> None:
    df = load_dataframe(INPUT_PATH)

    if TEXT_COL not in df.columns:
        raise KeyError(f"Column '{TEXT_COL}' not found in dataframe.")

    df_topics = classify_topics(df)
    save_dataframe(df_topics, OUTPUT_PATH)

    print(f"\nSaved {len(df_topics):,} rows to {OUTPUT_PATH}")

    print_distribution(df_topics)
    print_sample(df_topics, n=10)
    print_low_confidence_sample(df_topics, n=10)


if __name__ == "__main__":
    main()
