import logging
import random
from typing import List, Dict, Any, Literal

import pandas as pd
from src.data.kgdataset import KGDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class WikiDocKGLoader(KGDataset):
    name = "WikiDocKG"

    FILES = {
        "original": {
            "train": "wikidockg_train.jsonl",
            "test": "wikidockg_test.jsonl",
        },
        "filtered": {
            "train": "wikidockg_train.filtered.jsonl",
            "test": "wikidockg_test.filtered.jsonl",
        },
    }

    def __init__(
            self,
            data_dir: str,
            dev_size: float = 0.01,
            random_state: int = 42,
            data_version: Literal["original", "filtered"] = "filtered",
    ):
        if data_version not in self.FILES:
            raise ValueError(
                f"Unknown data_version={data_version!r}. "
                f"Expected one of {list(self.FILES)}."
            )

        super().__init__(
            data_dir,
            dev_size=dev_size,
            random_state=random_state,
            text_key="text",
            num_nodes_key="num_nodes",

        )

        random.seed(random_state)
        self.data_version = data_version

        self.load_data()

    def _load_data_file(self, filename: str) -> pd.DataFrame:
        file_path = self.DATA_DIR / filename
        logger.info(f"Loading {file_path}...")
        return pd.read_json(
            open(file_path, "r", encoding="utf-8"),
            orient="records",
            lines=True,
        )

    @staticmethod
    def _parse_triples(triples_dict: dict) -> List[Dict[str, str]]:
        def enrich_wqualifiers(object: str, qualifiers: list):
            if not qualifiers:
                return object

            res = object.strip()
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

    @staticmethod
    def _count_unique_nodes(triples: List[Dict[str, str]]) -> int:
        nodes = set()

        for t in triples:
            subj = t.get("subject")
            obj = t.get("object")

            if subj is not None:
                nodes.add(str(subj))
            if obj is not None:
                nodes.add(str(obj))

        return len(nodes)

    def _parse_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        processed_list: List[Dict[str, Any]] = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parse data"):
            triples = row.get("triples")
            text = row.get("input_text")

            if not isinstance(triples, list) or not isinstance(text, str):
                continue

            triples_parsed = self._parse_triples(triples)
            if not triples_parsed:
                continue

            num_nodes = self._count_unique_nodes(triples_parsed)

            sample = {
                "text": text,
                "triples_parsed": triples_parsed,
                "num_nodes": num_nodes,
            }

            # Optional: expose original text if loading filtered files.
            if isinstance(row.get("input_text_original"), str):
                sample["text_original"] = row.get("input_text_original")

            # Optional: expose filtering diagnostics.
            if isinstance(row.get("filter_stats"), dict):
                sample["filter_stats"] = row.get("filter_stats")

            processed_list.append(sample)

        return processed_list

    def load_data(self) -> None:
        files = self.FILES[self.data_version]

        try:
            train_df = self._load_data_file(files["train"])
            test_df = self._load_data_file(files["test"])
        except FileNotFoundError as e:
            logger.error(f"Error loading files: {e}")
            logger.error(
                f"Please ensure {files['train']!r} and {files['test']!r} "
                f"are in the data directory."
            )
            return

        train_data_list = self._parse_data(train_df)
        dev_list, train_list = self.make_dev_split(train_data_list)

        self.data["train"] = train_list
        self.data["dev"] = dev_list
        self.data["test"] = self._parse_data(test_df)

        logger.info(f"WikiDocKG dataset loaded successfully: {self.data_version}")
        logger.info(f"Train samples: {len(self.data['train'])}")
        logger.info(f"Dev samples:   {len(self.data['dev'])}")
        logger.info(f"Test samples:  {len(self.data['test'])}")

    def load_train(self):
        return self.data["train"]

    def load_test_500(self):
        return self.data["test"]
