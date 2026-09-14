import random
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm.auto import tqdm

from src.data.kgdataset import KGDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class LagrangeLoader(KGDataset):
    """
    Loads and processes the LAGRANGE-formatted dataset.

    Supports optional stratified clustering-based subsampling of the
    training and test splits, using `num_nodes` buckets and semantic
    clustering within each bucket.
    """

    name = "LAGRANGE"

    def __init__(
        self,
        data_dir: str,
        dev_size: float = 0.1,
        random_state: int = 42,
        create_train_subset: bool = False,
        train_subset_size: int = 50_000,
        train_subset_name: Optional[str] = None,
        save_train_subset_indices: bool = True,
        create_test_subset: bool = False,
        test_subset_size: int = 500,
        test_subset_name: Optional[str] = None,
        save_test_subset_indices: bool = True,
    ):
        super().__init__(
            data_dir,
            dev_size=dev_size,
            random_state=random_state,
            text_key="text",
            num_nodes_key="num_nodes",

        )
        random.seed(random_state)

        self.create_train_subset = create_train_subset
        self.train_subset_size = train_subset_size
        self.train_subset_name = train_subset_name
        self.save_train_subset_indices = save_train_subset_indices

        self.train_subset: Optional[List[Dict[str, Any]]] = None
        self.train_subset_indices: Optional[List[int]] = None
        self.train_subset_index_path: Optional[Path] = None

        self.create_test_subset = create_test_subset
        self.test_subset_size = test_subset_size
        self.test_subset_name = test_subset_name
        self.save_test_subset_indices = save_test_subset_indices

        self.test_subset: Optional[List[Dict[str, Any]]] = None
        self.test_subset_indices: Optional[List[int]] = None
        self.test_subset_index_path: Optional[Path] = None

        self.load_data()

        if self.create_train_subset:
            self._build_train_subset()

        if self.create_test_subset:
            self._build_test_subset()

    def _load_data_file(self, filename: str) -> pd.DataFrame:
        file_path = self.DATA_DIR / filename
        logger.info(f"Loading {file_path}...")
        return pd.read_json(
            open(file_path, "r", encoding="utf-8"),
            orient="records",
            lines=True,
        )

    @staticmethod
    def _parse_triples(triples_str: str) -> List[Dict[str, str]]:
        """
        Converts the triple string into a list of
        {"subject": ..., "predicate": ..., "object": ...}
        dictionaries, handling the '<sep>' delimiter.
        """
        parsed_triples: List[Dict[str, str]] = []

        individual_triple_strings = triples_str.split("<sep>")
        pattern = r"<S>(.*?)<P>(.*?)<O>(.*?)$"

        for triple_str in individual_triple_strings:
            triple_str = triple_str.strip()
            if not triple_str:
                continue

            matches = re.findall(pattern, triple_str)
            for subj, pred, obj in matches:
                parsed_triples.append(
                    {
                        "subject": subj.strip(),
                        "predicate": pred.strip(),
                        "object": obj.strip(),
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
        """
        Processes a DataFrame row-by-row into the required compact list format.
        """
        processed_list: List[Dict[str, Any]] = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Parse data"):
            triples_text = row.get("triples")
            sentence = row.get("sentence")

            if not isinstance(triples_text, str) or not isinstance(sentence, str):
                continue

            triples_parsed = self._parse_triples(triples_text)
            if not triples_parsed:
                continue

            num_nodes = self._count_unique_nodes(triples_parsed)

            sample = {
                "text": sentence,
                "triples_text": triples_text,
                "triples_parsed": triples_parsed,
                "num_nodes": num_nodes,
            }
            processed_list.append(sample)

        return processed_list

    def _build_train_subset(self) -> None:
        subset_name = self.train_subset_name
        if subset_name is None:
            subset_name = (
                f"lagrange_train_stratified_num_nodes_"
                f"n{self.train_subset_size}_seed{self.random_state}"
            )

        subset, indices, saved_path = self.cluster_and_sample_split_stratified(
            split="train",
            total_samples=self.train_subset_size,
            stratify_by="num_nodes",
            subset_name=subset_name,
            save_indices=self.save_train_subset_indices,
            extra_metadata={
                "dataset": "LAGRANGE",
                "purpose": "training subset",
                "full_train_split_available": True,
            },
        )

        self.train_subset = subset
        self.train_subset_indices = indices
        self.train_subset_index_path = saved_path

        logger.info(
            "Created LAGRANGE stratified training subset: %d samples",
            len(subset),
        )
        if saved_path is not None:
            logger.info("Saved LAGRANGE train subset indices to: %s", saved_path)

    def _build_test_subset(self) -> None:
        subset_name = self.test_subset_name
        if subset_name is None:
            subset_name = (
                f"lagrange_test_stratified_num_nodes_"
                f"n{self.test_subset_size}_seed{self.random_state}"
            )

        subset, indices, saved_path = self.cluster_and_sample_split_stratified(
            split="test",
            total_samples=self.test_subset_size,
            stratify_by="num_nodes",
            subset_name=subset_name,
            save_indices=self.save_test_subset_indices,
            extra_metadata={
                "dataset": "LAGRANGE",
                "purpose": "evaluation subset",
                "full_test_split_available": True,
            },
        )

        self.test_subset = subset
        self.test_subset_indices = indices
        self.test_subset_index_path = saved_path

        logger.info(
            "Created LAGRANGE stratified test subset: %d samples",
            len(subset),
        )
        if saved_path is not None:
            logger.info("Saved LAGRANGE test subset indices to: %s", saved_path)

    def get_train_split(self, full: bool = False) -> List[Dict[str, Any]]:
        if full:
            return self.data["train"]
        if self.train_subset is None:
            raise ValueError(
                "Train subset has not been created. Initialize with "
                "`create_train_subset=True`."
            )
        return self.train_subset

    def get_eval_split(self, full: bool = False) -> List[Dict[str, Any]]:
        if full:
            return self.data["test"]
        if self.test_subset is None:
            raise ValueError(
                "Test subset has not been created. Initialize with "
                "`create_test_subset=True`."
            )
        return self.test_subset

    def load_data(self) -> None:
        """
        Implements the data loading and splitting for the LAGRANGE format.
        """
        try:
            train_df = self._load_data_file("lagrange_train.json")
            test_df = self._load_data_file("lagrange_test.json")
        except FileNotFoundError as e:
            logger.error(f"Error loading files: {e}")
            logger.error(
                "Please ensure 'lagrange_train.json' and "
                "'lagrange_test.json' are in the data directory."
            )
            return

        train_data_list = self._parse_data(train_df)
        dev_list, train_list = self.make_dev_split(train_data_list)

        self.data["train"] = train_list
        self.data["dev"] = dev_list
        self.data["test"] = self._parse_data(test_df)

        logger.info("LAGRANGE dataset loaded successfully.")
        logger.info(f"Train samples: {len(self.data['train'])}")
        logger.info(f"Dev samples:   {len(self.data['dev'])}")
        logger.info(f"Test samples:  {len(self.data['test'])}")

    def load_train(self):
        train_subset, _, _, = super().load_subset_indices("lagrange_train_stratified_num_nodes_n50000_seed42")
        return train_subset

    def load_test_500(self):
        test_subset, _, _, = super().load_subset_indices("lagrange_test_stratified_num_nodes_n500_seed42")
        return test_subset
