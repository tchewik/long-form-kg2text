import logging
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.kgdataset import KGDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class WebNLGLoader(KGDataset):
    name = "WebNLG"

    def __init__(
        self,
        data_dir: str,
        dev_size: float = 0.1,
        random_state: int = 42,
        language: str = "en",
        use_modified_triples: bool = True,
        filter_good_only: bool = True,
        create_test_subset: bool = False,
        test_subset_size: int = 400,
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

        self.language = language
        self.use_modified_triples = use_modified_triples
        self.filter_good_only = filter_good_only

        self.create_test_subset = create_test_subset
        self.test_subset_size = test_subset_size
        self.test_subset_name = test_subset_name
        self.save_test_subset_indices = save_test_subset_indices

        self.test_subset: Optional[List[Dict[str, Any]]] = None
        self.test_subset_indices: Optional[List[int]] = None
        self.test_subset_index_path: Optional[Path] = None

        self.load_data()

        if self.create_test_subset:
            self._build_test_subset()

    def _build_test_subset(self) -> None:
        subset_name = self.test_subset_name
        if subset_name is None:
            subset_name = (
                f"webnlg_{self.language}_test_stratified_num_nodes_"
                f"n{self.test_subset_size}_seed{self.random_state}"
            )

        subset, indices, saved_path = self.cluster_and_sample_split_stratified(
            split="test",
            total_samples=self.test_subset_size,
            stratify_by="num_nodes",
            subset_name=subset_name,
            save_indices=self.save_test_subset_indices,
            extra_metadata={
                "dataset": "WebNLG",
                "language": self.language,
                "use_modified_triples": self.use_modified_triples,
                "filter_good_only": self.filter_good_only,
                "purpose": "main evaluation subset",
                "full_test_split_available": True,
            },
        )

        self.test_subset = subset
        self.test_subset_indices = indices
        self.test_subset_index_path = saved_path

        logger.info(
            "Created WebNLG stratified test subset: %d samples",
            len(subset),
        )
        if saved_path is not None:
            logger.info("Saved WebNLG test subset indices to: %s", saved_path)

    def get_eval_split(self, full: bool = False) -> List[Dict[str, Any]]:
        """
        Return either the full test split or the clustered evaluation subset.
        """
        if full:
            return self.data["test"]

        if self.test_subset is None:
            raise ValueError(
                "Test subset has not been created. Initialize with "
                "`create_test_subset=True` or call `_build_test_subset()`."
            )
        return self.test_subset

    @staticmethod
    def _count_unique_nodes(triples: List[Dict[str, str]]) -> int:
        nodes = set()
        for t in triples:
            if "subject" in t and t["subject"] is not None:
                nodes.add(str(t["subject"]))
            if "object" in t and t["object"] is not None:
                nodes.add(str(t["object"]))
        return len(nodes)

    def _resolve_language_dir(self) -> Path:
        base = self.DATA_DIR

        if (base / "release_v3.0").exists():
            base = base / "release_v3.0"

        lang_dir = base / self.language

        if not lang_dir.exists():
            raise FileNotFoundError(
                f"Could not find language directory '{lang_dir}'. "
                "Expected something like '.../release_v3.0/en/'."
            )

        return lang_dir

    @staticmethod
    def _parse_mtriple_text(text: Optional[str]) -> Optional[Dict[str, str]]:
        if not text:
            return None
        parts = [p.strip() for p in text.split("|")]
        if len(parts) != 3:
            return None
        subj, pred, obj = parts
        return {
            "subject": subj.replace("_", " "),
            "predicate": pred,
            "object": obj.replace("_", " "),
        }

    def _extract_triples_from_entry(self, entry: ET.Element) -> List[Dict[str, str]]:
        triples: List[Dict[str, str]] = []

        if self.use_modified_triples:
            tset = entry.find("modifiedtripleset")
            tag = "mtriple"
        else:
            tset = entry.find("originaltripleset")
            tag = "otriple"

        if tset is None:
            return triples

        for t in tset.iter(tag):
            triple = self._parse_mtriple_text(t.text)
            if triple is not None:
                triples.append(triple)

        return triples

    @staticmethod
    def _format_triples_text(triples: List[Dict[str, str]]) -> str:
        triple_strings: List[str] = []
        for t in triples:
            triple_strings.append(
                f"{t['subject']} | {t['predicate']} | {t['object']}"
            )
        return " <sep> ".join(triple_strings)

    def _select_lex_elements(
        self, entry: ET.Element, split: str
    ) -> List[ET.Element]:
        lex_elems = entry.findall("lex")

        if split == "test":
            return lex_elems

        if not self.filter_good_only:
            return lex_elems

        good_lexes = [
            lex
            for lex in lex_elems
            if str(lex.attrib.get("comment", "")).lower() == "good"
        ]

        return good_lexes or lex_elems

    def _parse_split_dir(self, split_dir: Path, split_name: str) -> List[Dict[str, Any]]:
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        xml_files = sorted(split_dir.glob("**/*.xml"))
        if not xml_files:
            logger.warning(f"No XML files found under {split_dir}")

        samples: List[Dict[str, Any]] = []

        logger.info(
            f"Parsing WebNLG v3.0 {split_name} from {split_dir} "
            f"({len(xml_files)} XML files)..."
        )

        for xml_path in tqdm(xml_files, desc=f"Parse WebNLG v3.0 {split_name}"):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError as e:
                logger.error(f"Failed to parse XML file '{xml_path}': {e}")
                continue

            root = tree.getroot()
            for entry in root.iter("entry"):
                category = entry.attrib.get("category")
                eid = entry.attrib.get("eid")
                shape = entry.attrib.get("shape")
                shape_type = entry.attrib.get("shape_type")
                size_attr = entry.attrib.get("size")

                triples = self._extract_triples_from_entry(entry)
                if not triples:
                    continue

                triples_text = self._format_triples_text(triples)

                try:
                    size = int(size_attr) if size_attr is not None else len(triples)
                except (TypeError, ValueError):
                    size = len(triples)

                num_nodes = self._count_unique_nodes(triples)

                lex_elems = self._select_lex_elements(entry, split_name)

                for lex in lex_elems:
                    text = (lex.text or "").strip()
                    if not text:
                        continue

                    sample: Dict[str, Any] = {
                        "text": text,
                        "triples_text": triples_text,
                        "triples_parsed": triples,
                        "category": category,
                        "entry_id": eid,
                        "shape": shape,
                        "shape_type": shape_type,
                        "size": size,
                        "num_nodes": num_nodes,
                        "xml_file": str(xml_path),
                        "split": split_name,
                    }
                    samples.append(sample)

                    break   # Use only the first lexicalization

        logger.info(
            f"Parsed {len(samples)} samples for WebNLG v3.0 {split_name} split."
        )
        return samples

    def load_data(self) -> None:
        try:
            lang_dir = self._resolve_language_dir()
        except FileNotFoundError as e:
            logger.error(e)
            logger.error(
                "Please make sure 'data_dir' points to either "
                "'.../release_v3.0' or '.../release_v3.0/en'."
            )
            return

        splits = ("train", "dev", "test")
        for split in splits:
            split_dir = lang_dir / split
            try:
                split_samples = self._parse_split_dir(split_dir, split)
            except FileNotFoundError as e:
                logger.error(e)
                logger.error(
                    f"Expected split directory '{split_dir}' "
                    f"for WebNLG v3.0 {split} split."
                )
                return

            self.data[split] = split_samples

        logger.info("WebNLG v3.0 dataset loaded successfully.")
        logger.info(f"Train samples: {len(self.data['train'])}")
        logger.info(f"Dev samples:   {len(self.data['dev'])}")
        logger.info(f"Test samples:  {len(self.data['test'])}")

    def load_train(self):
        return self.data["train"]

    def load_test_500(self):
        test_subset, _, _, = super().load_subset_indices("webnlg_en_test_stratified_num_nodes_n500_seed42")
        return test_subset
