from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import math
import random
import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class KGDataset(ABC):
    DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self, data_dir: str, **kwargs):
        self.DATA_DIR = Path(data_dir)
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.kwargs = kwargs

        self.text_key: str = kwargs.get("text_key", "verbalization")
        self.num_nodes_key: str = kwargs.get("num_nodes_key", "num_nodes")
        self.embedding_model_name: str = kwargs.get(
            "embedding_model_name",
            self.DEFAULT_EMBED_MODEL,
        )
        self.embedding_batch_size: int = kwargs.get("embedding_batch_size", 64)
        self.kmeans_backend = kwargs.get("kmeans_backend", "minibatch")
        self.kmeans_max_iter: int = kwargs.get("kmeans_max_iter", 300)
        self.kmeans_n_init: int = kwargs.get("kmeans_n_init", 10)
        self.random_state: int = kwargs.get("random_state", 42)

        self.indices_dir: Path = Path(
            kwargs.get("indices_dir", self.DATA_DIR / "subset_indices")
        )
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = None

        # NEW: logger setup
        self.logger = kwargs.get("logger", logging.getLogger(self.__class__.__name__))
        self.enable_bucket_logging = kwargs.get("enable_bucket_logging", True)

    @abstractmethod
    def load_data(self) -> None:
        pass

    def make_dev_split(
            self,
            train_data_list: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_data_list = list(train_data_list)

        rng = random.Random(self.random_state)
        rng.shuffle(train_data_list)

        dev_size = self.kwargs["dev_size"]
        total_train_size = len(train_data_list)
        split_idx = int(total_train_size * dev_size)

        dev_list = train_data_list[:split_idx]
        train_list = train_data_list[split_idx:]

        return dev_list, train_list

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        if split not in self.data:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'.")
        return self.data[split]

    def _load_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for text embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from e

            model_source = self.kwargs.get("embedding_model_path", self.embedding_model_name)

            self._embedder = SentenceTransformer(
                model_source,
                trust_remote_code=True,
                local_files_only=self.kwargs.get("local_files_only", False),
                cache_folder=self.kwargs.get("hf_cache_dir", None),
            )
        return self._embedder

    def _get_text(self, sample: Dict[str, Any]) -> str:
        if self.text_key not in sample:
            raise KeyError(f"Missing text field '{self.text_key}' in sample.")
        text = sample[self.text_key]
        return text if isinstance(text, str) else str(text)

    def _get_num_nodes(self, sample: Dict[str, Any]) -> int:
        if self.num_nodes_key not in sample:
            raise KeyError(f"Missing node-count field '{self.num_nodes_key}' in sample.")
        return int(sample[self.num_nodes_key])

    def build_text_representations(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        if not samples:
            raise ValueError("Cannot build representations for an empty sample list.")

        texts = [self._get_text(s) for s in samples]

        embedder = self._load_embedder()
        text_emb = embedder.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        reps = normalize(text_emb, norm="l2")
        return reps.astype(np.float32)

    def cluster_samples(self, samples: List[Dict[str, Any]], k: int):
        reps = self.build_text_representations(samples)

        if self.kmeans_backend == "sklearn":
            from sklearn.cluster import KMeans
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
                random_state=self.random_state,
            )
            labels = model.fit_predict(reps)
            return labels, model

        elif self.kmeans_backend == "minibatch":
            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                random_state=self.random_state,
                batch_size=self.kwargs.get("kmeans_batch_size", 4096),
                max_iter=self.kmeans_max_iter,
                n_init="auto",
            )
            labels = model.fit_predict(reps)
            return labels, model

        else:
            raise ValueError(f"Unknown kmeans_backend: {self.kmeans_backend}")

    def _group_indices_by_cluster(self, labels: np.ndarray) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)
        return clusters

    def _compute_proportional_cluster_quotas(
            self,
            group_sizes: Dict[Any, int],
            target_total: int,
            *,
            min_per_nonempty_group: int = 0,
    ) -> Dict[Any, int]:
        total_size = sum(group_sizes.values())
        if target_total <= 0:
            raise ValueError("target_total must be > 0")
        if target_total > total_size:
            raise ValueError(
                f"target_total={target_total} exceeds available samples={total_size}"
            )

        nonempty_groups = [g for g, sz in group_sizes.items() if sz > 0]

        quotas = {g: 0 for g in group_sizes}

        # Mandatory floor for rare buckets
        if min_per_nonempty_group > 0:
            required = min_per_nonempty_group * len(nonempty_groups)
            if required > target_total:
                raise ValueError(
                    f"target_total={target_total} is too small for "
                    f"min_per_nonempty_group={min_per_nonempty_group}"
                )
            for g in nonempty_groups:
                quotas[g] = min(min_per_nonempty_group, group_sizes[g])

            remaining_target = target_total - sum(quotas.values())
        else:
            remaining_target = target_total

        remaining_sizes = {g: group_sizes[g] - quotas[g] for g in group_sizes}
        remaining_total = sum(remaining_sizes.values())

        if remaining_target == 0:
            return quotas

        raw = {
            g: (remaining_sizes[g] / remaining_total) * remaining_target
            if remaining_total > 0 else 0.0
            for g in group_sizes
        }

        for g in group_sizes:
            quotas[g] += min(remaining_sizes[g], int(math.floor(raw[g])))

        assigned = sum(quotas.values())
        remainder = target_total - assigned

        fractional_order = sorted(
            group_sizes.keys(),
            key=lambda g: (raw[g] - math.floor(raw[g])),
            reverse=True,
        )

        i = 0
        while remainder > 0:
            g = fractional_order[i % len(fractional_order)]
            if quotas[g] < group_sizes[g]:
                quotas[g] += 1
                remainder -= 1
            i += 1

        return quotas

    def sample_from_clusters(
        self,
        samples: List[Dict[str, Any]],
        labels: np.ndarray,
        *,
        samples_per_cluster: int = 1,
        seed: Optional[int] = None,
        attach_cluster_id: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        rng = random.Random(self.random_state if seed is None else seed)
        clusters = self._group_indices_by_cluster(labels)
        selected_indices: List[int] = []

        for c, idxs in clusters.items():
            q = min(samples_per_cluster, len(idxs))
            selected_indices.extend(rng.sample(idxs, k=q))

        rng.shuffle(selected_indices)

        subset = []
        for idx in selected_indices:
            item = dict(samples[idx])
            if attach_cluster_id:
                item["cluster_id"] = int(labels[idx])
            item["source_idx"] = int(idx)
            subset.append(item)

        return subset, selected_indices

    def save_subset_indices(
        self,
        indices: List[int],
        *,
        split: str,
        subset_name: str,
        seed: Optional[int],
        sampling_mode: str,
        total_samples: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out_path = self.indices_dir / f"{subset_name}.json"

        payload: Dict[str, Any] = {
            "split": split,
            "subset_name": subset_name,
            "num_selected": len(indices),
            "indices": [int(i) for i in indices],
            "seed": self.random_state if seed is None else seed,
            "sampling_mode": sampling_mode,
            "total_samples": total_samples,
            "embedding_model": self.embedding_model_name,
            "text_key": self.text_key,
            "num_nodes_key": self.num_nodes_key,
            "kmeans_init": "k-means++",
            "kmeans_n_init": self.kmeans_n_init,
            "kmeans_max_iter": self.kmeans_max_iter,
        }

        if extra_metadata:
            payload["extra_metadata"] = extra_metadata

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return out_path

    def cluster_and_sample_split_stratified(
            self,
            split: str,
            *,
            total_samples: int,
            stratify_by: str = "num_nodes",
            subset_name: Optional[str] = None,
            seed: Optional[int] = None,
            save_indices: bool = True,
            extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[int], Optional[Path]]:
        samples = self.get_split(split)
        if total_samples <= 0:
            raise ValueError("total_samples must be > 0")
        if total_samples > len(samples):
            raise ValueError(
                f"total_samples={total_samples} exceeds split size={len(samples)}"
            )

        rng = random.Random(self.random_state if seed is None else seed)

        buckets: Dict[Any, List[int]] = {}
        for idx, sample in enumerate(samples):
            if stratify_by not in sample:
                raise KeyError(f"Sample is missing stratification key '{stratify_by}'.")
            bucket_value = sample[stratify_by]
            buckets.setdefault(bucket_value, []).append(idx)

        bucket_sizes = {b: len(idxs) for b, idxs in buckets.items()}
        nonempty_buckets = [b for b, size in bucket_sizes.items() if size > 0]

        if total_samples < len(nonempty_buckets):
            raise ValueError(
                f"total_samples={total_samples} is smaller than the number of non-empty "
                f"buckets={len(nonempty_buckets)}; cannot guarantee one mandatory sample "
                f"per non-empty bucket."
            )

        bucket_quotas: Dict[Any, int] = {b: 0 for b in buckets}

        # Mandatory coverage
        for b in nonempty_buckets:
            bucket_quotas[b] = 1

        remaining_budget = total_samples - len(nonempty_buckets)

        if remaining_budget > 0:
            residual_sizes = {
                b: bucket_sizes[b] - bucket_quotas[b]
                for b in buckets
            }
            residual_total = sum(residual_sizes.values())

            if residual_total > 0:
                raw_extra = {
                    b: (residual_sizes[b] / residual_total) * remaining_budget
                    for b in buckets
                }

                extra_floor = {
                    b: min(residual_sizes[b], int(math.floor(raw_extra[b])))
                    for b in buckets
                }

                for b in buckets:
                    bucket_quotas[b] += extra_floor[b]

                assigned = sum(bucket_quotas.values())
                remainder = total_samples - assigned

                fractional_order = sorted(
                    buckets.keys(),
                    key=lambda b: (raw_extra[b] - math.floor(raw_extra[b]), -bucket_sizes[b]),
                    reverse=True,
                )

                i = 0
                while remainder > 0:
                    b = fractional_order[i % len(fractional_order)]
                    if bucket_quotas[b] < bucket_sizes[b]:
                        bucket_quotas[b] += 1
                        remainder -= 1
                    i += 1

        if sum(bucket_quotas.values()) != total_samples:
            raise RuntimeError(
                f"Internal error: quotas sum to {sum(bucket_quotas.values())}, "
                f"expected {total_samples}"
            )

        if self.enable_bucket_logging:
            self.logger.info(
                "Starting stratified sampling: split=%s, stratify_by=%s, total_samples=%d, num_buckets=%d",
                split,
                stratify_by,
                total_samples,
                len(buckets),
            )
            self.logger.info("Bucket sizes: %s", bucket_sizes)
            self.logger.info("Bucket quotas: %s", bucket_quotas)

        # Strategy:
        #   - if quota >= bucket size: take all
        #   - else cluster into k=quota and take 1 per cluster
        #   - if clustering underfills, refill from remaining unsampled items in bucket
        all_selected_global_indices: List[int] = []
        selected_global_set: set[int] = set()

        for bucket_value in sorted(buckets.keys()):
            global_indices = buckets[bucket_value]
            quota = bucket_quotas[bucket_value]
            bucket_size = len(global_indices)

            if self.enable_bucket_logging:
                self.logger.info(
                    "Processing bucket=%r size=%d quota=%d",
                    bucket_value,
                    bucket_size,
                    quota,
                )

            if quota <= 0:
                if self.enable_bucket_logging:
                    self.logger.info("Skipping bucket=%r because quota <= 0", bucket_value)
                continue

            if quota > bucket_size:
                raise RuntimeError(
                    f"Quota overflow for bucket={bucket_value!r}: quota={quota}, size={bucket_size}"
                )

            bucket_samples = [samples[i] for i in global_indices]

            # Case A: quota covers the whole bucket
            if quota == bucket_size:
                chosen_local = list(range(bucket_size))
                if self.enable_bucket_logging:
                    self.logger.info(
                        "Bucket=%r selected all %d samples because quota == bucket size",
                        bucket_value,
                        bucket_size,
                    )

            # Case B: one sample needed, no need to cluster
            elif quota == 1:
                chosen_local = [rng.randrange(bucket_size)]
                if self.enable_bucket_logging:
                    self.logger.info(
                        "Bucket=%r selected 1 random sample without clustering",
                        bucket_value,
                    )

            # Case C: cluster-based selection with refill on underflow
            else:
                labels, _ = self.cluster_samples(bucket_samples, k=quota)

                _, chosen_local = self.sample_from_clusters(
                    bucket_samples,
                    labels,
                    samples_per_cluster=1,
                    seed=rng.randint(0, 10 ** 9),
                    attach_cluster_id=False,
                )

                unique_clusters = len(set(int(x) for x in labels.tolist()))
                initial_selected = len(chosen_local)

                if self.enable_bucket_logging:
                    self.logger.info(
                        "Bucket=%r clustered with requested_k=%d realized_clusters=%d initial_selected=%d",
                        bucket_value,
                        quota,
                        unique_clusters,
                        initial_selected,
                    )

                # Refill within the same bucket if KMeans yielded fewer effective clusters
                if len(chosen_local) < quota:
                    chosen_local_set = set(chosen_local)
                    remaining_local = [
                        i for i in range(bucket_size)
                        if i not in chosen_local_set
                    ]
                    refill_needed = quota - len(chosen_local)

                    if refill_needed > len(remaining_local):
                        raise RuntimeError(
                            f"Cannot refill bucket={bucket_value!r}: need {refill_needed}, "
                            f"have only {len(remaining_local)} remaining candidates."
                        )

                    refill_local = rng.sample(remaining_local, k=refill_needed)
                    chosen_local.extend(refill_local)

                    if self.enable_bucket_logging:
                        self.logger.warning(
                            "Bucket=%r underfilled by clustering; refilled %d extra samples from remaining bucket items",
                            bucket_value,
                            refill_needed,
                        )

            if len(chosen_local) != quota:
                raise RuntimeError(
                    f"Bucket={bucket_value!r} selected {len(chosen_local)} samples, expected quota={quota}"
                )

            selected_global_for_bucket = [global_indices[i] for i in chosen_local]

            # Deduplicate defensively
            deduped_bucket_selection = []
            for idx in selected_global_for_bucket:
                if idx not in selected_global_set:
                    deduped_bucket_selection.append(idx)
                    selected_global_set.add(idx)

            if len(deduped_bucket_selection) != quota:
                raise RuntimeError(
                    f"Bucket={bucket_value!r} produced duplicate selections: "
                    f"got {len(deduped_bucket_selection)} unique samples, expected {quota}"
                )

            all_selected_global_indices.extend(deduped_bucket_selection)

            if self.enable_bucket_logging:
                self.logger.info(
                    "Bucket=%r finalized with %d selected samples",
                    bucket_value,
                    len(deduped_bucket_selection),
                )

        # Final global refill, prioritized by least represented buckets first
        current_total = len(all_selected_global_indices)
        deficit = total_samples - current_total

        if deficit > 0:
            if self.enable_bucket_logging:
                self.logger.warning(
                    "Global underfill detected after per-bucket sampling: current_total=%d target_total=%d deficit=%d",
                    current_total,
                    total_samples,
                    deficit,
                )

            refill_candidates: List[int] = []

            # smallest buckets first
            bucket_order = sorted(
                buckets.keys(),
                key=lambda b: (bucket_sizes[b], b),
            )

            for b in bucket_order:
                for idx in buckets[b]:
                    if idx not in selected_global_set:
                        refill_candidates.append(idx)

            if deficit > len(refill_candidates):
                raise RuntimeError(
                    f"Unable to refill deficit={deficit}; only {len(refill_candidates)} unused samples remain."
                )

            refill_global = rng.sample(refill_candidates, k=deficit)
            all_selected_global_indices.extend(refill_global)
            selected_global_set.update(refill_global)

            if self.enable_bucket_logging:
                self.logger.warning(
                    "Performed global refill of %d samples, prioritizing least represented buckets",
                    deficit,
                )

        elif deficit < 0:
            raise RuntimeError(
                f"Internal error: oversampled {len(all_selected_global_indices)} items "
                f"for target_total={total_samples}"
            )

        if len(all_selected_global_indices) != total_samples:
            raise RuntimeError(
                f"Final selection size mismatch: got {len(all_selected_global_indices)}, "
                f"expected {total_samples}"
            )

        rng.shuffle(all_selected_global_indices)

        if self.enable_bucket_logging:
            realized_bucket_counts: Dict[Any, int] = {}
            for idx in all_selected_global_indices:
                bucket_value = samples[idx][stratify_by]
                realized_bucket_counts[bucket_value] = realized_bucket_counts.get(bucket_value, 0) + 1

            self.logger.info(
                "Completed stratified sampling: selected=%d samples",
                len(all_selected_global_indices),
            )
            self.logger.info("Realized per-bucket counts: %s", realized_bucket_counts)

        subset = []
        for idx in all_selected_global_indices:
            item = dict(samples[idx])
            item["source_idx"] = int(idx)
            item["stratify_bucket"] = samples[idx][stratify_by]
            subset.append(item)

        saved_path = None
        if save_indices:
            if subset_name is None:
                subset_name = (
                    f"{split}_stratified_{stratify_by}_n{total_samples}_"
                    f"seed{self.random_state if seed is None else seed}"
                )

            realized_bucket_counts: Dict[Any, int] = {}
            for idx in all_selected_global_indices:
                bucket_value = samples[idx][stratify_by]
                realized_bucket_counts[bucket_value] = realized_bucket_counts.get(bucket_value, 0) + 1

            saved_path = self.save_subset_indices(
                all_selected_global_indices,
                split=split,
                subset_name=subset_name,
                seed=seed,
                sampling_mode=f"stratified_kmeans_by_{stratify_by}_mandatory_coverage_with_refill",
                total_samples=total_samples,
                extra_metadata={
                    "stratify_by": stratify_by,
                    "bucket_sizes": {str(k): int(v) for k, v in bucket_sizes.items()},
                    "bucket_quotas": {str(k): int(v) for k, v in bucket_quotas.items()},
                    "realized_bucket_counts": {
                        str(k): int(v) for k, v in realized_bucket_counts.items()
                    },
                    "mandatory_min_per_nonempty_bucket": 1,
                    "exact_total_enforced": True,
                    **(extra_metadata or {}),
                },
            )

            if self.enable_bucket_logging:
                self.logger.info("Saved subset indices to %s", saved_path)

        return subset, all_selected_global_indices, saved_path

    def load_subset_indices(
            self,
            subset_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
        path = self.indices_dir / f"{subset_name}.json"

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        split = payload["split"]
        indices = [int(i) for i in payload["indices"]]
        samples = self.get_split(split)

        subset = []
        for idx in indices:
            item = dict(samples[idx])
            item["source_idx"] = idx
            subset.append(item)

        return subset, indices, payload
