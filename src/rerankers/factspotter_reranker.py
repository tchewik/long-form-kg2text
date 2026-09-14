from typing import Any, List, Dict, Tuple

import torch


class FactSpotterReranker:
    """
    Scores candidate texts against a list of KG triples using FactSpotter.
    Returns recall = (#entailed triples) / (#triples).
    Tie-break helper: mean entailment probability across triples.
    """

    def __init__(
            self,
            model_name: str,
            device: str = "cuda",
            batch_size: int = 32,
            entailment_threshold: float = 0.5,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.threshold = float(entailment_threshold)

        use_cuda = torch.cuda.is_available() and device.startswith("cuda")
        self.device = torch.device(device if use_cuda else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _batched(self, xs, bs: int):
        batch = []
        for x in xs:
            batch.append(x)
            if len(batch) >= bs:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _triple_to_str(t: Dict[str, str]) -> str:
        return f"{t.get('subject', '')} | {t.get('predicate', '')} | {t.get('object', '')}"

    def _entail_probs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Returns entailment probabilities for each (text, triple_str) pair.
        FactSpotter HF heads use class 0 = entailment.
        """
        entail_probs: List[float] = []
        with torch.no_grad():
            for batch_pairs in self._batched(pairs, self.batch_size):
                enc = self.tokenizer(
                    batch_pairs,
                    truncation=True,
                    padding=True,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                entail_probs.extend(probs[:, 0].detach().cpu().tolist())
        return entail_probs

    def score_text(self, triples: List[Dict[str, str]], text: str) -> Tuple[float, float]:
        """
        Returns: (recall, mean_entail_prob)
          recall = fraction of triples with P(entailment) >= threshold
        """
        result = self.score_text_detailed(triples, text)
        return result["recall"], result["mean_entail_prob"]

    def score_text_detailed(self, triples: List[Dict[str, str]], text: str) -> Dict[str, Any]:
        """
        Returns a detailed FactSpotter result.

        The important field for filtering is `entailed_mask`.
        We compare that mask before/after sentence deletion.

        Returns:
            {
                "recall": float,
                "mean_entail_prob": float,
                "entailed_mask": Tuple[bool, ...],
                "probs": Tuple[float, ...],
            }
        """
        if not triples:
            return {
                "recall": 0.0,
                "mean_entail_prob": 0.0,
                "entailed_mask": tuple(),
                "probs": tuple(),
            }

        triple_strs = [self._triple_to_str(t) for t in triples]
        pairs = [(text, ts) for ts in triple_strs]
        probs = self._entail_probs(pairs)

        entailed_mask = tuple(p >= self.threshold for p in probs)
        n = len(probs)
        n_entailed = sum(entailed_mask)

        return {
            "recall": float(n_entailed / n) if n else 0.0,
            "mean_entail_prob": float(sum(probs) / n) if n else 0.0,
            "entailed_mask": entailed_mask,
            "probs": tuple(float(p) for p in probs),
        }
