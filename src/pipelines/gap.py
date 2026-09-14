import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration, PreTrainedTokenizerBase
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.data.kgdataset import KGDataset
from src.pipelines.base_pipeline import graph_key

logger = logging.getLogger(__name__)


GAP_SPECIAL_TOKENS = [
    "[graph]",
    "[text]",
    "[head]",
    "[relation]",
    "[tail]",
]


def _canon_triple(t: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        (t.get("subject") or "").strip(),
        (t.get("predicate") or "").strip(),
        (t.get("object") or "").strip(),
    )


@dataclass
class GAPConfig:

    model_name_or_path: str = "facebook/bart-base"

    # Input/output lengths
    max_source_length: int = 256
    max_new_tokens: int = 128

    @property
    def max_target_length(self) -> int:
        return min(int(self.max_source_length) + int(self.max_new_tokens), 1022)

    # Graph construction
    max_nodes: int = 50
    relations_as_nodes: bool = True
    entity_entity: bool = True
    entity_relation: bool = True
    relation_entity: bool = False
    relation_relation: bool = False
    type_encoding: bool = True

    # Graph fusion module
    graph_layers: int = 6
    graph_dropout: float = 0.1
    graph_residual_scale: float = 1.0

    # Training
    batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: Optional[int] = 1600
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every: int = 50

    # Validation / checkpoint selection
    select_best_by_bleu: bool = True
    eval_every_epoch: bool = True
    early_stopping_patience: Optional[int] = 3
    dev_eval_limit: int = 200

    # Precision/device
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False

    # Decoding
    num_beams: int = 5
    length_penalty: float = 1.0
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # Runtime
    num_workers: int = 0
    seed: int = 42
    save_dir: Optional[str] = None
    load_if_exists: bool = False
    generation_batch_size: int = 8

    # Optional
    freeze_base_model: bool = False


class GAPGraphAttentionLayer(nn.Module):
    """
    adj_matrix:
      0 = no edge / masked
      1 = entity-entity
      2 = entity-relation
      3 = relation-entity
      4 = relation-relation
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        type_encoding: bool = True,
        num_edge_types: int = 4,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.type_encoding = bool(type_encoding)
        self.num_edge_types = int(num_edge_types)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        if self.type_encoding:
            # index 0 = no-edge; usually masked, but kept for stability
            self.edge_type_bias = nn.Embedding(self.num_edge_types + 1, num_heads)
        else:
            self.edge_type_bias = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, H, N, Dh]
        bsz, n_nodes, _ = x.shape
        return x.view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        node_states: torch.Tensor,
        node_mask: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_states: [B, N, D]
        node_mask:   [B, N], True for real graph nodes
        adj_matrix:  [B, N, N], edge type IDs
        """

        bsz, n_nodes, _ = node_states.shape
        device = node_states.device

        q = self._shape(self.q_proj(node_states))
        k = self._shape(self.k_proj(node_states))
        v = self._shape(self.v_proj(node_states))

        # [B, H, N, N]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        adj = adj_matrix.clamp(min=0, max=self.num_edge_types)

        if self.edge_type_bias is not None:
            # [B, N, N, H] -> [B, H, N, N]
            type_bias = self.edge_type_bias(adj).permute(0, 3, 1, 2)
            scores = scores + type_bias

        edge_allowed = adj > 0

        # Always allow valid nodes to attend to themselves.
        eye = torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        edge_allowed = edge_allowed | eye

        query_valid = node_mask[:, None, :, None]
        key_valid = node_mask[:, None, None, :]
        allowed = edge_allowed[:, None, :, :] & query_valid & key_valid

        # Prevent NaNs for rows with no valid neighbors.
        row_has_any = allowed.any(dim=-1, keepdim=True)

        scores = scores.masked_fill(~allowed, -1e4)
        scores = torch.where(row_has_any, scores, torch.zeros_like(scores))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, self.hidden_size)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        return out


class GAPBartForConditionalGeneration(BartForConditionalGeneration):
    """
    BART-only faithful GAP port.

    Use exactly like BartForConditionalGeneration, but pass:
      token_node_ids: [B, L]
      adj_matrix:     [B, N, N]

    During generation, this class precomputes graph-aware encoder outputs and
    then delegates decoding to Hugging Face's normal generate().
    """

    def __init__(self, config, gap_config=None):
        super().__init__(config)

        self.gap_config = gap_config

        if gap_config is not None:
            self._replace_encoder_layers_with_gap(gap_config)

    def _replace_encoder_layers_with_gap(self, gap_config) -> None:
        hidden_size = int(getattr(self.config, "d_model"))
        num_heads = int(getattr(self.config, "encoder_attention_heads"))

        new_layers = []
        for layer in self.model.encoder.layers:
            new_layers.append(
                GAPBartEncoderLayer(
                    bart_layer=layer,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    graph_dropout=getattr(gap_config, "graph_dropout", 0.1),
                    type_encoding=getattr(gap_config, "type_encoding", True),
                    graph_residual_scale=getattr(gap_config, "graph_residual_scale", 1.0),
                )
            )

        self.model.encoder.layers = nn.ModuleList(new_layers)

        logger.info(
            "Replaced %d BART encoder layers with GAP encoder layers.",
            len(new_layers),
        )

    def _set_graph_context(
        self,
        token_node_ids: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        for layer in self.model.encoder.layers:
            if hasattr(layer, "set_graph_context"):
                layer.set_graph_context(
                    token_node_ids=token_node_ids,
                    adj_matrix=adj_matrix,
                    attention_mask=attention_mask,
                )

    def _clear_graph_context(self) -> None:
        for layer in self.model.encoder.layers:
            if hasattr(layer, "clear_graph_context"):
                layer.clear_graph_context()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_node_ids: Optional[torch.Tensor] = None,
            adj_matrix: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Any] = None,
            **kwargs,
    ) -> Seq2SeqLMOutput:
        """
        Training/eval forward.

        During normal training, token_node_ids and adj_matrix are used to activate
        GAP graph context inside every BART encoder layer.

        During generation, encoder_outputs may already be precomputed by
        GAPBartForConditionalGeneration.generate().
        """

        if encoder_outputs is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **kwargs,
            )

        if token_node_ids is not None and adj_matrix is not None:
            self._set_graph_context(
                token_node_ids=token_node_ids,
                adj_matrix=adj_matrix,
                attention_mask=attention_mask,
            )
            try:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            finally:
                self._clear_graph_context()

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_node_ids: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Graph-aware generation.

        Hugging Face generate() normally calls the encoder internally. We need
        to precompute encoder_outputs while graph context is active, then pass
        those encoder outputs into normal BART generation.
        """

        if token_node_ids is not None and adj_matrix is not None:
            self._set_graph_context(
                token_node_ids=token_node_ids,
                adj_matrix=adj_matrix,
                attention_mask=attention_mask,
            )
            try:
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            finally:
                self._clear_graph_context()

            return super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **generation_kwargs,
            )

        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def save_gap_pretrained(self, save_dir: Path, tokenizer) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        if self.gap_config is not None:
            with (save_dir / "gap_config.json").open("w", encoding="utf-8") as f:
                json.dump(asdict(self.gap_config), f, indent=2)

        logger.info("Saved GAP-BART checkpoint to %s", save_dir)


class GAPBartEncoderLayer(nn.Module):
    """
    Faithful layer-level GAP port for modern Hugging Face BART.

    This wraps a normal BartEncoderLayer and inserts the GAP graph aggregation
    immediately after token self-attention and before the FFN, matching the
    official GAP code structure.
    """

    def __init__(
        self,
        bart_layer: nn.Module,
        hidden_size: int,
        num_heads: int,
        graph_dropout: float,
        type_encoding: bool,
        graph_residual_scale: float = 1.0,
    ):
        super().__init__()

        # Reuse/copy official BART layer modules so pretrained weights load
        # into the same names: self_attn, self_attn_layer_norm, fc1, fc2, ...
        self.self_attn = bart_layer.self_attn
        self.self_attn_layer_norm = bart_layer.self_attn_layer_norm
        self.fc1 = bart_layer.fc1
        self.fc2 = bart_layer.fc2
        self.final_layer_norm = bart_layer.final_layer_norm

        self.dropout = bart_layer.dropout
        self.activation_fn = bart_layer.activation_fn
        self.activation_dropout = bart_layer.activation_dropout

        self.graph_attn = GAPGraphAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=graph_dropout,
            type_encoding=type_encoding,
        )

        self.graph_residual_scale = float(graph_residual_scale)
        self.graph_gate = nn.Parameter(torch.tensor(1.0))

        # These are set per forward pass by GAPBartForConditionalGeneration.
        self._gap_token_node_ids: Optional[torch.Tensor] = None
        self._gap_adj_matrix: Optional[torch.Tensor] = None
        self._gap_attention_mask: Optional[torch.Tensor] = None

    def set_graph_context(
        self,
        token_node_ids: Optional[torch.Tensor],
        adj_matrix: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        self._gap_token_node_ids = token_node_ids
        self._gap_adj_matrix = adj_matrix
        self._gap_attention_mask = attention_mask

    def clear_graph_context(self) -> None:
        self._gap_token_node_ids = None
        self._gap_adj_matrix = None
        self._gap_attention_mask = None

    @staticmethod
    def _pool_tokens_to_nodes(
        token_states: torch.Tensor,
        token_node_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_states:   [B, L, D]
        token_node_ids: [B, L], -1 for non-node tokens
        attention_mask: [B, L]
        """

        bsz, seq_len, hidden = token_states.shape
        device = token_states.device
        dtype = token_states.dtype

        valid = (
            (token_node_ids >= 0)
            & (token_node_ids < max_nodes)
            & attention_mask.bool()
        )

        safe_ids = token_node_ids.clamp(min=0, max=max_nodes - 1)

        sums = torch.zeros(bsz, max_nodes, hidden, device=device, dtype=dtype)
        counts = torch.zeros(bsz, max_nodes, 1, device=device, dtype=dtype)

        expanded_ids = safe_ids.unsqueeze(-1).expand(-1, -1, hidden)
        src = token_states * valid.unsqueeze(-1).to(dtype)

        sums.scatter_add_(dim=1, index=expanded_ids, src=src)
        counts.scatter_add_(
            dim=1,
            index=safe_ids.unsqueeze(-1),
            src=valid.unsqueeze(-1).to(dtype),
        )

        node_states = sums / counts.clamp_min(1.0)
        node_mask = counts.squeeze(-1) > 0

        return node_states, node_mask

    @staticmethod
    def _gather_nodes_to_tokens(
        node_states: torch.Tensor,
        token_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_states:    [B, N, D]
        token_node_ids: [B, L]
        """

        bsz, n_nodes, hidden = node_states.shape
        safe_ids = token_node_ids.clamp(min=0, max=n_nodes - 1)

        gathered = torch.gather(
            node_states,
            dim=1,
            index=safe_ids.unsqueeze(-1).expand(-1, -1, hidden),
        )

        gathered = gathered.masked_fill((token_node_ids < 0).unsqueeze(-1), 0.0)
        return gathered

    def _apply_gap(self, hidden_states: torch.Tensor) -> torch.Tensor:
        token_node_ids = self._gap_token_node_ids
        adj_matrix = self._gap_adj_matrix
        attention_mask = self._gap_attention_mask

        if token_node_ids is None or adj_matrix is None or attention_mask is None:
            return hidden_states

        max_nodes = adj_matrix.shape[1]

        node_states, node_mask = self._pool_tokens_to_nodes(
            token_states=hidden_states,
            token_node_ids=token_node_ids,
            attention_mask=attention_mask,
            max_nodes=max_nodes,
        )

        graph_node_states = self.graph_attn(
            node_states=node_states,
            node_mask=node_mask,
            adj_matrix=adj_matrix,
        )

        token_graph_states = self._gather_nodes_to_tokens(
            node_states=graph_node_states,
            token_node_ids=token_node_ids,
        )

        scale = self.graph_residual_scale * torch.sigmoid(self.graph_gate)
        return hidden_states + scale * token_graph_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            **kwargs,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
            attn_weights = attn_out[1] if len(attn_out) > 1 else None
        else:
            hidden_states = attn_out
            attn_weights = None

        # GAP insertion point: after token self-attention, before dropout/residual/norm.
        hidden_states = self._apply_gap(hidden_states)

        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states,
            p=self.activation_dropout,
            training=self.training,
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # This transformers version expects BartEncoderLayer.forward() to return
        # hidden_states directly, not a tuple.
        return hidden_states


class GAPExampleEncoder:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: GAPConfig):
        self.tokenizer = tokenizer
        self.config = config

        self.graph_ids = self._encode_piece("[graph]", node_id=-1)
        self.text_ids = self._encode_piece("[text]", node_id=-1)
        self.head_ids = self._encode_piece("[head]", node_id=-1)
        self.rel_ids = self._encode_piece("[relation]", node_id=-1)
        self.tail_ids = self._encode_piece("[tail]", node_id=-1)

    def _encode_text_only(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def _encode_piece(self, text: str, node_id: int) -> Tuple[List[int], List[int]]:
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return ids, [node_id] * len(ids)

    def _add_piece(
        self,
        ids_out: List[int],
        node_out: List[int],
        text: str,
        node_id: int,
    ) -> None:
        ids = self._encode_text_only(text)
        ids_out.extend(ids)
        node_out.extend([node_id] * len(ids))

    def _make_node_maps(
        self,
        triples: List[Dict[str, str]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        entity_to_id: Dict[str, int] = {}
        relation_to_id: Dict[str, int] = {}

        def add_entity(x: str) -> None:
            if not x:
                return
            if x not in entity_to_id and len(entity_to_id) + len(relation_to_id) < self.config.max_nodes:
                entity_to_id[x] = len(entity_to_id) + len(relation_to_id)

        def add_relation(x: str) -> None:
            if not x or not self.config.relations_as_nodes:
                return
            if x not in relation_to_id and len(entity_to_id) + len(relation_to_id) < self.config.max_nodes:
                relation_to_id[x] = len(entity_to_id) + len(relation_to_id)

        for t in triples:
            s, p, o = _canon_triple(t)
            add_entity(s)
            add_entity(o)
            add_relation(p)

        return entity_to_id, relation_to_id

    def _build_adj(
        self,
        triples: List[Dict[str, str]],
        entity_to_id: Dict[str, int],
        relation_to_id: Dict[str, int],
    ) -> torch.Tensor:
        n = self.config.max_nodes
        adj = torch.zeros(n, n, dtype=torch.long)

        relation_nodes_in_triples: List[Tuple[int, str, str, str]] = []

        for t in triples:
            s, p, o = _canon_triple(t)
            h = entity_to_id.get(s)
            r = relation_to_id.get(p)
            tail = entity_to_id.get(o)

            if h is None or tail is None:
                continue

            if self.config.entity_entity:
                adj[h, tail] = 1
                adj[tail, h] = 1

            if r is not None:
                relation_nodes_in_triples.append((r, s, p, o))

                if self.config.entity_relation:
                    adj[h, r] = 2
                    adj[tail, r] = 2

                if self.config.relation_entity:
                    adj[r, h] = 3
                    adj[r, tail] = 3

        if self.config.relation_relation and relation_nodes_in_triples:
            # Connect relation nodes that share a subject or object entity.
            for i, (r1, s1, _p1, o1) in enumerate(relation_nodes_in_triples):
                ents1 = {s1, o1}
                for r2, s2, _p2, o2 in relation_nodes_in_triples[i + 1:]:
                    ents2 = {s2, o2}
                    if ents1 & ents2:
                        adj[r1, r2] = 4
                        adj[r2, r1] = 4

        return adj

    def encode_source(
        self,
        triples: List[Dict[str, str]],
    ) -> Dict[str, torch.Tensor]:
        triples = triples or []
        entity_to_id, relation_to_id = self._make_node_maps(triples)

        input_ids: List[int] = []
        token_node_ids: List[int] = []

        def append_ids(ids: List[int], node_ids: List[int]) -> None:
            input_ids.extend(ids)
            token_node_ids.extend(node_ids)

        append_ids(*self.graph_ids)

        for t in triples:
            s, p, o = _canon_triple(t)

            h_id = entity_to_id.get(s, -1)
            r_id = relation_to_id.get(p, -1)
            o_id = entity_to_id.get(o, -1)

            append_ids(*self.head_ids)
            self._add_piece(input_ids, token_node_ids, " " + s, h_id)

            append_ids(*self.rel_ids)
            self._add_piece(input_ids, token_node_ids, " " + p, r_id)

            append_ids(*self.tail_ids)
            self._add_piece(input_ids, token_node_ids, " " + o, o_id)

        append_ids(*self.text_ids)

        input_ids, token_node_ids = self._add_model_specials(input_ids, token_node_ids)
        input_ids, token_node_ids = self._truncate(input_ids, token_node_ids)

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_node_ids": torch.tensor(token_node_ids, dtype=torch.long),
            "adj_matrix": self._build_adj(triples, entity_to_id, relation_to_id),
        }

    def encode_target(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(
            text or "",
            add_special_tokens=True,
            max_length=self.config.max_target_length,
            truncation=True,
        )
        return torch.tensor(ids, dtype=torch.long)

    def _add_model_specials(
        self,
        input_ids: List[int],
        token_node_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id

        if bos is not None:
            input_ids = [bos] + input_ids
            token_node_ids = [-1] + token_node_ids

        if eos is not None:
            input_ids = input_ids + [eos]
            token_node_ids = token_node_ids + [-1]

        return input_ids, token_node_ids

    def _truncate(
        self,
        input_ids: List[int],
        token_node_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        max_len = self.config.max_source_length

        if len(input_ids) <= max_len:
            return input_ids, token_node_ids

        eos = self.tokenizer.eos_token_id
        input_ids = input_ids[:max_len]
        token_node_ids = token_node_ids[:max_len]

        if eos is not None:
            input_ids[-1] = eos
            token_node_ids[-1] = -1

        return input_ids, token_node_ids


class GAPDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        config: GAPConfig,
        include_labels: bool,
    ):
        self.samples = samples
        self.encoder = GAPExampleEncoder(tokenizer, config)
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        triples = sample.get("triples_parsed") or []
        text = sample.get("text") or ""

        rec = self.encoder.encode_source(triples)
        rec["idx"] = idx
        rec["triples"] = triples
        rec["reference"] = text

        if self.include_labels:
            rec["labels"] = self.encoder.encode_target(text)

        return rec


class GAPCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: GAPConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have either pad_token_id or eos_token_id.")

    def _pad_1d(
        self,
        tensors: List[torch.Tensor],
        pad_value: int,
    ) -> torch.Tensor:
        max_len = max(int(t.numel()) for t in tensors)
        out = torch.full(
            (len(tensors), max_len),
            fill_value=pad_value,
            dtype=torch.long,
        )
        for i, t in enumerate(tensors):
            out[i, : t.numel()] = t
        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = self._pad_1d(
            [x["input_ids"] for x in batch],
            pad_value=self.pad_token_id,
        )
        attention_mask = self._pad_1d(
            [x["attention_mask"] for x in batch],
            pad_value=0,
        )
        token_node_ids = self._pad_1d(
            [x["token_node_ids"] for x in batch],
            pad_value=-1,
        )

        adj_matrix = torch.stack([x["adj_matrix"] for x in batch], dim=0)

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_node_ids": token_node_ids,
            "adj_matrix": adj_matrix,
            "idx": [x["idx"] for x in batch],
            "triples": [x["triples"] for x in batch],
            "reference": [x["reference"] for x in batch],
        }

        if "labels" in batch[0]:
            labels = self._pad_1d(
                [x["labels"] for x in batch],
                pad_value=self.pad_token_id,
            )
            labels = labels.masked_fill(labels == self.pad_token_id, -100)
            out["labels"] = labels

        return out


def _simple_tokenize_for_bleu(text: str) -> List[str]:
    return (text or "").strip().lower().split()


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu_4(
    predictions: List[str],
    references: List[str],
    smooth: float = 1.0,
) -> float:
    """
    Lightweight corpus BLEU-4.

    Returns BLEU in [0, 100].

    This is not the exact COCO/WebNLG evaluation script, but is sufficient for
    dev checkpoint selection. Final reporting should still use your official
    evaluator.
    """

    if not predictions or not references:
        return 0.0

    assert len(predictions) == len(references)

    clipped = [0.0, 0.0, 0.0, 0.0]
    total = [0.0, 0.0, 0.0, 0.0]

    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        p_toks = _simple_tokenize_for_bleu(pred)
        r_toks = _simple_tokenize_for_bleu(ref)

        pred_len += len(p_toks)
        ref_len += len(r_toks)

        for n in range(1, 5):
            p_counts = _ngram_counts(p_toks, n)
            r_counts = _ngram_counts(r_toks, n)

            total[n - 1] += max(0, len(p_toks) - n + 1)

            for ng, count in p_counts.items():
                clipped[n - 1] += min(count, r_counts.get(ng, 0))

    if pred_len == 0:
        return 0.0

    precisions = []
    for c, t in zip(clipped, total):
        # Additive smoothing for stable early-epoch comparison.
        precisions.append((c + smooth) / (t + smooth))

    log_precision = sum(math.log(p) for p in precisions) / 4.0

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_len) / max(1.0, float(pred_len)))

    bleu = bp * math.exp(log_precision)
    return float(bleu * 100.0)


class GAPPipeline:
    """
    Pipeline object for supervised GAP fine-tuning and inference.

    Typical use:

        cfg = GAPConfig(
            model_name_or_path="facebook/bart-base",
            save_dir="outputs/gap/save",
            entity_entity=True,
            entity_relation=True,
            type_encoding=True,
        )

        pipe = GAPPipeline(dataset, cfg)
        pipe.train()
        results = pipe.generate_for_split("test")
    """

    def __init__(self, dataset: KGDataset, config: GAPConfig):
        self.dataset = dataset
        self.config = config

        try:
            logger.info(
                f"Initializing {self.__class__.__name__} with config: "
                f"{json.dumps(asdict(self.config), ensure_ascii=False, indent=2)}"
            )
        except Exception:
            logger.info(f"Initializing {self.__class__.__name__} with config (non-serializable)")

        torch.manual_seed(config.seed)

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        model_path = config.model_name_or_path
        save_dir = Path(config.save_dir) if config.save_dir else None

        if config.load_if_exists and save_dir is not None and (save_dir / "config.json").exists():
            model_path = str(save_dir)
            logger.info("Loading existing GAP-BART checkpoint from %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        special_map = getattr(self.tokenizer, "special_tokens_map_extended", {}) or {}
        existing_specials = list(special_map.get("additional_special_tokens", []) or [])

        added_vocab = set(getattr(self.tokenizer, "get_added_vocab", lambda: {})().keys())
        existing_vocab = set(self.tokenizer.get_vocab().keys())

        to_add = [
            tok for tok in GAP_SPECIAL_TOKENS
            if tok not in existing_specials
               and tok not in added_vocab
               and tok not in existing_vocab
        ]

        if to_add:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": existing_specials + to_add}
            )

        hf_cfg = AutoConfig.from_pretrained(model_path)
        if getattr(hf_cfg, "model_type", None) != "bart":
            raise ValueError(
                "Faithful GAP layer-level port currently supports only BART models. "
                f"Got model_type={getattr(hf_cfg, 'model_type', None)} from {model_path}."
            )

        self.model = GAPBartForConditionalGeneration.from_pretrained(
            model_path,
            gap_config=config,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def _autocast_context(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)

        if self.config.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)

        if self.config.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        return torch.autocast(device_type="cuda", enabled=False)

    def _get_split_samples(self, split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        split = split.lower()

        if split == "train":
            samples = self.dataset.load_train()
        elif split == "test":
            if hasattr(self.dataset, "load_test_500"):
                samples = self.dataset.load_test_500()
            else:
                samples = self.dataset.data["test"]
        elif split == "dev":
            if "dev" in self.dataset.data:
                samples = self.dataset.data["dev"]
            elif "val" in self.dataset.data:
                samples = self.dataset.data["val"]
            else:
                raise ValueError("Dataset has neither dev nor val split.")
        elif split == "val":
            if "val" in self.dataset.data:
                samples = self.dataset.data["val"]
            elif "dev" in self.dataset.data:
                samples = self.dataset.data["dev"]
            else:
                raise ValueError("Dataset has neither val nor dev split.")
        else:
            if split not in self.dataset.data:
                raise ValueError(f"Unknown split: {split}")
            samples = self.dataset.data[split]

        if limit is not None and limit > 0:
            samples = samples[:limit]

        return list(samples)

    def _make_loader(
        self,
        samples: List[Dict[str, Any]],
        include_labels: bool,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        ds = GAPDataset(
            samples=samples,
            tokenizer=self.tokenizer,
            config=self.config,
            include_labels=include_labels,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=GAPCollator(self.tokenizer, self.config),
        )

    @torch.no_grad()
    def evaluate_bleu_on_split(
            self,
            split: str = "dev",
            limit: Optional[int] = None,
    ) -> float:
        """
        Generate on dev/val and compute lightweight corpus BLEU-4.

        Used only for checkpoint selection.
        """

        self.model.eval()

        if limit is None:
            limit = self.config.dev_eval_limit

        samples = self._get_split_samples(split, limit=limit)

        if not samples:
            logger.warning("No samples found for GAP BLEU evaluation on split=%s", split)
            return 0.0

        loader = self._make_loader(
            samples=samples,
            include_labels=False,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )

        predictions: List[str] = []
        references: List[str] = []

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_beams": self.config.num_beams,
            "length_penalty": self.config.length_penalty,
            "do_sample": False,
        }

        for batch in tqdm(loader, desc=f"GAP dev BLEU ({split})", leave=False):
            batch = self._move_batch(batch)

            output_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_node_ids=batch["token_node_ids"],
                adj_matrix=batch["adj_matrix"],
                **gen_kwargs,
            )

            decoded = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            predictions.extend([(x or "").strip() for x in decoded])
            references.extend([(x or "").strip() for x in batch["reference"]])

        bleu = corpus_bleu_4(predictions, references)
        logger.info("GAP validation BLEU-4 on %s: %.4f", split, bleu)

        return bleu

    def train(self) -> None:
        train_samples = self._get_split_samples("train")
        if not train_samples:
            raise ValueError("No training samples found.")

        train_loader = self._make_loader(
            samples=train_samples,
            include_labels=True,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        update_steps_per_epoch = max(
            1,
            math.ceil(len(train_loader) / max(1, self.config.gradient_accumulation_steps)),
        )
        total_steps = update_steps_per_epoch * max(1, self.config.num_epochs)

        if self.config.warmup_steps is not None:
            warmup_steps = int(self.config.warmup_steps)
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            "Starting GAP training: samples=%d epochs=%d total_steps=%d warmup_steps=%d",
            len(train_samples),
            self.config.num_epochs,
            total_steps,
            warmup_steps,
        )

        best_bleu = -1.0
        best_epoch = -1
        bad_epochs = 0

        save_dir = Path(self.config.save_dir) if self.config.save_dir else None

        for epoch in range(self.config.num_epochs):
            self.model.train()
            running_loss = 0.0
            logged_steps = 0

            pbar = tqdm(
                train_loader,
                desc=f"GAP train epoch {epoch + 1}/{self.config.num_epochs}",
            )

            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(pbar, start=1):
                batch = self._move_batch(batch)

                with self._autocast_context():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_node_ids=batch["token_node_ids"],
                        adj_matrix=batch["adj_matrix"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss / max(1, self.config.gradient_accumulation_steps)

                loss.backward()

                running_loss += float(loss.detach().cpu()) * max(
                    1,
                    self.config.gradient_accumulation_steps,
                )
                logged_steps += 1

                if step % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            self.config.max_grad_norm,
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_update_step = epoch * update_steps_per_epoch + (
                            step // max(1, self.config.gradient_accumulation_steps)
                    )

                    if global_update_step % self.config.log_every == 0:
                        avg_loss = running_loss / max(1, logged_steps)
                        pbar.set_postfix(loss=f"{avg_loss:.4f}")
                        logger.info(
                            "GAP train epoch=%d update_step=%d loss=%.4f lr=%.3e",
                            epoch + 1,
                            global_update_step,
                            avg_loss,
                            scheduler.get_last_lr()[0],
                        )
                        running_loss = 0.0
                        logged_steps = 0

            if self.config.select_best_by_bleu and self.config.eval_every_epoch:
                try:
                    dev_bleu = self.evaluate_bleu_on_split("dev")
                    dev_split = "dev"
                except Exception as e:
                    logger.warning(
                        "Failed to evaluate on dev split (%s). Trying val split.",
                        e,
                    )
                    dev_bleu = self.evaluate_bleu_on_split("val")
                    dev_split = "val"

                logger.info(
                    "GAP epoch=%d validation split=%s BLEU=%.4f best_BLEU=%.4f best_epoch=%d",
                    epoch + 1,
                    dev_split,
                    dev_bleu,
                    best_bleu,
                    best_epoch,
                )

                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    best_epoch = epoch + 1
                    bad_epochs = 0

                    if save_dir is not None:
                        self.save(save_dir)

                    logger.info(
                        "New best GAP checkpoint: epoch=%d BLEU=%.4f saved_to=%s",
                        best_epoch,
                        best_bleu,
                        save_dir,
                    )
                else:
                    bad_epochs += 1
                    logger.info(
                        "GAP validation BLEU did not improve. bad_epochs=%d patience=%s",
                        bad_epochs,
                        self.config.early_stopping_patience,
                    )

                    if (
                            self.config.early_stopping_patience is not None
                            and bad_epochs >= self.config.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping GAP training at epoch=%d. Best epoch=%d, best BLEU=%.4f",
                            epoch + 1,
                            best_epoch,
                            best_bleu,
                        )
                        break

            else:
                # Fallback: save final checkpoint only.
                if save_dir is not None:
                    self.save(save_dir)

        if self.config.select_best_by_bleu:
            logger.info(
                "Finished GAP training. Best epoch=%d best validation BLEU=%.4f",
                best_epoch,
                best_bleu,
            )

            # Reload best checkpoint before test generation.
            if save_dir is not None and (save_dir / "config.json").exists():
                logger.info("Reloading best GAP checkpoint from %s", save_dir)

                self.model = GAPBartForConditionalGeneration.from_pretrained(
                    str(save_dir),
                    gap_config=self.config,
                )
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.to(self.device)

    def save(self, save_dir: Path) -> None:
        self.model.save_gap_pretrained(save_dir, self.tokenizer)

    @torch.no_grad()
    def generate_for_split(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        cache_path: Optional[str] = None,
        resume: bool = True,
    ) -> List[Dict[str, Any]]:
        samples = self._get_split_samples(split, limit=limit)

        cache_file = Path(cache_path) if cache_path is not None else None
        cache_by_idx: Dict[int, Dict[str, Any]] = {}

        if cache_file is not None and resume and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cache_by_idx[int(rec["idx"])] = rec
                    except Exception as e:
                        logger.warning("Failed to parse GAP cache line: %s", e)

            logger.info("Loaded %d cached GAP predictions from %s", len(cache_by_idx), cache_file)

        loader = self._make_loader(
            samples=samples,
            include_labels=False,
            batch_size=self.config.generation_batch_size,
            shuffle=False,
        )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "num_beams": self.config.num_beams,
            "length_penalty": self.config.length_penalty,
            "do_sample": self.config.do_sample,
        }

        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        self.model.eval()
        results: List[Dict[str, Any]] = []

        for batch in tqdm(loader, desc=f"GAP generating {split}"):
            original_indices = batch["idx"]

            # Skip fully cached batches.
            if all(i in cache_by_idx for i in original_indices):
                results.extend(cache_by_idx[i] for i in original_indices)
                continue

            batch = self._move_batch(batch)

            output_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_node_ids=batch["token_node_ids"],
                adj_matrix=batch["adj_matrix"],
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                do_sample=self.config.do_sample,
            )

            decoded = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            for local_i, pred in enumerate(decoded):
                idx = original_indices[local_i]

                if idx in cache_by_idx:
                    results.append(cache_by_idx[idx])
                    continue

                triples = batch["triples"][local_i]
                reference = batch["reference"][local_i]
                pred = (pred or "").strip()

                rec = {
                    "idx": idx,
                    "graph_key": graph_key(triples),
                    "prompt_key": None,
                    "triples": triples,
                    "reference": reference,
                    "prediction": pred,
                    "all_prediction": pred,
                    "raw_samples": [pred],
                    "usage": {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    "all_usage": {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    "raw_usage": [],
                    "method": "gap",
                }

                results.append(rec)

                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with cache_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        results.sort(key=lambda r: r["idx"])
        return results


def run_gap_baseline(
    dataset: KGDataset,
    model_name: str,
    output_dir: Path,
    num_epochs: Optional[int] = None,
    num_examples: int = -1,
    load_if_exists: bool = False,
    batch_size: Optional[int] = None,
    generation_batch_size: Optional[int] = None,
    max_source_length: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    learning_rate: Optional[float] = None,
    entity_entity: bool = True,
    entity_relation: bool = True,
    relation_entity: bool = False,
    relation_relation: bool = False,
    type_encoding: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function matching run_experiment.py style.

    Dataset-specific defaults follow the GAP hyperparameter table:
      WebNLG / LAGRANGE / LAGRANGE_doc / WikiDocKG.
    User-provided arguments override these defaults when not None.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "save"

    dataset_key = (getattr(dataset, "name", "") or "").strip()
    gap_defaults: Dict[str, Dict[str, Any]] = {
        "WebNLG": {
            "num_epochs": 40,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "max_source_length": 256,
            "max_new_tokens": 128,
            "max_nodes": 50,
            "num_beams": 5,
            "length_penalty": 1.0,
        },
        "LAGRANGE": {
            "num_epochs": 20,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "max_source_length": 512,
            "max_new_tokens": 128,
            "max_nodes": 64,
            "num_beams": 5,
            "length_penalty": 1.0,
        },
        "LAGRANGE_doc": {
            "num_epochs": 20,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "max_source_length": 512,
            "max_new_tokens": 256,
            "max_nodes": 64,
            "num_beams": 5,
            "length_penalty": 2.0,
        },
        "WikiDocKG": {
            "num_epochs": 20,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_source_length": 512,
            "max_new_tokens": 512,
            "max_nodes": 512,
            "num_beams": 3,
            "length_penalty": 2.0,
        },
    }

    hp = gap_defaults[dataset_key]

    effective_num_epochs = int(num_epochs if num_epochs is not None else hp["num_epochs"])
    effective_batch_size = int(batch_size if batch_size is not None else hp["batch_size"])
    effective_generation_batch_size = int(
        generation_batch_size if generation_batch_size is not None else effective_batch_size
    )
    effective_max_source_length = int(
        max_source_length if max_source_length is not None else hp["max_source_length"]
    )
    effective_max_new_tokens = int(
        max_new_tokens if max_new_tokens is not None else hp["max_new_tokens"]
    )
    effective_learning_rate = float(learning_rate if learning_rate is not None else 2e-5)

    cfg = GAPConfig(
        model_name_or_path=model_name,
        save_dir=str(save_dir),
        load_if_exists=load_if_exists,

        # Dataset-specific table values
        num_epochs=effective_num_epochs,
        batch_size=effective_batch_size,
        eval_batch_size=effective_generation_batch_size,
        generation_batch_size=effective_generation_batch_size,
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
        max_source_length=effective_max_source_length,
        max_new_tokens=effective_max_new_tokens,
        max_nodes=int(hp["max_nodes"]),
        num_beams=int(hp["num_beams"]),
        length_penalty=float(hp["length_penalty"]),

        # Shared table values
        learning_rate=effective_learning_rate,
        warmup_steps=None,
        warmup_ratio=0.03,
        weight_decay=0.0,
        early_stopping_patience=3,
        select_best_by_bleu=True,
        eval_every_epoch=True,

        relations_as_nodes=True,
        entity_entity=entity_entity,
        entity_relation=entity_relation,
        relation_entity=relation_entity,
        relation_relation=relation_relation,
        type_encoding=type_encoding,
        graph_dropout=0.1,
    )

    logger.info(
        "Using GAP hyperparameters for dataset=%s: %s",
        dataset_key,
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str),
    )

    pipeline = GAPPipeline(dataset=dataset, config=cfg)

    should_train = not (
        load_if_exists
        and save_dir.exists()
        and (save_dir / "config.json").exists()
    )

    if should_train:
        pipeline.train()
    else:
        logger.info("Skipping GAP training because checkpoint exists at %s", save_dir)

    test_results = pipeline.generate_for_split(
        split="test",
        limit=num_examples,
        cache_path=str(output_dir / "cache.json"),
        resume=True,
    )

    out_path = output_dir / "test_predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for result in test_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("GAP predictions saved to %s", out_path)

    return {
        "predictions_path": str(out_path),
        "model_dir": str(save_dir),
        "num_predictions": len(test_results),
        "dataset_key": dataset_key,
        "gap_config": asdict(cfg),
    }
