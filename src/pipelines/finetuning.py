import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.pipelines.base_pipeline import BasePipeline, PipelineConfig, format_triples_as_text
from src.data.kgdataset import KGDataset
from src.models.language_model import LanguageModel

logger = logging.getLogger(__name__)

HF_MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
)

PEFT_ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def _dir_has_hf_model(path: Path) -> bool:
    if path is None or not Path(path).exists():
        return False

    p = Path(path)
    if not (p / "config.json").exists():
        return False

    return any((p / f).exists() for f in HF_MODEL_WEIGHT_FILES)


def _dir_has_peft_adapter(path: Path) -> bool:
    """
    Heuristic: a directory produced by `peft.PeftModel.save_pretrained`.
    """
    if path is None or not Path(path).exists():
        return False

    p = Path(path)
    if not (p / "adapter_config.json").exists():
        return False

    return any((p / f).exists() for f in PEFT_ADAPTER_FILES)


def trained_artifacts_exist(path: Path) -> bool:
    return _dir_has_hf_model(path) or _dir_has_peft_adapter(path)


@dataclass
class FineTuningConfig(PipelineConfig):
    # Training
    max_length: int = 256
    max_new_tokens: int = 128
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    log_every: int = 50

    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.03

    # Prompting
    prompt_template: str = "Triples:\n{triples}\n\nText:"

    # ClearML
    use_clearml: bool = False
    clearml_project: str = "KG2Text"
    clearml_task_name: str = "FineTuningPipeline"
    clearml_tags: Optional[List[str]] = None
    clearml_output_uri: Optional[str] = None

    # Saving / loading
    save_tokenizer: bool = True
    save_training_config: bool = True
    load_if_exists: bool = False

    out_dir: Optional[Path] = None


class TriplesToTextDataset(Dataset):
    """
    Torch Dataset that turns KG triples + reference text into
    (input_ids, attention_mask, labels) for causal LM fine-tuning.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        prompt_template: str,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template

        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer pad_token_id is None. Set tokenizer.pad_token before building the dataset."
            )

        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer eos_token is None. Cannot append EOS to supervised targets."
            )

        pad_id = tokenizer.pad_token_id

        for sample in data:
            triples = sample.get("triples_parsed") or []
            target_text = sample.get("text")

            if not triples or not isinstance(target_text, str):
                continue

            triples_text = format_triples_as_text(triples)
            prompt = self.prompt_template.format(triples=triples_text).strip() + " "
            target = target_text.strip() + tokenizer.eos_token

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)

            input_ids = prompt_ids + target_ids

            if len(input_ids) > max_length:
                excess = len(input_ids) - max_length

                if excess >= len(target_ids):
                    input_ids = input_ids[-max_length:]
                    prompt_len = min(len(prompt_ids), max_length)
                    target_ids = input_ids[prompt_len:]
                else:
                    target_ids = target_ids[excess:]
                    input_ids = prompt_ids + target_ids
                    prompt_len = len(prompt_ids)
            else:
                prompt_len = len(prompt_ids)

            attention_mask = [1] * len(input_ids)

            labels = [-100] * prompt_len + target_ids
            if len(labels) > max_length:
                labels = labels[:max_length]

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len

            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        logger.info("TriplesToTextDataset built with %d examples", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class FineTuningPipeline(BasePipeline):
    """
    Supervised fine-tuning for triples -> text.

    Training lives here.
    Inference (generate_for_split) is inherited from BasePipeline.
    """

    _LORA_TRAINABLE_NAME_TOKENS: Tuple[str, ...] = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    )

    def __init__(
        self,
        lm: LanguageModel,
        dataset: KGDataset,
        config: Optional[FineTuningConfig] = None,
        *,
        dedup_by_graph: bool = False,
    ):
        self.config: FineTuningConfig = config or FineTuningConfig()

        super().__init__(
            lm=lm,
            dataset=dataset,
            config=self.config,
            dedup_by_graph=dedup_by_graph,
        )

        self.train_data: List[Dict[str, Any]] = dataset.load_train()
        self.dev_data: List[Dict[str, Any]] = self.data.get("dev", [])

        self.train_ds = TriplesToTextDataset(
            self.train_data,
            tokenizer=self.lm.tokenizer,
            max_length=self.config.max_length,
            prompt_template=self.config.prompt_template,
        )

        self.dev_ds = (
            TriplesToTextDataset(
                self.dev_data,
                tokenizer=self.lm.tokenizer,
                max_length=self.config.max_length,
                prompt_template=self.config.prompt_template,
            )
            if self.dev_data
            else None
        )

        self._clearml_task = None
        self._clearml_logger = None
        self._init_clearml_if_enabled()

    def build_prompt(self, triples: List[Dict[str, str]]) -> str:
        triples_text = format_triples_as_text(triples)
        return self.config.prompt_template.format(triples=triples_text).strip() + " "

    def extract_final(self, text: str) -> str:
        return (text or "").strip()

    def prompt_fingerprint(self) -> Dict[str, Any]:
        return {
            "pipeline": self.__class__.__name__,
            "prompt_template": self.config.prompt_template,
        }

    @property
    def device(self):
        return self.lm.device

    def _init_clearml_if_enabled(self) -> None:
        if not getattr(self.config, "use_clearml", False):
            return

        logger.info("Connect to ClearML...")

        try:
            from clearml import Task  # type: ignore
        except ImportError:
            logger.warning(
                "ClearML logging requested (use_clearml=True) but clearml is not installed. "
                "Install with: pip install clearml"
            )
            return

        self._clearml_task = Task.init(
            project_name=getattr(self.config, "clearml_project", "KG2Text"),
            task_name=getattr(self.config, "clearml_task_name", "FineTuningPipeline"),
            tags=getattr(self.config, "clearml_tags", None),
            output_uri=getattr(self.config, "clearml_output_uri", None),
        )
        self._clearml_logger = self._clearml_task.get_logger()

        try:
            self._clearml_task.connect(asdict(self.config))
        except Exception:
            self._clearml_task.connect(vars(self.config))

        try:
            self._clearml_task.set_comment(
                f"Model: {getattr(self.lm, 'name', 'unknown')} | "
                f"Train size: {len(self.train_data)} | Dev size: {len(self.dev_data)}"
            )
        except Exception:
            pass

        logger.info("Connected.")

    def _get_trainable_named_parameters(
        self,
        model: torch.nn.Module,
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]

    def _validate_trainable_parameters(
        self,
        model: torch.nn.Module,
    ) -> List[torch.nn.Parameter]:
        """
        Return only trainable parameters and fail early if the model appears to
        be a PEFT/LoRA model but has non-LoRA tensors marked trainable.
        """
        trainable_named_params = self._get_trainable_named_parameters(model)

        if not trainable_named_params:
            raise RuntimeError(
                "No trainable parameters found. "
                "For LoRA/QLoRA, check use_lora=True and target modules."
            )

        trainable_names = [name for name, _ in trainable_named_params]

        logger.info("Trainable parameter tensors: %d", len(trainable_names))
        for name in trainable_names[:100]:
            logger.info("  %s", name)
        if len(trainable_names) > 100:
            logger.info("  ... %d more trainable tensors", len(trainable_names) - 100)

        is_peft_or_lora = (
            hasattr(model, "peft_config")
            or any(
                token in name
                for name in trainable_names
                for token in self._LORA_TRAINABLE_NAME_TOKENS
            )
        )

        if is_peft_or_lora:
            unexpected = [
                name
                for name in trainable_names
                if not any(token in name for token in self._LORA_TRAINABLE_NAME_TOKENS)
            ]

            if unexpected:
                preview = "\n".join(unexpected[:50])
                raise RuntimeError(
                    "Refusing to train: PEFT/LoRA model has non-LoRA trainable parameters.\n"
                    "Only LoRA adapter tensors should have requires_grad=True.\n"
                    f"Unexpected trainable parameters:\n{preview}"
                )

        total_params = sum(param.numel() for param in model.parameters())
        trainable_params_count = sum(param.numel() for _, param in trainable_named_params)

        logger.info(
            "Trainable parameters: %s / %s = %.6f%%",
            f"{trainable_params_count:,}",
            f"{total_params:,}",
            100.0 * trainable_params_count / max(total_params, 1),
        )

        return [param for _, param in trainable_named_params]

    def _build_optimizer(
        self,
        trainable_params: Sequence[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            trainable_params,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_opt_steps: int,
    ):
        scheduler_type = (self.config.lr_scheduler_type or "none").lower()
        warmup_ratio = float(self.config.warmup_ratio)

        if total_opt_steps <= 0:
            raise ValueError("total_opt_steps must be positive.")

        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1], got {self.config.warmup_ratio}."
            )

        warmup_steps = int(round(warmup_ratio * total_opt_steps))
        warmup_steps = min(max(warmup_steps, 0), total_opt_steps)

        logger.info(
            "Scheduler: type=%s, warmup_steps=%d, total_opt_steps=%d",
            scheduler_type,
            warmup_steps,
            total_opt_steps,
        )

        if scheduler_type in {"none", "off", "disabled"}:
            return None

        if scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps,
            )

        if scheduler_type == "constant_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )

        if scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_opt_steps,
            )

        raise ValueError(
            f"Unknown scheduler: {self.config.lr_scheduler_type}. "
            'Expected one of: "constant_with_warmup", "linear", "cosine", "none".'
        )

    @staticmethod
    def _get_current_lr(
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> float:
        if scheduler is not None:
            return float(scheduler.get_last_lr()[0])
        return float(optimizer.param_groups[0]["lr"])

    def train(self) -> None:
        if getattr(self.config, "load_if_exists", False):
            out_dir = getattr(self.config, "out_dir", None)
            if out_dir is not None and trained_artifacts_exist(Path(out_dir)):
                logger.info(
                    "Found existing trained artifacts in %s "
                    "(load_if_exists=True). Skipping fine-tuning.",
                    Path(out_dir).resolve(),
                )
                return

        logger.info("Run training.")

        if len(self.train_ds) == 0:
            raise RuntimeError("Training dataset is empty after preprocessing.")

        model = self.lm.model
        model.train()

        trainable_params = self._validate_trainable_parameters(model)

        train_loader = DataLoader(
            self.train_ds,
            batch_size=int(self.config.batch_size),
            shuffle=True,
        )

        grad_acc = max(int(self.config.gradient_accumulation_steps), 1)
        log_every = max(int(self.config.log_every), 1)
        num_epochs = max(int(self.config.num_epochs), 1)

        steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
        total_opt_steps = steps_per_epoch * num_epochs

        if total_opt_steps <= 0:
            raise RuntimeError(
                f"No optimizer steps will be run. "
                f"len(train_loader)={len(train_loader)}, grad_acc={grad_acc}, epochs={num_epochs}."
            )

        optimizer = self._build_optimizer(trainable_params)
        scheduler = self._build_scheduler(optimizer, total_opt_steps)

        logger.info(
            "Training config: batch_size=%s, grad_acc=%s, epochs=%s, "
            "lr=%s, weight_decay=%s, max_grad_norm=%s, total_opt_steps=%s",
            self.config.batch_size,
            grad_acc,
            num_epochs,
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.max_grad_norm,
            total_opt_steps,
        )

        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss_sum = 0.0
            epoch_micro_steps = 0

            running_loss_sum = 0.0
            running_count = 0

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train | epoch {epoch + 1}/{num_epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            optimizer.zero_grad(set_to_none=True)

            for micro_step, batch in pbar:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = model(**batch)

                raw_loss = float(outputs.loss.detach().item())
                loss = outputs.loss / grad_acc
                loss.backward()

                epoch_loss_sum += raw_loss
                epoch_micro_steps += 1
                running_loss_sum += raw_loss
                running_count += 1

                avg_loss = running_loss_sum / max(running_count, 1)
                lr_before_step = self._get_current_lr(optimizer, scheduler)

                pbar.set_postfix(
                    loss=f"{raw_loss:.4f}",
                    avg=f"{avg_loss:.4f}",
                    lr=f"{lr_before_step:.2e}",
                    step=f"{global_step}/{total_opt_steps}",
                )

                should_step = (micro_step + 1) % grad_acc == 0

                if should_step:
                    clip_grad_norm_(
                        trainable_params,
                        float(self.config.max_grad_norm),
                    )

                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    lr_after_step = self._get_current_lr(optimizer, scheduler)

                    if global_step % log_every == 0:
                        if self._clearml_logger is not None:
                            self._clearml_logger.report_scalar(
                                title="train",
                                series="loss_avg",
                                value=avg_loss,
                                iteration=global_step,
                            )
                            self._clearml_logger.report_scalar(
                                title="train",
                                series="lr",
                                value=lr_after_step,
                                iteration=global_step,
                            )

                        pbar.write(
                            f"[epoch {epoch + 1}] step {global_step}/{total_opt_steps} "
                            f"loss(avg)={avg_loss:.4f} lr={lr_after_step:.2e}"
                        )

                        running_loss_sum = 0.0
                        running_count = 0

            # Handle leftover micro-batches when len(train_loader) is not divisible by grad_acc.
            if len(train_loader) % grad_acc != 0:
                clip_grad_norm_(
                    trainable_params,
                    float(self.config.max_grad_norm),
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_avg_loss = epoch_loss_sum / max(epoch_micro_steps, 1)

            logger.info(
                "Finished epoch %d/%d (avg loss=%.4f, opt_steps=%d)",
                epoch + 1,
                num_epochs,
                epoch_avg_loss,
                global_step,
            )

            if self._clearml_logger is not None:
                self._clearml_logger.report_scalar(
                    title="train",
                    series="loss_epoch_avg",
                    value=epoch_avg_loss,
                    iteration=epoch + 1,
                )

        logger.info("Fine-tuning complete.")

        if self._clearml_task is not None:
            try:
                self._clearml_task.flush()
            except Exception:
                pass

        self._save_model()

    def _save_model(self) -> Path:
        out_dir = self.config.out_dir
        assert out_dir is not None, "FineTuningConfig.out_dir must be set to save artifacts."

        out_dir.mkdir(parents=True, exist_ok=True)

        model = getattr(self.lm.model, "module", self.lm.model)
        model.save_pretrained(out_dir)

        if getattr(self.config, "save_tokenizer", True):
            self.lm.tokenizer.save_pretrained(out_dir)

        if getattr(self.config, "save_training_config", True):
            (out_dir / "fine_tuning_config.json").write_text(
                json.dumps(
                    asdict(self.config),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        logger.info("Saved fine-tuned model to: %s", out_dir.resolve())
        return out_dir