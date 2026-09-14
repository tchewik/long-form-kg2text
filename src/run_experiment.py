import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List

try:
    import unsloth
except ImportError:
    pass

from src.data import keys
import numpy as np
import torch
from dotenv import load_dotenv
from src.data.kgdataset import KGDataset
from src.data.lagrange.lagrange_doc_loader import LagrangeDocLoader
from src.data.lagrange.lagrange_loader import LagrangeLoader
from src.data.webnlg_loader import WebNLGLoader
from src.data.wikidockg.wikidockg_loader import WikiDocKGLoader
from src.models.language_model import (
    LanguageModel,
    LanguageModelConfig,
)
from src.pipelines.cot import (
    CoTPipeline,
    CoTConfig,
)
from src.pipelines.direct import (
    DirectPipeline,
    DirectConfig
)
from src.pipelines.fbf import (
    FactByFactPipeline,
    FactByFactConfig
)
from src.pipelines.finetuning import (
    FineTuningPipeline,
    FineTuningConfig,
    _dir_has_hf_model,
    _dir_has_peft_adapter,
)
from src.pipelines.gap import run_gap_baseline
try:
    from src.pipelines.dualpath import run_dualpath_baseline
except ImportError:
    run_dualpath_baseline = None

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(module)s.py: %(message)s",
)

os.environ["CLEARML_API_HOST"] = keys.CLEARML_API_HOST
os.environ["CLEARML_WEB_HOST"] = keys.CLEARML_WEB_HOST
os.environ["CLEARML_FILES_HOST"] = keys.CLEARML_FILES_HOST
os.environ["CLEARML_API_ACCESS_KEY"] = keys.CLEARML_KEY
os.environ["CLEARML_API_SECRET_KEY"] = keys.CLEARML_SECRET


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LoggingRLTrainer:
    """
    Minimal stub that just logs rewards. This is where you would hook in
    a real PPO / TRL trainer if you want actual RL updates.

    The interface matches the `update_callback` expected by `RLAIFPipeline`.
    """

    def __init__(self):
        self.step = 0

    def __call__(self, prompts: List[str], candidates: List[str], rewards: List[float]):
        self.step += 1
        avg_r = float(np.mean(rewards)) if rewards else 0.0
        logger.debug(
            f"[RLAIF] step={self.step}, batch_size={len(rewards)}, "
            f"avg_reward={avg_r:.3f}"
        )
        # TODO: actual PPO / RL update code.


def write_test_predictions(out_path, results):
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Predictions on test saved to %s", out_path)


def run_rlaif_experiment(
        dataset: KGDataset,
        policy_model_name: str,
        reward_model_name: str,
        output_dir: Path,
        num_epochs: int,
        batch_size: int,
        load_in_4bit: bool,
        load_in_8bit: bool,
        use_lora_policy: bool,
        use_lora_reward: bool,
) -> Dict[str, Any]:
    pass


def run_direct_baseline(
    dataset: KGDataset,
    model_name: str,
    output_dir: Path,
    num_shots: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    aggregation: str,
    num_examples: int,
    generation_batch_size: int
) -> Dict[str, Any]:
    logger.info("=== Direct baseline ====")

    output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = 128
    if dataset.name == "WikiDocKG":
        if not "unsloth" in model_name:
            max_new_tokens = 512
        else:
            max_new_tokens = 256

    timeout_s = 60  # Default
    timeout_s = int(timeout_s / 128. * max_new_tokens)

    temperature = 0.0 if model_name.startswith('openai:') else 0.7

    torch_dtype = "auto"
    repetition_penalty = 1.4

    if model_name.startswith('meta-llama'):
        repetition_penalty = 1.0
        max_new_tokens = 512

    lm_cfg = LanguageModelConfig(
        model_name_or_path=model_name,
        openai_timeout_s = timeout_s,
        openrouter_timeout_s=timeout_s * 2,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_lora=False,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        enable_thinking=False,
        torch_dtype=torch_dtype
    )

    lm = LanguageModel(lm_cfg)

    num_samples = 1
    if (aggregation or "first").lower() in {"majority", "factspotter"}:
        num_samples = 5

    stop_strings = None

    direct_cfg = DirectConfig(
        prompt_style="wikipedia",
        num_samples_per_example=num_samples,
        max_new_tokens=max_new_tokens,
        num_shots=num_shots,
        aggregation=aggregation,
        generation_batch_size=generation_batch_size,
        stop_strings=stop_strings
    )
    direct = DirectPipeline(lm, dataset, direct_cfg)

    results = direct.generate_for_split(
        split="test",
        limit=num_examples,
        cache_path=output_dir / "cache.json",
    )

    out_path = output_dir / "test_predictions.jsonl"
    write_test_predictions(out_path, results)

    return {
        "num_predictions": len(results),
        "predictions_path": str(out_path),
        "aggregation": aggregation,
        "num_samples_per_example": num_samples,
    }


def run_fbf_baseline(
    dataset: KGDataset,
    model_name: str,
    output_dir: Path,
    load_in_4bit: bool,
    load_in_8bit: bool,
    num_examples: int,
    generation_batch_size: int,
    first_stage_only: bool = False,
) -> Dict[str, Any]:
    logger.info("==== FBF baseline ====")

    output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = 64
    if dataset.name == 'LAGRANGE_doc':
        max_new_tokens = 128
    if dataset.name == 'WikiDocKG':
        if not "unsloth" in model_name:
            max_new_tokens = 512
        else:
            max_new_tokens = 256

    timeout_s = 10

    lm_cfg = LanguageModelConfig(
        model_name_or_path=model_name,
        openai_timeout_s=timeout_s,
        openrouter_timeout_s=timeout_s * 2,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_lora=False,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.4,
    )
    lm = LanguageModel(lm_cfg)

    fbf_cfg = FactByFactConfig(
        naturalize=not first_stage_only,
        triple_max_new_tokens=32,
        triple_stop_strings=None,
        naturalize_max_new_tokens=max_new_tokens,
        naturalize_stop_strings=None,
        generation_batch_size=generation_batch_size
    )

    fbf = FactByFactPipeline(lm, dataset, fbf_cfg)

    results = fbf.generate_for_split(
        split="test",
        limit=num_examples,
        cache_path=output_dir / "cache.json",
    )

    out_path = output_dir / "test_predictions.jsonl"
    write_test_predictions(out_path, results)

    return {
        "num_predictions": len(results),
        "predictions_path": str(out_path),
    }


def run_cot_baseline(
        dataset: KGDataset,
        model_name: str,
        output_dir: Path,
        load_in_4bit: bool,
        load_in_8bit: bool,
        num_shots: int,
        aggregation: str,
        num_examples: int,
        num_samples: int = 1,
        generation_batch_size: int = 1
) -> Dict[str, Any]:
    logger.info("==== CoT baseline ====")

    output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = 512
    if dataset.name == 'LAGRANGE_doc' or dataset.name == 'WikiDocKG':
        if not "unsloth" in model_name:
            max_new_tokens = 1024
        else:
            max_new_tokens = 256

    timeout_s = 20  # Default 120; 40 works
    timeout_s = int(timeout_s / 128. * max_new_tokens)

    lm_cfg = LanguageModelConfig(
        model_name_or_path=model_name,
        openai_timeout_s=timeout_s,
        openrouter_timeout_s=timeout_s * 2,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_lora=False,
        device_map="auto",
        max_new_tokens=max_new_tokens,

    )
    lm = LanguageModel(lm_cfg)

    if (aggregation or "first").lower() in {"majority", "factspotter"}:
        num_samples = 5

    temperature = 0.0 if model_name.startswith('openai:') else 0.7

    cot_cfg = CoTConfig(
        num_samples_per_example=num_samples,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        aggregation=aggregation,
        num_shots=num_shots,
    )
    cot = CoTPipeline(lm, dataset, cot_cfg)

    results = cot.generate_for_split(
        split="test",
        limit=num_examples,
        cache_path=output_dir / "cache.json"
    )

    out_path = output_dir / "test_predictions.jsonl"
    write_test_predictions(out_path, results)

    return {
        "num_predictions": len(results),
        "predictions_path": str(out_path),
    }


def run_finetuning_baseline(
        dataset: KGDataset,
        model_name: str,
        output_dir: Path,
        load_in_4bit: bool,
        load_in_8bit: bool,
        use_lora: bool,
        num_epochs: int,
        num_examples: int,
        use_clearml: bool = False,
        clearml_project: str = "KG2Text",
        clearml_task_name: str = "FineTuningPipeline",
        clearml_tags: Optional[List[str]] = None,
        clearml_output_uri: Optional[str] = None,
        load_if_exists: bool = False,
        tuning_batch_size: int = 1,
        generation_batch_size: int = 1,
        aggregation: str = "first",

        learning_rate: Optional[float] = None,
        lr_scheduler_type: Optional[str] = None,
        warmup_ratio: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        gradient_accumulation_steps: int = 1,

        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[Any] = None,

        torch_dtype: str = "auto",
) -> Dict[str, Any]:
    logger.info("==== Supervised fine-tuning baseline ====")

    if load_in_4bit and load_in_8bit:
        raise ValueError("Set only one of load_in_4bit=True or load_in_8bit=True, not both.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = output_dir / "save"

    max_new_tokens = 128
    if dataset.name == "LAGRANGE_doc":
        max_new_tokens = 256
    elif dataset.name == "WikiDocKG":
        if not "unsloth" in model_name:
            max_new_tokens = 512
        else:
            max_new_tokens = 256

    max_len = max_new_tokens * 2

    is_adapter_training = bool(use_lora)
    is_quantized_adapter_training = bool(use_lora and (load_in_4bit or load_in_8bit))

    effective_learning_rate = (
        learning_rate
        if learning_rate is not None
        else 2e-4
        if is_adapter_training
        else 5e-5
    )

    effective_lr_scheduler_type = (
        lr_scheduler_type
        if lr_scheduler_type is not None
        else "constant_with_warmup"
        if is_adapter_training
        else "linear"
    )

    effective_warmup_ratio = (
        warmup_ratio
        if warmup_ratio is not None
        else 0.03
        if is_adapter_training
        else 0.10
    )

    effective_max_grad_norm = (
        max_grad_norm
        if max_grad_norm is not None
        else 0.3
        if is_adapter_training
        else 1.0
    )

    if is_quantized_adapter_training:
        logger.info(
            "Using QLoRA-style defaults: lr=%s, scheduler=%s, warmup_ratio=%s, max_grad_norm=%s",
            effective_learning_rate,
            effective_lr_scheduler_type,
            effective_warmup_ratio,
            effective_max_grad_norm,
        )
    elif is_adapter_training:
        logger.info(
            "Using LoRA adapter training: lr=%s, scheduler=%s, warmup_ratio=%s, max_grad_norm=%s",
            effective_learning_rate,
            effective_lr_scheduler_type,
            effective_warmup_ratio,
            effective_max_grad_norm,
        )
    else:
        logger.info(
            "Using full/frozen model fine-tuning path: lr=%s, scheduler=%s, warmup_ratio=%s, max_grad_norm=%s",
            effective_learning_rate,
            effective_lr_scheduler_type,
            effective_warmup_ratio,
            effective_max_grad_norm,
        )

    lm_cfg = LanguageModelConfig(
        model_name_or_path=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch_dtype,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    do_train = True
    lm = None

    if load_if_exists and save_dir.exists():
        # Case 1: full HF model dir: config.json + full model weights.
        if _dir_has_hf_model(save_dir):
            logger.info(
                "Found existing fine-tuned full model at %s. Loading and skipping training.",
                save_dir.resolve(),
            )

            lm_cfg = LanguageModelConfig(
                model_name_or_path=str(save_dir),
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                use_lora=False,
                device_map="auto",
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            lm = LanguageModel(lm_cfg)
            do_train = False

        # Case 2: PEFT adapter dir: adapter_config.json + adapter weights.
        elif _dir_has_peft_adapter(save_dir):
            logger.info(
                "Found existing LoRA/PEFT adapter at %s. Loading adapter and skipping training.",
                save_dir.resolve(),
            )

            base_cfg = LanguageModelConfig(
                model_name_or_path=model_name,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                use_lora=False,
                device_map="auto",
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            lm = LanguageModel(base_cfg)

            # Prefer tokenizer saved next to the adapter, if present.
            try:
                from transformers import AutoTokenizer  # type: ignore

                lm.tokenizer = AutoTokenizer.from_pretrained(
                    str(save_dir),
                    use_fast=True,
                    padding_side="left",
                )

                if lm.tokenizer.pad_token is None and lm.tokenizer.eos_token is not None:
                    lm.tokenizer.pad_token = lm.tokenizer.eos_token

                if getattr(lm, "model", None) is not None and lm.tokenizer.pad_token_id is not None:
                    lm.model.config.pad_token_id = lm.tokenizer.pad_token_id

            except Exception as e:
                logger.warning(
                    "Could not load tokenizer from %s: %s. Using base tokenizer.",
                    save_dir,
                    e,
                )

            try:
                from peft import PeftModel  # type: ignore

                base_model = getattr(lm.model, "module", lm.model)

                try:
                    lm.model = PeftModel.from_pretrained(
                        base_model,
                        str(save_dir),
                        is_trainable=False,
                    )
                except TypeError:
                    # Older PEFT versions may not expose is_trainable.
                    lm.model = PeftModel.from_pretrained(
                        base_model,
                        str(save_dir),
                    )

                lm.model.eval()

            except Exception as e:
                logger.warning(
                    "Failed to load PEFT adapter from %s: %s. Falling back to training.",
                    save_dir,
                    e,
                )
                do_train = True
                lm = None
            else:
                do_train = False

    if do_train:
        lm = LanguageModel(lm_cfg)

    assert lm is not None, "LanguageModel was not initialized."

    num_samples = 1
    if (aggregation or "first").lower() in {"majority", "factspotter"}:
        num_samples = 5

    gradient_accumulation_steps = 1
    if dataset.name in ("WebNLG", "LAGRANGE"):
        gradient_accumulation_steps = 4
    elif dataset.name == "LAGRANGE_doc":
        gradient_accumulation_steps = 2

    ft_kwargs = dict(
        batch_size=tuning_batch_size,
        num_epochs=num_epochs,
        max_length=max_len,
        max_new_tokens=max_new_tokens,
        learning_rate=effective_learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=effective_max_grad_norm,
        log_every=50,

        lr_scheduler_type=effective_lr_scheduler_type,
        warmup_ratio=effective_warmup_ratio,

        use_clearml=use_clearml,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_tags=clearml_tags,
        clearml_output_uri=clearml_output_uri,

        out_dir=save_dir,
        load_if_exists=load_if_exists,
        generation_batch_size=generation_batch_size,

        num_samples_per_example=num_samples,
        aggregation=aggregation,
    )

    supported_ft_fields = getattr(FineTuningConfig, "__dataclass_fields__", None)
    if supported_ft_fields is not None:
        unsupported = sorted(set(ft_kwargs) - set(supported_ft_fields))
        if unsupported:
            logger.warning(
                "FineTuningConfig does not expose these fields yet; dropping them: %s",
                unsupported,
            )
            ft_kwargs = {
                key: value
                for key, value in ft_kwargs.items()
                if key in supported_ft_fields
            }

    ft_cfg = FineTuningConfig(**ft_kwargs)

    pipeline = FineTuningPipeline(lm, dataset, ft_cfg)

    if do_train:
        pipeline.train()
    else:
        logger.info("Skipping fine-tuning because trained artifacts were loaded.")

    # dev_out = output_dir / "dev_predictions.jsonl"
    # dev_results = pipeline.generate_for_split(
    #     split="dev",
    #     limit=32,
    #     cache_path=str(dev_out),
    #     resume=False,
    # )

    # with dev_out.open("w", encoding="utf-8") as f:
    #     for result in dev_results:
    #         f.write(json.dumps(result, ensure_ascii=False) + "\n")
    #
    # logger.info("Fine-tuning dev predictions saved to %s", dev_out)

    test_results = pipeline.generate_for_split(
        split="test",
        limit=num_examples,
        cache_path=str(output_dir / "cache.json"),
        resume=False,
    )

    out_path = output_dir / "test_predictions.jsonl"
    write_test_predictions(out_path, test_results)

    return {
        "predictions_path": str(out_path),
        "model_dir": str(ft_cfg.out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single seed experiment on KG-to-text dataset.",
    )

    parser.add_argument("--dataset", type=str, required=True,
                        help="KG dataset name (from: WebNLG, Lagrange, Lagrange_doc, WikiDocKG).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write experiment outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true", help="Removes cached outputs in the output_dir.")

    parser.add_argument("--run_direct", action="store_true", help="Run direct prompting baseline.")
    parser.add_argument("--run_fbf", action="store_true", help="Run Fact-by-Fact (FBF) baseline.")
    parser.add_argument("--run_cot", action="store_true", help="Run CoT baseline.")
    parser.add_argument("--run_finetune", action="store_true", help="Run supervised fine-tuning baseline.")
    parser.add_argument("--run_rlaif", action="store_true", help="Run RLAIF pipeline.")
    parser.add_argument("--run_gap", action="store_true", help="Run GAP graph-aware fine-tuning baseline.")
    parser.add_argument("--run_dualpath", action="store_true", help="Run LREC-COLING 2024 DualPath encoder + alignment + guidance baseline.")

    parser.add_argument("--policy_model_name", type=str, default="gpt2",
                        help="HF model name/path for the RLAIF policy.")
    parser.add_argument("--reward_model_name", type=str, default="gpt2",
                        help="HF model name/path for the RLAIF reward model.")
    parser.add_argument("--cot_model_name", type=str, default=None,
                        help="HF model for CoT baseline (defaults to policy_model_name if None).")
    parser.add_argument("--finetune_model_name", type=str, default=None,
                        help="HF model for SFT baseline (defaults to policy_model_name if None).")

    # general model options
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Enable 4-bit quantization for memory savings.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Enable 8-bit quantization for memory savings.")
    parser.add_argument("--policy_use_lora", action="store_true",
                        help="Use LoRA for policy model.")
    parser.add_argument("--reward_use_lora", action="store_true",
                        help="Use LoRA for reward model.")
    parser.add_argument("--finetune_use_lora", action="store_true",
                        help="Use LoRA for SFT baseline model.")

    # FBF params
    parser.add_argument("--fbf_first_stage_only", action="store_true",
                        help="Return first-stage FBF verbalizations without discourse naturalization.")

    # CoT params
    parser.add_argument("--num_shots", type=int, default=0,
                        help="Number of shots for few-shot CoT.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples for self-consistency.")
    parser.add_argument("--aggregation", type=str, default="first",
                        help="Aggregation type for self-consistency.")

    # Finetuning params
    parser.add_argument("--tuning_batch_size", type=int, default=1,
                        help="Batch size for finetuning")

    # RLAIF params
    parser.add_argument("--rlaif_epochs", type=int, default=1,
                        help="Number of RLAIF epochs.")
    parser.add_argument("--rlaif_batch_size", type=int, default=8,
                        help="RLAIF batch size.")

    parser.add_argument("--num_test_examples", type=int, default=-1,
                        help="Number of test examples, or -1 for all.")
    parser.add_argument("--finetune_epochs", type=int, default=1,
                        help="Number of epochs for SFT baseline.")
    parser.add_argument("--finetune_load_if_exists", action="store_true",
                        help="If output_dir/finetune/save already contains a trained model (or PEFT adapter), load it and skip training.")
    parser.add_argument("--generation_batch_size", type=int, default=1,
                        help="Batch size for inference")

    # GAP params
    parser.add_argument("--gap_model_name", type=str, default=None,
                        help="Seq2seq model for GAP, defaults to finetune_model_name or policy_model_name.")
    parser.add_argument("--gap_load_if_exists", action="store_true")

    # DualPath params
    parser.add_argument("--dualpath_model_name", type=str, default=None,
                        help="Seq2seq model for DualPath, defaults to gap_model_name, finetune_model_name, or policy_model_name.")
    parser.add_argument("--dualpath_load_if_exists", action="store_true")
    parser.add_argument("--dualpath_copy_loss_weight", type=float, default=0.05,
                        help="Auxiliary copy-guidance loss weight for the DualPath guidance module.")

    return parser.parse_args()


def load_dataset(dataset_name: str, random_state: float):
    if dataset_name == 'webnlg':
        logger.info('Loading WebNLG v3.0 ...')
        return WebNLGLoader(data_dir='data/webnlg2023/', random_state=random_state)

    if dataset_name == 'lagrange':
        logger.info('Loading LAGRANGE ...')
        return LagrangeLoader(data_dir='data/lagrange/', random_state=random_state)

    if dataset_name == 'lagrange_doc':
        logger.info('Loading LAGRANGE_doc ...')
        return LagrangeDocLoader(data_dir='data/lagrange/', random_state=random_state)

    if dataset_name == 'wikidockg':
        logger.info('Loading WikiDocKG ...')
        return WikiDocKGLoader(data_dir='data/wikidockg/', random_state=random_state)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset.strip().lower(), random_state=args.seed)

    summary: Dict[str, Any] = {
        "args": vars(args),
        "results": {},
    }

    if args.run_direct:
        if args.num_shots == 0:
            method_out_dir = output_dir / "direct"
        else:
            method_out_dir = output_dir / f"direct-{args.num_shots}-shot"

        if args.no_cache and os.path.isfile(method_out_dir / 'cache.json'):
            os.remove(method_out_dir / 'cache.json')

        res_direct = run_direct_baseline(
            dataset=dataset,
            model_name=args.policy_model_name,
            output_dir=method_out_dir,
            num_shots=args.num_shots,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            aggregation=args.aggregation,
            num_examples=args.num_test_examples,
            generation_batch_size=args.generation_batch_size
        )
        summary["results"]["direct"] = res_direct


    if args.run_fbf:
        method_out_dir = output_dir / "fbf"

        if args.no_cache and os.path.isfile(method_out_dir / 'cache.json'):
            os.remove(method_out_dir / 'cache.json')

        fbf_model_name = args.cot_model_name or args.policy_model_name

        res_fbf = run_fbf_baseline(
            dataset=dataset,
            model_name=fbf_model_name,
            output_dir=method_out_dir,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            num_examples=args.num_test_examples,
            generation_batch_size=args.generation_batch_size,
            first_stage_only=args.fbf_first_stage_only,
        )

        summary["results"]["fbf"] = res_fbf


    if args.run_cot:
        if args.num_shots == 0:
            method_out_dir = output_dir / "cot"
        else:
            method_out_dir = output_dir / f"cot-{args.num_shots}-shot"

        if args.no_cache and os.path.isfile(method_out_dir / 'cache.json'):
            os.remove(method_out_dir / 'cache.json')

        cot_model_name = args.cot_model_name or args.policy_model_name
        res_cot = run_cot_baseline(
            dataset=dataset,
            model_name=cot_model_name,
            output_dir=method_out_dir,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            num_shots=args.num_shots,
            aggregation=args.aggregation,
            num_samples=args.num_samples,
            num_examples=args.num_test_examples,
        )
        summary["results"]["cot"] = res_cot

    if args.run_finetune:
        ft_model_name = args.finetune_model_name or args.policy_model_name
        clearml_task_name = (f'FTPipeline-{args.dataset.strip()}-seed{args.seed}-'
                             + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                             )
        res_ft = run_finetuning_baseline(
            dataset=dataset,
            model_name=ft_model_name,
            output_dir=output_dir / "finetune",
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            use_lora=args.finetune_use_lora,
            num_examples=args.num_test_examples,
            num_epochs=args.finetune_epochs,
            clearml_task_name=clearml_task_name,
            load_if_exists=args.finetune_load_if_exists,
            tuning_batch_size=args.tuning_batch_size,
            generation_batch_size=args.generation_batch_size,
            aggregation=args.aggregation
        )
        summary["results"]["finetune"] = res_ft

    if args.run_gap:
        gap_model_name = args.gap_model_name or args.finetune_model_name or args.policy_model_name

        res_gap = run_gap_baseline(
            dataset=dataset,
            model_name=gap_model_name,
            output_dir=output_dir / "gap",
            num_epochs=args.finetune_epochs,
            num_examples=args.num_test_examples,
            load_if_exists=args.gap_load_if_exists,
            batch_size=args.tuning_batch_size,
            generation_batch_size=args.generation_batch_size,
        )

        summary["results"]["gap"] = res_gap


    if args.run_dualpath:
        if run_dualpath_baseline is None:
            raise RuntimeError("DualPath was referenced by the development code but is not included in this public source archive.")
        dualpath_model_name = (
            args.dualpath_model_name
            or args.gap_model_name
            or args.finetune_model_name
            or args.policy_model_name
        )

        res_dualpath = run_dualpath_baseline(
            dataset=dataset,
            model_name=dualpath_model_name,
            output_dir=output_dir / "dualpath",
            num_epochs=args.finetune_epochs,
            num_examples=args.num_test_examples,
            load_if_exists=args.dualpath_load_if_exists,
            batch_size=args.tuning_batch_size,
            generation_batch_size=args.generation_batch_size,
            copy_loss_weight=args.dualpath_copy_loss_weight,
        )

        summary["results"]["dualpath"] = res_dualpath

    summary_path = output_dir / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Experiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
