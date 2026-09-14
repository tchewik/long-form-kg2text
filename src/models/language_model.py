import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pathlib import Path

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    _PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    _PEFT_AVAILABLE = False

from src.models.openrouter_language_model import (
    OpenRouterLanguageModel,
    OpenRouterLanguageModelConfig,
)

from src.models.openai_language_model import (
    OpenAILanguageModel,
    OpenAILanguageModelConfig,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(module)s.py: %(message)s",
)

_INTERNAL_PROMPT_MARKER_RE = re.compile(
    r"<\|[^|>]*(?:USER|ASSISTANT|CHATBOT|PROMPT|RESPONSE|END)[^|>]*\|>|"
    r"<start_of_turn>|<end_of_turn>|"
    r"</?s>",
    flags=re.IGNORECASE,
)

def _strip_internal_prompt_markers(self, text: str) -> str:
    return self._INTERNAL_PROMPT_MARKER_RE.sub("", text).strip()


@dataclass
class LanguageModelConfig:
    model_name_or_path: str
    max_new_tokens: int = 128
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    enable_thinking: bool = False
    device_map: str = "auto"

    use_chat_template: bool = True
    chat_template_delimiter: str = "\n<|USER_PROMPT|>\n"
    clean_generated_text: bool = True

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "auto"

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[Union[List[str], str]] = "all-linear"

    gradient_checkpointing: bool = False
    pad_token_as_eos: bool = True

    # OpenRouter options
    openrouter_api_key: Optional[str] = None
    openrouter_http_referer: Optional[str] = None
    openrouter_x_title: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    openrouter_timeout_s: float = 120.0
    openrouter_max_retries: int = 6
    openrouter_tokenizer_name_or_path: str = "gpt2"
    openrouter_system_prompt: Optional[str] = None

    # OpenAI options
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_timeout_s: float = 60.0
    openai_max_retries: int = 6
    openai_tokenizer_name_or_path: str = "gpt2"
    openai_system_prompt: Optional[str] = None

    def to_generation_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )

        if self.do_sample:
            kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )

        return kwargs


class LanguageModel:
    """
    Unified wrapper:
      - HF Causal LM (local)
      - OpenRouter chat-completions (remote) when model_name_or_path starts with "openrouter:"
      - OpenAI chat-completions (remote) when model_name_or_path starts with "openai:"

    LoRA / QLoRA behavior:
      - If use_lora=True and load_in_4bit/load_in_8bit=True, the base model is prepared
        with PEFT's prepare_model_for_kbit_training before adapters are attached.
      - After adapter attachment, a hard audit verifies that only LoRA adapter tensors
        are trainable. This prevents accidental full-model fine-tuning.
    """

    OPENROUTER_PREFIX = "openrouter:"
    OPENAI_PREFIX = "openai:"

    _LORA_TRAINABLE_NAME_TOKENS: Tuple[str, ...] = (
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    )

    _GEMMA_CONTROL_TOKEN_RE = re.compile(
        r"</?s>|"
        r"<\|USER_PROMPT\|>|<\|END_USER_PROMPT\|>|"
        r"<\|ASSISTANT_PROMPT\|>|<\|END_ASSISTANT_PROMPT\|>|"
        r"<\|USER_RESPONSE\|>|<\|END_USER_RESPONSE\|>",
        flags=re.IGNORECASE,
    )

    def __init__(self, config: LanguageModelConfig):
        self.config = config
        logger.info(
            "Initializing LanguageModel with config: %s",
            json.dumps(asdict(config), indent=4, default=str),
        )

        self._backend = None
        self.tokenizer = None
        self.model = None

        model_name = (config.model_name_or_path or "").strip()

        if model_name.startswith(self.OPENROUTER_PREFIX):
            self._init_openrouter_backend(model_name)
            return

        if model_name.startswith(self.OPENAI_PREFIX):
            self._init_openai_backend(model_name)
            return

        self._init_transformers_backend()

    @staticmethod
    def _mask_secret(value: Optional[str], *, visible_prefix: int = 6, visible_suffix: int = 4) -> Optional[str]:
        if value is None:
            return None

        value = str(value)
        if not value:
            return value

        if len(value) <= visible_prefix + visible_suffix:
            return "*" * len(value)

        return f"{value[:visible_prefix]}...{value[-visible_suffix:]}"

    @classmethod
    def _safe_config_for_logging(cls, config: OpenAILanguageModelConfig) -> Dict[str, Any]:
        payload = asdict(config)
        payload["api_key"] = cls._mask_secret(payload.get("api_key"))
        return payload

    def _gemma_bad_words_ids(self) -> List[List[int]]:
        """
        Prevent Gemma from generating project-specific control markers.

        This is Gemma-only. Other models keep previous behavior.
        """
        markers = [
            "<|USER_PROMPT|>",
            "<|USER_PROMPT_END|>",
            "<|END_USER_PROMPT|>",
            "<|ASSISTANT_PROMPT|>",
            "<|ASSISTANT_PROMPT_END|>",
            "<|END_ASSISTANT_PROMPT|>",
            "<|USER_RESPONSE|>",
            "<|END_USER_RESPONSE|>",
            "<|CHATBOT_PROMPT|>",
            "<|END|>",
        ]

        bad_words_ids = []

        for marker in markers:
            token_ids = self.tokenizer(
                marker,
                add_special_tokens=False,
            ).input_ids

            if token_ids:
                bad_words_ids.append(token_ids)

        return bad_words_ids

    def _maybe_load_gemma_chat_template(self) -> None:
        """
        Gemma 4 compatibility shim.

        Some Gemma 4 HF repos/snapshots provide chat_template.jinja as a separate
        file instead of embedding it in tokenizer_config.json. In that case,
        tokenizer.chat_template is None and apply_chat_template() cannot work.

        This method is intentionally Gemma-only and only mutates tokenizer.chat_template
        when it is missing.
        """
        if not self._is_gemma_model():
            return

        if getattr(self.tokenizer, "chat_template", None):
            return

        candidate_paths = []

        model_path = Path(self.config.model_name_or_path)

        # Local model directory case.
        if model_path.exists():
            candidate_paths.append(model_path / "chat_template.jinja")

        # HF cache / downloaded snapshot case.
        try:
            from huggingface_hub import hf_hub_download

            candidate_paths.append(
                Path(
                    hf_hub_download(
                        repo_id=self.config.model_name_or_path,
                        filename="chat_template.jinja",
                    )
                )
            )
        except Exception as exc:
            logger.warning(
                "Could not download Gemma chat_template.jinja from Hugging Face: %s",
                exc,
            )

        for path in candidate_paths:
            if path.exists():
                template = path.read_text(encoding="utf-8")
                if template.strip():
                    self.tokenizer.chat_template = template
                    logger.info("Loaded Gemma chat template from %s", path)
                    return

        logger.warning(
            "Gemma model detected, but no chat_template was available and "
            "chat_template.jinja could not be loaded. Falling back to raw prompts."
        )

    def _init_openrouter_backend(self, model_name: str) -> None:
        config = self.config
        or_model = model_name[len(self.OPENROUTER_PREFIX):].strip()

        if not or_model:
            raise ValueError(
                'model_name_or_path starts with "openrouter:" but no model was provided '
                '(expected e.g. "openrouter:openai/gpt-4.1-mini")'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(config, "openrouter_tokenizer_name_or_path", "gpt2"),
            use_fast=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None and config.pad_token_as_eos:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._backend = OpenRouterLanguageModel(
            OpenRouterLanguageModelConfig(
                model=or_model,
                api_key=config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
                base_url=(
                    config.openrouter_base_url
                    or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                ),
                http_referer=(
                    config.openrouter_http_referer
                    or os.environ.get("OPENROUTER_HTTP_REFERER")
                ),
                x_title=config.openrouter_x_title or os.environ.get("OPENROUTER_X_TITLE"),
                max_new_tokens=config.max_new_tokens,
                repetition_penalty=config.repetition_penalty,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout_s=config.openrouter_timeout_s,
                max_retries=getattr(config, "openrouter_max_retries", 6),
                system_prompt=getattr(config, "openrouter_system_prompt", None),
            )
        )

        self.name = self._backend.name
        logger.info("LanguageModel using OpenRouter backend: %s", self.name)

    def _init_openai_backend(self, model_name: str) -> None:
        config = self.config
        oa_model = model_name[len(self.OPENAI_PREFIX):].strip()

        if not oa_model:
            raise ValueError(
                'model_name_or_path starts with "openai:" but no model was provided '
                '(expected e.g. "openai:openai/gpt-4.1-mini")'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(config, "openai_tokenizer_name_or_path", "gpt2"),
            use_fast=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None and config.pad_token_as_eos:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._backend = OpenAILanguageModel(
            OpenAILanguageModelConfig(
                model=oa_model,
                api_key=config.openai_api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=(
                    config.openai_base_url
                    or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                ),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout_s=config.openai_timeout_s,
                max_retries=getattr(config, "openai_max_retries", 6),
                system_prompt=getattr(config, "openai_system_prompt", None),
            )
        )

        self.name = self._backend.name
        logger.info("LanguageModel using OpenAI backend: %s", self.name)

    def _is_qwen_model(self) -> bool:
        return self.config.model_name_or_path.startswith("Qwen/Qwen3.5-")

    def _is_gemma_model(self) -> bool:
        return "gemma" in (self.config.model_name_or_path or "").lower()

    def _render_gemma_chat_prompts(
            self,
            prompts: List[Union[str, List[Dict[str, str]]]],
    ) -> List[str]:
        conversations = self._normalize_messages_for_gemma(prompts)

        if getattr(self.tokenizer, "chat_template", None):
            rendered = []

            for messages in conversations:
                try:
                    rendered.append(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=bool(self.config.enable_thinking),
                        )
                    )
                except TypeError:
                    rendered.append(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

            return rendered

        logger.warning(
            "Gemma tokenizer has no chat_template. Using manual Gemma chat format."
        )

        return [
            self._manual_gemma_chat_template(messages)
            for messages in conversations
        ]

    def _init_transformers_backend(self) -> None:
        config = self.config

        if config.load_in_4bit and config.load_in_8bit:
            raise ValueError("Set only one of load_in_4bit=True or load_in_8bit=True, not both.")

        quantization_config = self._build_quantization_config()
        dtype = self._resolve_torch_dtype(config.torch_dtype)

        self.name = os.path.basename(config.model_name_or_path.rstrip("/"))

        if self._is_gemma_model():
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                use_fast=True,
                padding_side="left",
            )

            self._maybe_load_gemma_chat_template()

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                use_fast=True,
                padding_side="left",
                enable_thinking=config.enable_thinking,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            device_map=config.device_map,
            quantization_config=quantization_config,
            dtype=dtype,
        )

        self._configure_padding()

        if config.use_lora:
            self._prepare_and_wrap_with_lora()
        else:
            self._maybe_enable_gradient_checkpointing()

        self.model.eval()

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> Union[str, torch.dtype]:
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "auto":
            return "auto"

        raise ValueError(
            f"Unsupported torch_dtype={dtype_name!r}. "
            'Expected one of: "auto", "float16", "bfloat16", "float32".'
        )

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        cfg = self.config

        if not (cfg.load_in_4bit or cfg.load_in_8bit):
            return None

        compute_dtype = (
            torch.float16
            if cfg.torch_dtype in ("auto", "float16")
            else torch.bfloat16
            if cfg.torch_dtype == "bfloat16"
            else torch.float32
        )

        if cfg.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    def _configure_padding(self) -> None:
        cfg = self.config

        if not cfg.pad_token_as_eos:
            return

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has neither pad_token nor eos_token; "
                    "cannot set pad_token_as_eos=True."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _maybe_enable_gradient_checkpointing(self) -> None:
        if not self.config.gradient_checkpointing:
            return

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        else:
            logger.warning(
                "gradient_checkpointing=True, but model has no gradient_checkpointing_enable()."
            )

    def _prepare_and_wrap_with_lora(self) -> None:
        cfg = self.config

        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with `pip install peft`.")

        if cfg.gradient_checkpointing and hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if cfg.load_in_4bit or cfg.load_in_8bit:
            logger.info("Preparing k-bit model for LoRA/QLoRA training.")
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )
        else:
            self._freeze_base_model_parameters()
            self._maybe_enable_gradient_checkpointing()

        self._wrap_with_lora()
        self._assert_only_lora_trainable()

    def _freeze_base_model_parameters(self) -> None:
        """
        Defensive freeze for non-quantized LoRA.

        PEFT normally freezes the base model during LoRA wrapping, but freezing
        before adapter injection makes the intent explicit and prevents accidental
        full-model fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def _wrap_with_lora(self) -> None:
        cfg = self.config

        if cfg.lora_target_modules is None:
            target_modules: Union[List[str], str] = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
        else:
            target_modules = cfg.lora_target_modules

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_cfg)
        logger.info("Wrapped base model with LoRA adapter. target_modules=%s", target_modules)

    def _assert_only_lora_trainable(self) -> None:
        """
        Hard safety check: when use_lora=True, training must update only LoRA
        adapter tensors. This catches accidental full-model tuning before the
        training loop runs.
        """
        trainable_names = [
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

        if not trainable_names:
            raise RuntimeError("LoRA is enabled, but no trainable parameters were found.")

        unexpected = [
            name for name in trainable_names
            if not any(token in name for token in self._LORA_TRAINABLE_NAME_TOKENS)
        ]

        if unexpected:
            preview = "\n".join(unexpected[:50])
            raise RuntimeError(
                "Refusing to continue: non-LoRA parameters are trainable.\n"
                "Only LoRA adapter tensors should have requires_grad=True.\n"
                f"Unexpected trainable parameters:\n{preview}"
            )

        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel()
            for param in self.model.parameters()
            if param.requires_grad
        )

        logger.info(
            "LoRA trainable parameters: %s / %s = %.6f%%",
            f"{trainable_params:,}",
            f"{total_params:,}",
            100.0 * trainable_params / max(total_params, 1),
        )

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    @property
    def device(self):
        if self._backend is not None:
            return getattr(self._backend, "device", "remote")
        return next(self.model.parameters()).device

    @property
    def last_generation_usages(self) -> List[Dict[str, int]]:
        """
        Usage records from the most recent remote backend generate() call.

        For OpenAI-backed models, this is one usage dict per prompt/output.
        For local HF models and remote backends that do not expose usage, this
        returns an empty list.
        """
        if self._backend is None:
            return []
        return list(getattr(self._backend, "last_generation_usages", []) or [])

    @property
    def token_usage(self) -> Dict[str, int]:
        """
        Cumulative token usage for remote backends that expose it.

        For OpenAI-backed models, this is the cumulative usage for the lifetime
        of the OpenAILanguageModel instance. For local HF models and remote
        backends that do not expose usage, this returns an empty dict.
        """
        if self._backend is None:
            return {}

        usage = getattr(self._backend, "token_usage", None)
        return dict(usage) if isinstance(usage, dict) else {}

    @staticmethod
    def _flat_prompt_to_messages(
        prompt: str,
        delimiter: str = "\n<|USER_PROMPT|>\n",
    ) -> List[Dict[str, str]]:
        parts = prompt.split(delimiter, 1)

        if len(parts) == 2:
            system_part, user_part = parts
            system_part = system_part.strip()
            user_part = user_part.strip()

            messages = []
            if system_part:
                messages.append({"role": "system", "content": system_part})
            messages.append({"role": "user", "content": user_part})
            return messages

        return [{"role": "user", "content": prompt.strip()}]

    @staticmethod
    def _normalize_for_chat_template(
        prompts: List[Union[str, List[Dict[str, str]]]],
        delimiter: str = "\n<|USER_PROMPT|>\n",
    ) -> List[List[Dict[str, str]]]:
        conversations = []

        for prompt in prompts:
            if isinstance(prompt, str):
                conversations.append(
                    LanguageModel._flat_prompt_to_messages(
                        prompt,
                        delimiter=delimiter,
                    )
                )
            else:
                conversations.append(prompt)

        return conversations

    def _clean_gemma_generation(self, text: str) -> str:
        text = text.strip()

        # Cut at the first turn/control boundary if Gemma continues the dialogue.
        stop_markers = [
            "<end_of_turn>",
            "<start_of_turn>user",
            "<start_of_turn>model",
            "<|USER_PROMPT|>",
            "<|USER_PROMPT_END|>",
            "<|END_USER_PROMPT|>",
            "<|ASSISTANT_PROMPT|>",
            "<|CHATBOT_PROMPT|>",
            "<|END|>",
        ]

        lower = text.lower()
        cuts = [
            lower.find(marker.lower())
            for marker in stop_markers
            if lower.find(marker.lower()) >= 0
        ]

        if cuts:
            text = text[:min(cuts)]

        text = self._strip_internal_prompt_markers(text)

        return text.strip()

    def _strip_internal_prompt_markers(self, text: str) -> str:
        return self._GEMMA_CONTROL_TOKEN_RE.sub("", text).strip()

    def _normalize_messages_for_gemma(
            self,
            prompts: List[Union[str, List[Dict[str, str]]]],
    ) -> List[List[Dict[str, str]]]:
        conversations = self._normalize_for_chat_template(
            prompts,
            delimiter=self.config.chat_template_delimiter,
        )

        normalized = []

        for messages in conversations:
            clean_messages = []

            for message in messages:
                role = message.get("role", "user")
                content = self._strip_internal_prompt_markers(
                    str(message.get("content", ""))
                )

                if not content:
                    continue

                # Gemma uses user/model roles. Fold system into user.
                if role == "assistant":
                    role = "model"
                elif role not in {"user", "model"}:
                    role = "user"

                clean_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

            normalized.append(clean_messages or [{"role": "user", "content": ""}])

        return normalized

    def _manual_gemma_chat_template(
            self,
            messages: List[Dict[str, str]],
    ) -> str:
        chunks = []

        for message in messages:
            role = message["role"]
            content = message["content"].strip()

            if not content:
                continue

            role = "model" if role == "model" else "user"
            chunks.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")

        chunks.append("<start_of_turn>model\n")
        return "".join(chunks)

    def generate(
            self,
            prompts: List[Union[str, List[Dict[str, str]]]],
            generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if self._backend is not None:
            return self._backend.generate(prompts, generation_kwargs=generation_kwargs)

        if generation_kwargs is None:
            generation_kwargs = self.config.to_generation_kwargs()
        else:
            generation_kwargs = dict(generation_kwargs)

        if self._is_qwen_model():
            # Preserve previous Qwen behavior exactly.
            if self.config.enable_thinking is False:
                conversations = self._normalize_for_chat_template(
                    prompts,
                    delimiter="\n<|USER_PROMPT|>\n",
                )
                prompts = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    enable_thinking=False,
                    add_generation_prompt=True,
                )

        elif self._is_gemma_model():
            # Gemma must not receive raw project-specific prompt markers.
            # _render_gemma_chat_prompts() should use either tokenizer.chat_template
            # or the manual Gemma <start_of_turn>user/model format.
            prompts = self._render_gemma_chat_prompts(prompts)

        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        if self._is_gemma_model():
            generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

            eos_ids = []

            if self.tokenizer.eos_token_id is not None:
                eos_ids.append(self.tokenizer.eos_token_id)

            end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0:
                eos_ids.append(end_of_turn_id)

            if eos_ids:
                generation_kwargs.setdefault("eos_token_id", eos_ids)

            bad_words_ids = self._gemma_bad_words_ids()
            if bad_words_ids:
                generation_kwargs.setdefault("bad_words_ids", bad_words_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                **generation_kwargs,
            )

        # Decode only newly generated tokens
        generated = outputs[:, enc["input_ids"].shape[1]:]

        texts = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if self._is_gemma_model():
            return [self._clean_gemma_generation(text) for text in texts]

        return [text.strip() for text in texts]

    def save(self, output_dir: str) -> None:
        if self._backend is not None:
            raise NotImplementedError("Remote backend has no local weights to save.")

        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        logger.info("Saved model + tokenizer to %s", output_dir)
