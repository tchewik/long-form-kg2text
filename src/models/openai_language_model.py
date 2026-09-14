import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)



class OpenAINonRetryableError(RuntimeError):
    pass


class OpenAIContentPolicyViolationError(OpenAINonRetryableError):
    pass


@dataclass
class OpenAILanguageModelConfig:
    """
    OpenAI-compatible Chat Completions config.

    Defaults:
      - base_url from OPENAI_BASE_URL or https://api.openai.com/v1
      - api_key from OPENAI_API_KEY

    Endpoint:
      POST {base_url}/chat/completions
    """
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoint_path: str = "/chat/completions"

    # Generation params
    max_new_tokens: int = 128  # maps to max_tokens
    temperature: float = 0.0
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None

    # Request behavior
    timeout_s: float = 60.0
    max_retries: int = 6
    retry_backoff_base_s: float = 0.5
    retry_backoff_max_s: float = 10.0

    max_prompt_chars: Optional[int] = None
    prompt_truncation_marker: str = "\n\n[... prompt truncated because it exceeded max_chars ...]\n\n"

    # Message framing
    system_prompt: Optional[str] = None
    user_role: str = "user"

    # Usage logging
    log_token_usage: bool = True

    def api_url(self) -> str:
        resolved_base_url = (
            self.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return resolved_base_url.rstrip("/") + self.endpoint_path

    def to_request_body(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(self.max_new_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "stream": False,
        }
        if self.stop:
            body["stop"] = list(self.stop)
        if self.presence_penalty is not None:
            body["presence_penalty"] = float(self.presence_penalty)
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = float(self.frequency_penalty)
        if self.seed is not None:
            body["seed"] = int(self.seed)
        return body


class OpenAILanguageModel:
    """
    OpenAI-compatible chat completions client using requests.

    Environment variables:
      - OPENAI_API_KEY
      - OPENAI_BASE_URL

    Token usage:
      - Per-request usage is logged when the API returns a `usage` object.
      - Cumulative usage is kept in `self.token_usage`.
      - Call `get_token_usage()` to retrieve current totals.
      - Call `reset_token_usage()` to clear current totals.
    """

    _USAGE_KEYS = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
    )

    def __init__(self, config: OpenAILanguageModelConfig):
        self.config = config

        if not self.config.api_key:
            self.config.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.config.base_url:
            self.config.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key missing. Provide config.api_key or set env OPENAI_API_KEY."
            )

        self.name = f"openai:{self.config.model}"
        logger.info(
            "Initializing OpenAILanguageModel with config: %s",
            json.dumps(self._safe_config_for_logging(self.config), indent=2),
        )

        self._session = requests.Session()

        # One usage record per prompt from the latest generate() call.
        # This is what BasePipeline will read to write usage into cache/prediction JSONL.
        self.last_generation_usages: List[Dict[str, int]] = []

        # Cumulative usage for this OpenAILanguageModel instance.
        self.token_usage: Dict[str, int] = self._empty_usage_totals()

    @staticmethod
    def _mask_secret(value: Optional[str], *, visible_prefix: int = 3, visible_suffix: int = 3) -> Optional[str]:
        if value is None:
            return None

        value = str(value)
        if not value:
            return value

        if len(value) <= visible_prefix + visible_suffix:
            return "*" * len(value)

        return f"{value[:visible_prefix]}...{value[-visible_suffix:]}"

    def _truncate_prompt_if_needed(self, prompt: str) -> str:
        max_chars = self.config.max_prompt_chars

        if max_chars is None or max_chars <= 0:
            return prompt

        if len(prompt) <= max_chars:
            return prompt

        marker = self.config.prompt_truncation_marker

        # If the marker itself is too large, fall back to hard truncation.
        if len(marker) >= max_chars:
            truncated = prompt[:max_chars]
            logger.warning(
                "Prompt truncated from %s chars to %s chars using hard truncation.",
                len(prompt),
                len(truncated),
            )
            return truncated

        available = max_chars - len(marker)

        # Keep mostly the beginning, but preserve some tail because prompts often put
        # task instructions or output constraints near the end.
        head_chars = int(available * 0.85)
        tail_chars = available - head_chars

        truncated = prompt[:head_chars] + marker + prompt[-tail_chars:]

        logger.warning(
            "Prompt truncated from %s chars to %s chars. max_prompt_chars=%s",
            len(prompt),
            len(truncated),
            max_chars,
        )

        return truncated

    @classmethod
    def _safe_config_for_logging(cls, config: OpenAILanguageModelConfig) -> Dict[str, Any]:
        payload = asdict(config)
        payload["api_key"] = cls._mask_secret(payload.get("api_key"))
        return payload

    @classmethod
    def _empty_usage_totals(cls) -> Dict[str, int]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except Exception:
            return 0

    @classmethod
    def _normalize_usage(cls, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """
        Normalize OpenAI/OpenAI-compatible usage payloads.

        Chat Completions commonly returns:
          {
            "prompt_tokens": ...,
            "completion_tokens": ...,
            "total_tokens": ...,
            "prompt_tokens_details": {"cached_tokens": ...},
            "completion_tokens_details": {"reasoning_tokens": ...}
          }

        Some OpenAI-compatible providers omit details and only return the
        top-level token fields.
        """
        usage = usage or {}

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        return {
            "prompt_tokens": cls._as_int(usage.get("prompt_tokens")),
            "completion_tokens": cls._as_int(usage.get("completion_tokens")),
            "total_tokens": cls._as_int(usage.get("total_tokens")),
            "cached_tokens": cls._as_int(prompt_details.get("cached_tokens")),
            "reasoning_tokens": cls._as_int(completion_details.get("reasoning_tokens")),
        }

    @classmethod
    def sum_usages(cls, usages: List[Dict[str, Any]]) -> Dict[str, int]:
        total = cls._empty_usage_totals()

        for usage in usages or []:
            normalized = cls._normalize_usage(usage)
            total["requests"] += int(usage.get("requests", 1) or 1)
            total["prompt_tokens"] += normalized["prompt_tokens"]
            total["completion_tokens"] += normalized["completion_tokens"]
            total["total_tokens"] += normalized["total_tokens"]
            total["cached_tokens"] += normalized["cached_tokens"]
            total["reasoning_tokens"] += normalized["reasoning_tokens"]

        return total

    def _record_token_usage(self, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        normalized = self._normalize_usage(usage)

        self.token_usage["requests"] += 1
        self.token_usage["prompt_tokens"] += normalized["prompt_tokens"]
        self.token_usage["completion_tokens"] += normalized["completion_tokens"]
        self.token_usage["total_tokens"] += normalized["total_tokens"]
        self.token_usage["cached_tokens"] += normalized["cached_tokens"]
        self.token_usage["reasoning_tokens"] += normalized["reasoning_tokens"]

        if self.config.log_token_usage:
            logger.info(
                (
                    "OpenAI token usage request: model=%s "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
                    "cached_tokens=%s reasoning_tokens=%s | "
                    "cumulative_requests=%s cumulative_prompt_tokens=%s "
                    "cumulative_completion_tokens=%s cumulative_total_tokens=%s "
                    "cumulative_cached_tokens=%s cumulative_reasoning_tokens=%s"
                ),
                self.config.model,
                normalized["prompt_tokens"],
                normalized["completion_tokens"],
                normalized["total_tokens"],
                normalized["cached_tokens"],
                normalized["reasoning_tokens"],
                self.token_usage["requests"],
                self.token_usage["prompt_tokens"],
                self.token_usage["completion_tokens"],
                self.token_usage["total_tokens"],
                self.token_usage["cached_tokens"],
                self.token_usage["reasoning_tokens"],
            )

        return normalized

    def get_token_usage(self) -> Dict[str, int]:
        return dict(self.token_usage)

    def reset_token_usage(self) -> None:
        self.token_usage = self._empty_usage_totals()
        self.last_generation_usages = []

    @property
    def device(self):
        # For parity with HF LanguageModel
        return "openai"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _messages_for_prompt(self, prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.config.system_prompt:
            msgs.append({"role": "system", "content": self.config.system_prompt})
        msgs.append({"role": self.config.user_role, "content": prompt})
        return msgs

    def _post_with_retries(
            self,
            body: Dict[str, Any],
            timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = self.config.api_url()
        headers = self._headers()

        base_timeout = float(timeout_s or self.config.timeout_s)
        effective_timeout = base_timeout
        max_read_timeout_retries = 1
        read_timeout_count = 0

        last_err: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._session.post(
                    url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=effective_timeout,
                )

                if resp.status_code in (400, 401, 403, 404):
                    logger.error(
                        "OpenAI non-retryable error status=%s body=%s",
                        resp.status_code,
                        resp.text[:2000],
                    )

                    if "ContentPolicyViolationError" in resp.text or "content management policy" in resp.text:
                        raise OpenAIContentPolicyViolationError(
                            f"OpenAI content policy violation status={resp.status_code}: {resp.text[:2000]}"
                        )

                    raise OpenAINonRetryableError(
                        f"OpenAI non-retryable status={resp.status_code}: {resp.text[:2000]}"
                    )

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(
                        self.config.retry_backoff_max_s,
                        self.config.retry_backoff_base_s * (2 ** attempt) + (0.05 * attempt),
                    )
                    logger.warning(
                        "OpenAI transient error status=%s attempt=%s/%s; backing off %.2fs; body=%s",
                        resp.status_code,
                        attempt + 1,
                        self.config.max_retries + 1,
                        wait,
                        resp.text[:300],
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.ReadTimeout as e:
                last_err = e
                read_timeout_count += 1

                logger.warning(
                    "OpenAI read timeout %s/%s on attempt=%s/%s timeout=%.1fs; err=%r",
                    read_timeout_count,
                    max_read_timeout_retries + 1,
                    attempt + 1,
                    self.config.max_retries + 1,
                    effective_timeout,
                    e,
                )

                if read_timeout_count > max_read_timeout_retries:
                    raise RuntimeError(
                        f"OpenAI request read-timed-out after {read_timeout_count} read-timeout attempts. "
                        f"base_timeout={base_timeout:.1f}s final_timeout={effective_timeout:.1f}s "
                        f"last_error={e!r}"
                    )

                effective_timeout = min(effective_timeout * 2.0, 900.0)
                logger.warning(
                    "Retrying OpenAI request with increased timeout=%.1fs",
                    effective_timeout,
                )
                time.sleep(1.0)
                continue

            except OpenAINonRetryableError:
                raise

            except Exception as e:
                last_err = e
                wait = min(
                    self.config.retry_backoff_max_s,
                    self.config.retry_backoff_base_s * (2 ** attempt) + (0.05 * attempt),
                )
                logger.warning(
                    "OpenAI request failed attempt=%s/%s; backing off %.2fs; err=%r",
                    attempt + 1,
                    self.config.max_retries + 1,
                    wait,
                    e,
                )
                time.sleep(wait)

        raise RuntimeError(f"OpenAI request failed after retries. Last error: {last_err!r}")

    @staticmethod
    def _extract_text_from_message_content(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item["text"]))
            return "".join(parts)

        return str(content)

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except Exception:
            return 0

    def generate(
            self,
            prompts: List[str],
            generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Notes:
          - Sends one request per prompt.
          - generation_kwargs can override config values.
          - last_generation_usages[i] corresponds to outputs[i].
        """
        if generation_kwargs:
            if "max_new_tokens" in generation_kwargs:
                self.config.max_new_tokens = int(generation_kwargs["max_new_tokens"])
            if "temperature" in generation_kwargs:
                self.config.temperature = float(generation_kwargs["temperature"])
            if "top_p" in generation_kwargs:
                self.config.top_p = float(generation_kwargs["top_p"])
            if "stop_strings" in generation_kwargs and generation_kwargs["stop_strings"]:
                self.config.stop = list(generation_kwargs["stop_strings"])

        self.last_generation_usages = []
        batch_usage_before = self.get_token_usage()

        outputs: List[str] = []

        for p in prompts:
            try:
                messages = self._messages_for_prompt(p)
                body = self.config.to_request_body(messages)

                data = self._post_with_retries(body)

                normalized_usage = self._record_token_usage(data.get("usage"))
                self.last_generation_usages.append(normalized_usage)

                choice0 = (data.get("choices") or [])[0]
                msg = choice0.get("message") or {}
                content = self._extract_text_from_message_content(msg.get("content"))
                if not content:
                    content = choice0.get("text", "")

                outputs.append((content or "").strip())

            except Exception as e:
                logger.exception("OpenAI generation failed for one prompt: %r", e)
                self.last_generation_usages.append(self._empty_usage_totals())
                outputs.append("")

        if self.config.log_token_usage and len(prompts) > 1:
            batch_usage_after = self.get_token_usage()
            logger.info(
                (
                    "OpenAI token usage batch: model=%s requests=%s "
                    "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
                    "cached_tokens=%s reasoning_tokens=%s"
                ),
                self.config.model,
                batch_usage_after["requests"] - batch_usage_before["requests"],
                batch_usage_after["prompt_tokens"] - batch_usage_before["prompt_tokens"],
                batch_usage_after["completion_tokens"] - batch_usage_before["completion_tokens"],
                batch_usage_after["total_tokens"] - batch_usage_before["total_tokens"],
                batch_usage_after["cached_tokens"] - batch_usage_before["cached_tokens"],
                batch_usage_after["reasoning_tokens"] - batch_usage_before["reasoning_tokens"],
            )

        return outputs
