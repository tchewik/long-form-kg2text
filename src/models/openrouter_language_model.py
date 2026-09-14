import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterLanguageModelConfig:
    """
    OpenRouter Chat Completions (OpenAI-compatible) config.

    Endpoint: POST https://openrouter.ai/api/v1/chat/completions
    Docs: https://openrouter.ai/docs/quickstart
    """
    model: str
    api_key: Optional[str] = None  # if None, uses env OPENROUTER_API_KEY
    base_url: str = "https://openrouter.ai/api/v1"
    endpoint_path: str = "/chat/completions"

    # Optional app attribution headers (recommended for leaderboard/analytics attribution)
    http_referer: Optional[str] = None
    x_title: Optional[str] = None

    # Generation params (OpenAI-style)
    max_new_tokens: int = 128  # maps to max_tokens
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None  # not universally supported; passed via "repetition_penalty" if set
    seed: Optional[int] = None

    # Request behavior
    timeout_s: float = 60.0
    max_retries: int = 6
    retry_backoff_base_s: float = 0.5
    retry_backoff_max_s: float = 10.0

    # Message framing
    system_prompt: Optional[str] = None  # if provided, prepended as system message
    user_role: str = "user"

    def api_url(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint_path

    def to_request_body(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # OpenRouter supports both max_tokens and max_completion_tokens; max_tokens is the common OpenAI-style key.
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
        if self.repetition_penalty is not None:
            # Not part of vanilla OpenAI ChatCompletions, but some providers accept it.
            body["repetition_penalty"] = float(self.repetition_penalty)
        if self.seed is not None:
            body["seed"] = int(self.seed)
        return body


class OpenRouterLanguageModel:
    """
    Docs:
      - Quickstart + raw API example and endpoint: https://openrouter.ai/docs/quickstart
      - Auth (Bearer tokens): https://openrouter.ai/docs/api/reference/authentication
      - ChatCompletions parameters: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
    """

    def __init__(self, config: OpenRouterLanguageModelConfig):
        self.config = config
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key missing. Provide config.api_key or set env OPENROUTER_API_KEY."
            )

        self.name = f"openrouter:{self.config.model}"
        logger.info("Initializing OpenRouterLanguageModel with config: %s", json.dumps(asdict(config), indent=2))

        self._session = requests.Session()

    @property
    def device(self):
        # For parity with HF LanguageModel
        return "openrouter"

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers
        if self.config.http_referer:
            h["HTTP-Referer"] = self.config.http_referer
        if self.config.x_title:
            h["X-Title"] = self.config.x_title
        return h

    def _messages_for_prompt(self, prompt: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.config.system_prompt:
            msgs.append({"role": "system", "content": self.config.system_prompt})
        msgs.append({"role": self.config.user_role, "content": prompt})
        return msgs

    def _post_with_retries(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = self.config.api_url()
        headers = self._headers()

        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._session.post(
                    url,
                    headers=headers,
                    data=json.dumps(body),
                    timeout=self.config.timeout_s,
                )

                # Retry on rate limits + transient server issues
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(
                        self.config.retry_backoff_max_s,
                        self.config.retry_backoff_base_s * (2 ** attempt) + (0.05 * attempt),
                    )
                    logger.warning(
                        "OpenRouter transient error status=%s attempt=%s/%s; backing off %.2fs; body=%s",
                        resp.status_code, attempt + 1, self.config.max_retries + 1, wait, resp.text[:300],
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_err = e
                wait = min(
                    self.config.retry_backoff_max_s,
                    self.config.retry_backoff_base_s * (2 ** attempt) + (0.05 * attempt),
                )
                logger.warning(
                    "OpenRouter request failed attempt=%s/%s; backing off %.2fs; err=%r",
                    attempt + 1, self.config.max_retries + 1, wait, e,
                )
                time.sleep(wait)

        raise RuntimeError(f"OpenRouter request failed after retries. Last error: {last_err!r}")

    def generate(
        self,
        prompts: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Notes:
          - This implementation sends one request per prompt (simple + robust).
          - If you need higher throughput, you can add client-side concurrency (threads/async)
            while respecting rate limits.
        """
        if generation_kwargs:
            # Map common keys if present:
            if "max_new_tokens" in generation_kwargs:
                self.config.max_new_tokens = int(generation_kwargs["max_new_tokens"])
            if "temperature" in generation_kwargs:
                self.config.temperature = float(generation_kwargs["temperature"])
            if "top_p" in generation_kwargs:
                self.config.top_p = float(generation_kwargs["top_p"])
            if "stop_strings" in generation_kwargs and generation_kwargs["stop_strings"]:
                self.config.stop = list(generation_kwargs["stop_strings"])
            # OpenRouter chat-completions doesn't have a standard top_k; ignore safely.
            if "repetition_penalty" in generation_kwargs:
                self.config.repetition_penalty = float(generation_kwargs["repetition_penalty"])

        outputs: List[str] = []
        for p in prompts:
            messages = self._messages_for_prompt(p)
            body = self.config.to_request_body(messages)

            data = self._post_with_retries(body)

            try:
                choice0 = (data.get("choices") or [])[0]
                print(f'{choice0 = }')
                msg = choice0.get("message") or {}
                content = msg.get("content")
                if content is None:
                    # Some providers may use "text" in older schemas; fallback.
                    content = choice0.get("text", "")
            except Exception:
                content = ""

            outputs.append((content or "").strip())

        return outputs
