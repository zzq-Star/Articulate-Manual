import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import anthropic
from anthropic import AsyncAnthropic

from articulate_core.config.settings import ArticulateConfig
from articulate_core.exceptions import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMResponse:
    """Wrapped LLM completion response."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        cache_metrics: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.cache_metrics = cache_metrics or {}


# ── Abstract Base ─────────────────────────────────────────────────────


class BaseLLMClient(ABC):
    """Abstract interface for all LLM providers."""

    model: str

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        ...

    async def complete_structured(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        output_model: Type[T],
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> T:
        """Request structured JSON output, parse into Pydantic model.

        Retries on parse failure (up to 2x).
        """
        parse_retries = 2
        last_parse_error: Optional[Exception] = None

        for attempt in range(parse_retries + 1):
            enhanced_messages = list(messages)
            format_instruction = (
                "\n\nYou MUST respond with valid JSON only, no other text. "
                f"The JSON must match this schema: {output_model.model_json_schema()}"
            )

            if enhanced_messages and enhanced_messages[-1]["role"] == "user":
                enhanced_messages[-1] = {
                    "role": "user",
                    "content": enhanced_messages[-1]["content"] + format_instruction,
                }
            else:
                enhanced_messages.append({"role": "user", "content": format_instruction})

            response = await self.complete(
                system=system,
                messages=enhanced_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            try:
                data = json.loads(response.content)
                return output_model.model_validate(data)
            except (json.JSONDecodeError, ValueError) as e:
                last_parse_error = e
                logger.warning(
                    "Parse failure (attempt %d/%d): %s",
                    attempt + 1, parse_retries, e,
                )
                if attempt < parse_retries:
                    enhanced_messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    enhanced_messages.append({
                        "role": "user",
                        "content": (
                            "The previous response was not valid JSON. "
                            "Please ensure your response is valid JSON matching the specified schema."
                        ),
                    })
                continue

        raise LLMError(f"Failed to parse structured output after {parse_retries + 1} attempts") from last_parse_error

    async def count_tokens(self, text: str) -> int:
        """Count tokens. Not supported by all providers."""
        raise NotImplementedError(f"count_tokens not supported by {type(self).__name__}")

    async def close(self):
        """Close underlying HTTP client."""
        ...


# ── Anthropic / Claude ────────────────────────────────────────────────


class ClaudeClient(BaseLLMClient):
    """Thread-safe wrapper around anthropic.AsyncAnthropic.

    Handles retries, structured output parsing, and token tracking.
    """

    def __init__(self, config: ArticulateConfig):
        self._client = AsyncAnthropic(api_key=config.api_key)
        self.model = config.llm_model
        self.max_retries = config.llm_max_retries
        self.retry_base_delay = config.llm_retry_base_delay

    async def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.messages.create(
                    model=self.model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )

                content = ""
                if response.content:
                    for block in response.content:
                        if hasattr(block, "text"):
                            content = block.text
                            break
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                } if response.usage else {}

                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                )

            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(f"Rate limit exceeded after {self.max_retries} retries") from e

            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < self.max_retries:
                    last_error = e
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Server error %d (attempt %d/%d), retrying in %.1fs",
                        e.status_code, attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(f"API error: {e}") from e

            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Connection error (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(f"Connection error after {self.max_retries} retries") from e

        raise LLMError(f"Request failed after {self.max_retries} retries") from last_error

    async def count_tokens(self, text: str) -> int:
        response = await self._client.count_tokens(text)
        return response.input_tokens

    async def close(self):
        await self._client.close()


# ── DeepSeek (OpenAI-compatible) ──────────────────────────────────────


class DeepSeekClient(BaseLLMClient):
    """OpenAI-compatible client for DeepSeek (and other compatible providers).

    Uses the OpenAI Python SDK with a custom base_url.
    """

    def __init__(self, config: ArticulateConfig):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "DeepSeek support requires the 'openai' package. "
                "Install it with: pip install openai"
            )
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = config.llm_model
        self.max_retries = config.llm_max_retries
        self.retry_base_delay = config.llm_retry_base_delay

    async def complete(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        last_error: Optional[Exception] = None

        # Build messages list with system prompt as first message
        openai_messages: List[Dict[str, str]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                content = response.choices[0].message.content or ""
                usage = {}
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    }

                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                )

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_retryable = any(
                    keyword in err_str
                    for keyword in ["rate", "timeout", "server error", "503", "502", "429"]
                )
                if is_retryable and attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LLMError(f"DeepSeek API error: {e}") from e

        raise LLMError(f"Request failed after {self.max_retries} retries") from last_error

    async def close(self):
        await self._client.close()


# ── Model → Provider inference ────────────────────────────────────────

MODEL_PROVIDER_MAP = [
    ("claude-", "anthropic"),
    ("deepseek-", "deepseek"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
]


def infer_provider_from_model(model: str) -> Optional[str]:
    """Infer LLM provider from model name prefix.

    Returns provider name or None if unknown.
    """
    name = model.lower()
    for prefix, provider in MODEL_PROVIDER_MAP:
        if name.startswith(prefix):
            return provider
    return None


# ── Factory ───────────────────────────────────────────────────────────


def create_llm_client(config: ArticulateConfig) -> BaseLLMClient:
    """Factory: create the appropriate LLM client based on config."""
    provider = config.llm_provider.lower()

    if provider == "anthropic":
        return ClaudeClient(config)
    elif provider == "deepseek":
        return DeepSeekClient(config)
    elif provider == "openai":
        return DeepSeekClient(config)
    else:
        raise LLMError(f"Unsupported LLM provider: {provider}")
