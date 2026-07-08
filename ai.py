"""AI manager and provider implementations.

Provides an async-friendly AIAgent that routes requests to available
LLM providers (Gemini now, OpenAI in future). Keeps short-term
conversation history and exposes a synchronous wrapper for existing code
that expects blocking behavior.

Design goals:
- Provider interface for extensibility
- Async calls with retry and timeout
- Safe handling when providers are not installed or API keys missing
- Logging, type hints, and docstrings
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Try optional imports; we handle absence gracefully
try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - depends on user environment
    genai = None

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - depends on user environment
    openai = None


Message = Dict[str, str]


class Provider(Protocol):
    """Protocol describing an LLM provider implementation."""

    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        """Generate a text response for the provided chat messages."""
        ...


@dataclass
class GeminiProvider:
    """Provider implementation for Google's GenAI (gemini).

    This implementation calls blocking gemini client methods inside
    asyncio.to_thread so the interface remains async-friendly.
    """

    api_key: Optional[str] = settings.GEMINI_API_KEY
    model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        if genai and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)  # type: ignore
                logger.debug("Initialized Gemini client")
            except Exception as exc:
                self._client = None
                logger.exception("Failed to initialize Gemini client: %s", exc)
        else:
            self._client = None
            logger.debug("Gemini SDK not available or API key missing")

    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        if not self._client:
            raise RuntimeError("Gemini client is not configured")

        # Gemini client expects a string prompt; we join the chat history
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        def _sync_call() -> str:
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                # response.text may not be guaranteed; coerce safely
                text = getattr(response, "text", None) or str(response)
                return text.strip()
            except Exception as exc:  # pragma: no cover - depends on SDK
                logger.exception("Gemini generation failed: %s", exc)
                raise

        return await asyncio.to_thread(_sync_call)


@dataclass
class OpenAIProvider:
    """Placeholder OpenAI provider; will be used if openai package is present.

    Currently uses the legacy ChatCompletion API for compatibility, but can
    be extended to the newer OpenAI Python SDK streaming interfaces.
    """

    api_key: Optional[str] = settings.OPENAI_API_KEY
    model: str = "gpt-3.5-turbo"

    def __post_init__(self) -> None:
        if openai and self.api_key:
            try:
                openai.api_key = self.api_key  # type: ignore
                logger.debug("Configured OpenAI client")
            except Exception as exc:
                logger.exception("Failed to configure OpenAI client: %s", exc)
        else:
            logger.debug("OpenAI SDK not available or API key missing")

    async def generate(self, messages: List[Message], **kwargs: Any) -> str:
        if not openai or not self.api_key:
            raise RuntimeError("OpenAI client is not configured")

        # Wrap the blocking call in a thread so we don't block the loop
        def _sync_call() -> str:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:  # pragma: no cover - depends on API
                logger.exception("OpenAI generation failed: %s", exc)
                raise

        return await asyncio.to_thread(_sync_call)


@dataclass
class AIAgent:
    """High-level AI agent that manages conversation state and providers.

    Usage: create an AIAgent() and call `await agent.ask(prompt)` in async
    contexts. For synchronous callers there's a `ask_sync` wrapper.
    """

    system_prompt: str = field(default_factory=lambda: (
        f"You are {settings.ASSISTANT_NAME}, a helpful digital assistant for {settings.USER_NAME}."
    ))
    providers: List[Provider] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)
    default_timeout: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        # Initialize history with system prompt
        self.history.insert(0, {"role": "system", "content": self.system_prompt})

        # Auto-register available providers
        if genai and settings.GEMINI_API_KEY:
            self.providers.append(GeminiProvider())
            logger.debug("Registered GeminiProvider")

        if openai and settings.OPENAI_API_KEY:
            self.providers.append(OpenAIProvider())
            logger.debug("Registered OpenAIProvider")

        if not self.providers:
            logger.warning("No AI providers configured. Set OPENAI_API_KEY or GEMINI_API_KEY.")

    async def ask(self, prompt: str) -> str:
        """Ask the AI agent a prompt and return the assistant reply.

        This method updates short-term history and attempts to generate
        a response using the first available provider with retry/timeout
        behavior.
        """
        if not prompt or not prompt.strip():
            logger.debug("Empty prompt received")
            return "Please say something so I can help."

        user_message = {"role": "user", "content": prompt.strip()}
        self.history.append(user_message)

        if not self.providers:
            return "AI is not configured. Set OPENAI_API_KEY or GEMINI_API_KEY in environment."

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            provider = self.providers[0]
            try:
                logger.debug("Attempt %d using provider %s", attempt, provider.__class__.__name__)
                coro = provider.generate(self.history)
                response_text = await asyncio.wait_for(coro, timeout=self.default_timeout)
                assistant_message = {"role": "assistant", "content": response_text}
                self.history.append(assistant_message)
                logger.info("AI response received (len=%d)", len(response_text))
                return response_text
            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError("Provider timed out")
                logger.warning("AI provider timed out on attempt %d", attempt)
            except Exception as exc:
                last_exc = exc
                logger.exception("AI provider error on attempt %d: %s", attempt, exc)

            # simple backoff
            await asyncio.sleep(0.5 * attempt)

        logger.error("All retries failed: %s", last_exc)
        return f"AI error: {last_exc}"

    def ask_sync(self, prompt: str) -> str:
        """Synchronous wrapper for ask(). Runs the async ask in an event loop.

        If called from within an existing running event loop this will create
        a new task and wait for it.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an existing loop (e.g., GUI). Run as a task and wait.
            fut = asyncio.run_coroutine_threadsafe(self.ask(prompt), loop)
            return fut.result()
        else:
            return asyncio.run(self.ask(prompt))
