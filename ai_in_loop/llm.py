"""LLM setup for an individual model instance.

This module provides:
- MockChatModel: Deterministic mock for testing without API calls
- get_llm: Factory function to get the appropriate LLM based on config
- get_text: Helper to extract text from LLM responses
"""

import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import Config
from dotenv import load_dotenv
load_dotenv()  # reads .env before Config tries to read env vars

class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for testing without API calls.

    Returns predictable responses based on the input prompt, useful for
    stable tests and offline development.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def bind_tools(self, tools: list, **kwargs: Any) -> "MockChatModel":
        """Return self (mock doesn't use tools for paper analyzer)."""
        return self

    def _is_summarize_request(self, text: str) -> bool:
        """Check if the text is a summarization request."""
        lower = text.lower()
        return "summarize" in lower or "summary" in lower

    def _is_metadata_extraction_request(self, text: str) -> bool:
        """Check if the text is a metadata extraction request."""
        lower = text.lower()
        return "extract metadata" in lower or "json" in lower and "title" in lower

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response based on the input messages."""
        # Find the last human message content
        human_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                human_content = str(msg.content)
                break
        # Default mock response
        line_count = human_content.count("\n") + 1
        char_count = len(human_content)
        mock_text = f"[MOCK] Received {line_count} line(s), {char_count} char(s)"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=mock_text))]
        )

# POSSIBLE TODO: add other models (llama, claude, chat, etc.) that can be called if api key is provided or something.
# would probably need another argument indicating model choice.
def get_llm(cfg: Config) -> BaseChatModel:
    """Get a LangChain chat model based on configuration.

    Returns a MockChatModel if USE_GEMINI is disabled or no API key is provided.
    Otherwise returns a ChatGoogleGenerativeAI instance.

    Args:
        cfg: Configuration object

    Returns:
        Chat model instance
    """
    if not cfg.use_gemini:
        return MockChatModel()

    if not cfg.gemini_api_key:
        # Fall back to mock rather than crash
        return MockChatModel()

    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {
        "model": cfg.gemini_model,
        "temperature": cfg.temperature,
        "google_api_key": cfg.gemini_api_key,
    }

    # Pass appropriate thinking parameter based on model
    if cfg.gemini_model.startswith("gemini-3"):
        # Gemini 3+: use thinking_level if set
        if cfg.thinking_level:
            kwargs["thinking_level"] = cfg.thinking_level
    else:
        # Gemini 2.5 and earlier: use thinking_budget
        if cfg.thinking_budget is not None:
            kwargs["thinking_budget"] = cfg.thinking_budget

    return ChatGoogleGenerativeAI(**kwargs)

