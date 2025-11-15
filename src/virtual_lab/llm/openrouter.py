import os

from virtual_lab.llm.openai import OpenAIClient


class OpenRouterClient(OpenAIClient):
    """LLM client for OpenRouter models."""

    def __init__(self, api_key: str | None = None, **kwargs: any) -> None:
        """Initializes the OpenRouter client."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key=self.api_key, base_url="https://openrouter.ai/api/v1", **kwargs)
