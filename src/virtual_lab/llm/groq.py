import os
from groq import Groq
from virtual_lab.llm.chat_completion_client import ChatCompletionClient


class GroqClient(ChatCompletionClient):
    """LLM client for Groq models."""

    def __init__(self, api_key: str | None = None, **kwargs: any) -> None:
        """Initializes the Groq client."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        client = Groq(api_key=self.api_key, **kwargs)
        super().__init__(chat_completion_function=client.chat.completions.create)
