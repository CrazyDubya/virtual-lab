import ollama
from virtual_lab.llm.chat_completion_client import ChatCompletionClient


class OllamaClient(ChatCompletionClient):
    """LLM client for Ollama models."""

    def __init__(self, **kwargs: any) -> None:
        """Initializes the Ollama client."""
        client = ollama.Client(**kwargs)
        super().__init__(chat_completion_function=client.chat)
