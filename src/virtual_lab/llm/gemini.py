import os
import google.generativeai as genai
from typing import Any, Dict, List

from virtual_lab.agent import Agent
from virtual_lab.llm.base import LLMClient
from virtual_lab.llm.chat_completion_client import ChatCompletionClient

class GeminiClient(ChatCompletionClient):
    """LLM client for Google Gemini models."""

    def __init__(self, api_key: str | None = None, **kwargs: any) -> None:
        """Initializes the Gemini client."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)

        # The chat completion function for Gemini is different.
        # We need a wrapper.
        model = genai.GenerativeModel('gemini-1.5-flash')

        def chat_completion_function(**kwargs):
            # This is a simplified wrapper. A real implementation would be more robust.
            contents = []
            for msg in kwargs["messages"]:
                contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

            # The tools format is also different.
            # For now, we ignore tools.

            response = model.generate_content(contents)

            # Adapt response to look like OpenAI's
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.text,
                        "tool_calls": None,
                    }
                }]
            }

        super().__init__(chat_completion_function=chat_completion_function)
