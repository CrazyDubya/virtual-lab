import os
import json
from anthropic import Anthropic

from virtual_lab.llm.chat_completion_client import ChatCompletionClient


class AnthropicClient(ChatCompletionClient):
    """LLM client for Anthropic models."""

    def __init__(self, api_key: str | None = None, **kwargs: any) -> None:
        """Initializes the Anthropic client."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=self.api_key, **kwargs)

        def chat_completion_function(**kwargs):
            messages = kwargs.get("messages", [])
            system_prompt = ""
            if messages and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
                messages = messages[1:]

            response = client.messages.create(
                model=kwargs.get("model"),
                system=system_prompt,
                messages=messages,
                tools=kwargs.get("tools"),
                temperature=kwargs.get("temperature"),
                max_tokens=4096,
            )

            tool_calls = []
            if response.stop_reason == "tool_use":
                for tool_use in response.content:
                    if hasattr(tool_use, 'type') and tool_use.type == "tool_use":
                        tool_calls.append({
                            "id": tool_use.id,
                            "type": "function",
                            "function": {
                                "name": tool_use.name,
                                "arguments": json.dumps(tool_use.input),
                            }
                        })

            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text if response.content and hasattr(response.content[0], 'text') else None,
                        "tool_calls": tool_calls if tool_calls else None,
                    }
                }]
            }

        super().__init__(chat_completion_function=chat_completion_function)
