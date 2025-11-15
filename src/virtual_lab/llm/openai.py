import os
from typing import Any, Dict, List

from openai import OpenAI

from virtual_lab.agent import Agent
from virtual_lab.constants import PUBMED_TOOL_DESCRIPTION
from virtual_lab.llm.base import LLMClient
from virtual_lab.utils import run_tools as run_tools_util


class OpenAIClient(LLMClient):
    """LLM client for OpenAI models."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Initializes the OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key, **kwargs)

    def create_assistant(
        self, agent: Agent, pubmed_search: bool = False
    ) -> Any:
        """Creates an assistant."""
        assistant_params = {"tools": [PUBMED_TOOL_DESCRIPTION]} if pubmed_search else {}
        return self.client.beta.assistants.create(
            name=agent.title,
            instructions=agent.prompt,
            model=agent.model,
            **assistant_params,
        )

    def create_thread(self) -> Any:
        """Creates a thread."""
        return self.client.beta.threads.create()

    def create_message(self, thread_id: str, content: str, role: str = "user") -> Any:
        """Creates a message."""
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content,
        )

    def run_thread_and_poll(
        self, thread_id: str, assistant_id: str, model: str, temperature: float
    ) -> Any:
        """Runs a thread and polls for completion."""
        return self.client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            model=model,
            temperature=temperature,
        )

    def get_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Gets all messages from a thread."""
        messages = []
        last_message = None
        params = {
            "thread_id": thread_id,
            "limit": 100,
            "order": "asc",
        }

        while True:
            if last_message is not None:
                params["after"] = last_message.id
            elif "after" in params:
                del params["after"]

            new_messages = self.client.beta.threads.messages.list(**params)

            messages.extend(new_messages.data)

            if not new_messages.has_next_page():
                break

            last_message = messages[-1]

        return [message.model_dump() for message in messages]

    def run_tools(self, run: Any) -> list[dict[str, str]]:
        """Runs the tools required by the run."""
        return run_tools_util(run=run)

    def submit_tool_outputs_and_poll(self, run: Any, tool_outputs: list) -> Any:
        """Submits the tool outputs to the run and polls for completion."""
        return self.client.beta.threads.runs.submit_tool_outputs_and_poll(
            thread_id=run.thread_id, run_id=run.id, tool_outputs=tool_outputs
        )
