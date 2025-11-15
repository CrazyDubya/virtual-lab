from abc import ABC, abstractmethod
from typing import Any, Dict, List

from virtual_lab.agent import Agent


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """Initializes the LLM client."""
        pass

    @abstractmethod
    def create_assistant(
        self, agent: Agent, pubmed_search: bool = False
    ) -> Any:
        """Creates an assistant.

        :param agent: The agent to create an assistant for.
        :param pubmed_search: Whether to include a PubMed search tool.
        :return: The assistant object.
        """
        pass

    @abstractmethod
    def create_thread(self) -> Any:
        """Creates a thread.

        :return: The thread object.
        """
        pass

    @abstractmethod
    def create_message(self, thread_id: str, content: str, role: str = "user") -> Any:
        """Creates a message.

        :param thread_id: The ID of the thread.
        :param content: The content of the message.
        :param role: The role of the message sender.
        :return: The message object.
        """
        pass

    @abstractmethod
    def run_thread_and_poll(
        self, thread_id: str, assistant_id: str, model: str, temperature: float
    ) -> Any:
        """Runs a thread and polls for completion.

        :param thread_id: The ID of the thread.
        :param assistant_id: The ID of the assistant.
        :param model: The model to use.
        :param temperature: The sampling temperature.
        :return: The run object.
        """
        pass

    @abstractmethod
    def get_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Gets all messages from a thread.

        :param thread_id: The ID of the thread.
        :return: A list of messages.
        """
        pass

    @abstractmethod
    def run_tools(self, run: Any) -> list[dict[str, str]]:
        """Runs the tools required by the run.

        :param run: The run object.
        :return: A list of tool outputs.
        """
        pass

    @abstractmethod
    def submit_tool_outputs_and_poll(self, run: Any, tool_outputs: list) -> Any:
        """Submits the tool outputs to the run and polls for completion.

        :param run: The run object.
        :param tool_outputs: The tool outputs.
        :return: The run object.
        """
        pass
