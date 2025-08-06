import uuid
import json
from typing import Any, Dict, List

from virtual_lab.agent import Agent
from virtual_lab.llm.base import LLMClient
from virtual_lab.constants import PUBMED_TOOL_DESCRIPTION
from virtual_lab.utils import run_pubmed_search


from typing import Callable


class ChatCompletionClient(LLMClient):
    """A generic LLM client that uses a chat/completions endpoint and simulates the Assistants API."""

    def __init__(self, chat_completion_function: Callable) -> None:
        """Initializes the ChatCompletionClient."""
        self.chat_completion_function = chat_completion_function
        self.threads: Dict[str, List[Dict[str, Any]]] = {}
        self.assistants: Dict[str, Dict[str, Any]] = {}

    def create_assistant(
        self, agent: Agent, pubmed_search: bool = False
    ) -> Any:
        """Creates an assistant."""
        assistant_id = agent.title.lower().replace(" ", "-")
        assistant = {
            "id": assistant_id,
            "name": agent.title,
            "instructions": agent.prompt,
            "model": agent.model,
            "tools": [PUBMED_TOOL_DESCRIPTION] if pubmed_search else [],
        }
        self.assistants[assistant_id] = assistant
        return assistant

    def create_thread(self) -> Any:
        """Creates a thread."""
        thread_id = str(uuid.uuid4())
        self.threads[thread_id] = []
        return {"id": thread_id}

    def create_message(self, thread_id: str, content: str, role: str = "user") -> Any:
        """Creates a message."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID {thread_id} not found.")

        message = {"role": role, "content": content}
        self.threads[thread_id].append(message)
        return message

    def run_thread_and_poll(
        self, thread_id: str, assistant_id: str, model: str, temperature: float
    ) -> Any:
        """Runs a thread and polls for completion."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID {thread_id} not found.")
        if assistant_id not in self.assistants:
            raise ValueError(f"Assistant with ID {assistant_id} not found.")

        assistant = self.assistants[assistant_id]
        messages = self.threads[thread_id]

        system_prompt = {"role": "system", "content": assistant["instructions"]}

        response = self.client.chat.completions.create(
            model=model,
            messages=[system_prompt] + messages,
            tools=assistant["tools"],
            temperature=temperature,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        self.threads[thread_id].append(response_message)

        run_id = str(uuid.uuid4())

        if response_message.tool_calls:
            status = "requires_action"
        else:
            status = "completed"

        return {
            "id": run_id,
            "thread_id": thread_id,
            "status": status,
            "required_action": {
                "submit_tool_outputs": {
                    "tool_calls": response_message.tool_calls
                }
            } if status == "requires_action" else None,
            "model": model,
            "assistant_id": assistant_id,
        }

    def get_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Gets all messages from a thread."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID {thread_id} not found.")

        formatted_messages = []
        for msg in self.threads[thread_id]:
            if isinstance(msg, dict): # a message from user or tool
                content = msg["content"]
                role = msg.get("role")
                name = msg.get("name")
                assistant_id = None
                tool_call_id = msg.get("tool_call_id")
            else: # a response from the model
                content = msg.content
                role = msg.role
                name = None
                assistant_id = "chat_completion_assistant" # placeholder
                tool_call_id = None


            formatted_messages.append({
                "id": str(uuid.uuid4()),
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "run_id": None,
                "role": role,
                "content": [{"type": "text", "text": {"value": content, "annotations": []}}],
                "created_at": 0,
                "metadata": {},
                "attachments": [],
                "name": name,
                "tool_call_id": tool_call_id,
            })
        return formatted_messages

    def run_tools(self, run: Any) -> list[dict[str, str]]:
        """Runs the tools required by the run."""
        tool_outputs = []
        for tool_call in run["required_action"]["submit_tool_outputs"]["tool_calls"]:
            if tool_call.function.name == "pubMedSearch":
                args = json.loads(tool_call.function.arguments)
                output = run_pubmed_search(**args)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": output,
                })
            else:
                raise ValueError(f"Unknown tool: {tool_call.function.name}")
        return tool_outputs

    def submit_tool_outputs_and_poll(self, run: Any, tool_outputs: list) -> Any:
        """Submits the tool outputs to the run and polls for completion."""
        thread_id = run["thread_id"]
        if thread_id not in self.threads:
            raise ValueError(f"Thread with ID {thread_id} not found.")

        assistant_id = run["assistant_id"]
        assistant = self.assistants[assistant_id]

        messages = self.threads[thread_id]

        last_message = self.threads[thread_id][-1]
        tool_calls = last_message.tool_calls

        for tool_call in tool_calls:
            output = next((o["output"] for o in tool_outputs if o["tool_call_id"] == tool_call.id), None)
            if output:
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": output,
                })

        system_prompt = {"role": "system", "content": assistant["instructions"]}

        response = self.client.chat.completions.create(
            model=assistant["model"],
            messages=[system_prompt] + messages,
        )

        response_message = response.choices[0].message
        self.threads[thread_id].append(response_message)

        return {"status": "completed"}
