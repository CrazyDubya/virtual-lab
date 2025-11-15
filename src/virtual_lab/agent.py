"""The LLM agent class."""


class Agent:
    """An LLM agent."""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str,
        provider: str = "openai",
    ) -> None:
        """Initializes the agent.

        :param title: The title of the agent.
        :param expertise: The expertise of the agent.
        :param goal: The goal of the agent.
        :param role: The role of the agent.
        :param model: The model of the agent.
        :param provider: The provider of the agent.
        """
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.provider = provider

    @property
    def prompt(self) -> str:
        """Returns the prompt for the agent."""
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}."
        )

    @property
    def message(self) -> dict[str, str]:
        """Returns the message for the agent in OpenAI API form."""
        return {
            "role": "system",
            "content": self.prompt,
        }

    def __hash__(self) -> int:
        """Returns the hash of the agent."""
        return hash(self.title)

    def __eq__(self, other: object) -> bool:
        """Checks if the agent is equal to another agent (based on title)."""
        if not isinstance(other, Agent):
            return False

        return (
            self.title == other.title
            and self.expertise == other.expertise
            and self.goal == other.goal
            and self.role == other.role
            and self.model == other.model
            and self.provider == other.provider
        )

    def __str__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title

    def __repr__(self) -> str:
        """Returns the string representation of the agent (i.e., the agent's title)."""
        return self.title
