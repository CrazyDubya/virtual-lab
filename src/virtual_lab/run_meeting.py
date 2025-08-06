"""Runs a meeting with LLM agents."""

import time
from pathlib import Path
from typing import Literal

from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import CONSISTENT_TEMPERATURE
from virtual_lab.llm.base import LLMClient
from virtual_lab.llm.openai import OpenAIClient
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import (
    convert_messages_to_discussion,
    count_discussion_tokens,
    count_tokens,
    get_summary,
    print_cost_and_time,
    save_meeting,
)


from virtual_lab.llm.groq import GroqClient
from virtual_lab.llm.anthropic import AnthropicClient
from virtual_lab.llm.ollama import OllamaClient
from virtual_lab.llm.gemini import GeminiClient
from virtual_lab.llm.openrouter import OpenRouterClient


def get_llm_client(agent: Agent) -> LLMClient:
    """Gets the LLM client for the agent.

    :param agent: The agent.
    :return: The LLM client.
    """
    provider = getattr(agent, "provider", "openai")
    if provider == "openai":
        return OpenAIClient()
    if provider == "groq":
        return GroqClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "ollama":
        return OllamaClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "openrouter":
        return OpenRouterClient()
    raise ValueError(f"Invalid provider: {provider}")


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Agent | None = None,
    team_members: tuple[Agent, ...] | None = None,
    team_member: Agent | None = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    pubmed_search: bool = False,
    return_summary: bool = False,
) -> str:
    """Runs a meeting with a LLM agents.

    :param meeting_type: The type of meeting.
    :param agenda: The agenda for the meeting.
    :param save_dir: The directory to save the discussion.
    :param save_name: The name of the discussion file that will be saved.
    :param team_lead: The team lead for a team meeting (None for individual meeting).
    :param team_members: The team members for a team meeting (None for individual meeting).
    :param team_member: The team member for an individual meeting (None for team meeting).
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the meeting.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param temperature: The sampling temperature.
    :param pubmed_search: Whether to include a PubMed search tool.
    :param return_summary: Whether to return the summary of the meeting.
    :return: The summary of the meeting (i.e., the last message) if return_summary is True, else None.
    """
    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError(
                "Individual meeting does not require team lead or team members"
            )
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    # Set up team
    if meeting_type == "team":
        team = [team_lead] + list(team_members)
    else:
        team = [team_member] + [SCIENTIFIC_CRITIC]

    # Get LLM client from the first agent
    # TODO: Handle multiple clients if agents have different providers
    llm_client = get_llm_client(team[0])

    # Set up the assistants
    agent_to_assistant = {
        agent: llm_client.create_assistant(agent=agent, pubmed_search=pubmed_search)
        for agent in team
    }

    # Map assistant IDs to agents
    assistant_id_to_title = {
        assistant.id: agent.title for agent, assistant in agent_to_assistant.items()
    }

    # Set up tool token count
    tool_token_count = 0

    # Set up the thread
    thread = llm_client.create_thread()

    # Initial prompt for team meeting
    if meeting_type == "team":
        llm_client.create_message(
            thread_id=thread.id,
            content=team_meeting_start_prompt(
                team_lead=team_lead,
                team_members=team_members,
                agenda=agenda,
                agenda_questions=agenda_questions,
                agenda_rules=agenda_rules,
                summaries=summaries,
                contexts=contexts,
                num_rounds=num_rounds,
            ),
        )

    # Loop through rounds
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        # Loop through team and elicit responses
        for agent in tqdm(team, desc="Team"):
            # Prompt based on agent and round number
            if meeting_type == "team":
                # Team meeting prompts
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(
                            team_lead=team_lead
                        )
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent, round_num=round_num, num_rounds=num_rounds
                    )
            else:
                # Individual meeting prompts
                if agent == SCIENTIFIC_CRITIC:
                    prompt = individual_meeting_critic_prompt(
                        critic=SCIENTIFIC_CRITIC, agent=team_member
                    )
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(
                            critic=SCIENTIFIC_CRITIC, agent=team_member
                        )

            # Create message from user to agent
            llm_client.create_message(thread_id=thread.id, content=prompt)

            # Run the agent
            run = llm_client.run_thread_and_poll(
                thread_id=thread.id,
                assistant_id=agent_to_assistant[agent].id,
                model=agent.model,
                temperature=temperature,
            )

            # Check if run requires action
            if run.status == "requires_action":
                # Run the tools
                tool_outputs = llm_client.run_tools(run=run)

                # Update tool token count
                tool_token_count += sum(
                    count_tokens(tool_output["output"]) for tool_output in tool_outputs
                )

                # Submit the tool outputs
                run = llm_client.submit_tool_outputs_and_poll(
                    run=run, tool_outputs=tool_outputs
                )

                # Add tool outputs to the thread so it's visible for later rounds
                llm_client.create_message(
                    thread_id=thread.id,
                    content="Tool Output:\n\n"
                    + "\n\n".join(
                        tool_output["output"] for tool_output in tool_outputs
                    ),
                )

            # Check run status
            if run.status != "completed":
                raise ValueError(f"Run failed: {run.status}")

            # If final round, only team lead or team member responds
            if round_index == num_rounds:
                break

    # Get messages from the discussion
    messages = llm_client.get_messages(thread_id=thread.id)

    # Convert messages to discussion format
    discussion = convert_messages_to_discussion(
        messages=messages, assistant_id_to_title=assistant_id_to_title
    )

    # Count discussion tokens
    token_counts = count_discussion_tokens(discussion=discussion)

    # Add tool token count to total token count
    token_counts["tool"] = tool_token_count

    # Print cost and time
    # TODO: handle different models for different agents
    print_cost_and_time(
        token_counts=token_counts,
        model=team_lead.model if meeting_type == "team" else team_member.model,
        elapsed_time=time.time() - start_time,
    )

    # Save the discussion as JSON and Markdown
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion,
    )

    # Optionally, return summary
    if return_summary:
        return get_summary(discussion)
