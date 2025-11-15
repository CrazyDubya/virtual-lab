import gradio as gr
from virtual_lab.run_meeting import run_meeting
from virtual_lab.agent import Agent
from pathlib import Path
import os
import json

# A dictionary to store defined agents
defined_agents = {}

def add_agent(title, expertise, goal, role, model, provider):
    agent = Agent(
        title=title,
        expertise=expertise,
        goal=goal,
        role=role,
        model=model,
        provider=provider,
    )
    defined_agents[title] = agent
    return f"Agent '{title}' added.", list(defined_agents.keys())

def run_meeting_interface(agenda, team_lead_title, team_member_titles):
    if not agenda:
        return "Agenda cannot be empty."
    if not team_lead_title:
        return "Please select a team lead."
    if not team_member_titles:
        return "Please select at least one team member."

    team_lead = defined_agents[team_lead_title]
    team_members = tuple(defined_agents[title] for title in team_member_titles)

    save_dir = Path("discussions")
    save_dir.mkdir(exist_ok=True)

    # Simplified call for now
    run_meeting(
        meeting_type="team",
        agenda=agenda,
        save_dir=save_dir,
        team_lead=team_lead,
        team_members=team_members,
        num_rounds=1,
    )

    discussion_file = save_dir / "discussion.json"
    with open(discussion_file, "r") as f:
        discussion_data = json.load(f)

    # Format discussion for display
    discussion_text = ""
    for turn in discussion_data:
        discussion_text += f"**{turn['agent']}**: {turn['message']}\n\n"

    return discussion_text

with gr.Blocks() as demo:
    gr.Markdown("# Virtual Lab GUI")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 1. Define Agents")
            agent_title = gr.Textbox(label="Title")
            agent_expertise = gr.Textbox(label="Expertise")
            agent_goal = gr.Textbox(label="Goal")
            agent_role = gr.Textbox(label="Role")
            agent_model = gr.Textbox(label="Model", value="gpt-4o-2024-08-06")
            agent_provider = gr.Dropdown(
                label="Provider",
                choices=["openai", "groq", "anthropic", "ollama", "gemini", "openrouter"],
                value="openai",
            )
            add_agent_button = gr.Button("Add Agent")
            agent_status = gr.Textbox(label="Status", interactive=False)

            defined_agents_list = gr.CheckboxGroup(label="Defined Agents", interactive=False)
            add_agent_button.click(
                fn=add_agent,
                inputs=[
                    agent_title,
                    agent_expertise,
                    agent_goal,
                    agent_role,
                    agent_model,
                    agent_provider,
                ],
                outputs=[agent_status, defined_agents_list],
            )

        with gr.Column():
            gr.Markdown("## 2. Run Meeting")
            agenda_input = gr.Textbox(label="Agenda", lines=5)
            team_lead_dropdown = gr.Dropdown(label="Team Lead", choices=list(defined_agents.keys()))
            team_members_checklist = gr.CheckboxGroup(label="Team Members", choices=list(defined_agents.keys()))

            run_button = gr.Button("Run Meeting")

    gr.Markdown("## 3. Discussion")
    discussion_output = gr.Markdown()

    run_button.click(
        fn=run_meeting_interface,
        inputs=[agenda_input, team_lead_dropdown, team_members_checklist],
        outputs=[discussion_output],
    )

    # Update dropdowns when new agents are added
    def update_agent_lists():
        agent_names = list(defined_agents.keys())
        return gr.Dropdown(choices=agent_names), gr.CheckboxGroup(choices=agent_names)

    demo.load(update_agent_lists, [], [team_lead_dropdown, team_members_checklist])
    add_agent_button.click(update_agent_lists, [], [team_lead_dropdown, team_members_checklist])


if __name__ == "__main__":
    demo.launch()
