from google.adk.agents import Agent
from google.adk.models import LiteLlm

AGENT_MODEL = 'ollama/gemma3'

root_agent = Agent(
    name="travel_planner_agent",
    model=LiteLlm(model=AGENT_MODEL),
    tools=[],
    description="A helpful travel planner assistant.",
    instruction=(
        "You are a helpful travel planner assistant. You can help users plan their trips "
        "by providing information about destinations, suggesting activities, and answering "
        "any travel-related questions they may have. Always provide helpful and accurate "
        "information. If you don't know the answer to a question, say you don't know rather "
        "than providing incorrect information."
    ),
)
