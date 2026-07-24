
import os
import sys

from dotenv import load_dotenv

from anthropic import (
    Anthropic
)

# Load environment variables from the .env file so ANTHROPIC_API_KEY is
# available to the Anthropic SDK (which reads it automatically).
load_dotenv()

from anthropic.types.beta import (
    BetaEnvironment,
    BetaManagedAgentsAgent
)


OPEN_ENVIRONMMENT_NAME = "Open Environment" 
LOCKED_ENVIRONMENT_NAME = "Locked Environment" 

PYTHON_AGENT_NAME = "Python Agent"
FULL_AGENT_NAME = "Full Agent"

OPUS_MODEL_NAME="claude-opus-4-8" 
HAIKU_MODEL_NAME="claude-haiku-4-5-20251101" 


def create_environment ( client: Anthropic , environment_name : str ) -> BetaEnvironment:  
    environment = client.beta.environments.create(
        name= environment_name ,
        config={
            "type": "cloud",
            "packages": {
                "pip": ["pandas", "numpy", "scikit-learn"],
                "npm": ["express"],
            },
            "networking": {"type": "unrestricted"},
        },
    )
    return environment 


def create_agent ( client: Anthropic, agent_name: str , model : str  ) -> BetaManagedAgentsAgent: 
    agent = client.beta.agents.create(
        name= agent_name ,
        model= model ,
        system="You are a helpful coding assistant. Write clean, well-documented code.",
        tools=[
            {"type": "agent_toolset_20260401"},
        ],
    )
    return agent 


def main ():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        sys.exit(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your real key (or export ANTHROPIC_API_KEY in your shell)."
        )

    client = Anthropic()

    environment_name = OPEN_ENVIRONMMENT_NAME 
    agent_name = FULL_AGENT_NAME 
    agent_id = "agent_0115Y8Fb1PNChQUoseuoafwv" 
    environment_id = "env_01Sze1ytPCRjWkCMosoBEMEP" 

    environment = client.beta.environments.retrieve(environment_id) if environment_id else ( client , OPEN_ENVIRONMMENT_NAME ) 
    assert environment , "Expected environment" 

    if agent_id:
        agent = client.beta.agents.retrieve ( agent_id )                  
    else:          
        agent =  create_agent( client) 
        print (f"Created agent {agent.name} ( id: {agent.id})" )

    assert agent , "We should have an agent before creating session " 
    

    session = client.beta.sessions.create(
        agent=agent.id,
        environment_id=environment.id,
        title="Quickstart session",
    ) 

    print(f"Session ID: {session.id}")

    with client.beta.sessions.events.stream(session.id) as stream:
        # Send the user message after the stream opens
        client.beta.sessions.events.send(
            session.id,
            events=[
                {
                    "type": "user.message",
                    "content": [
                        {
                            "type": "text",
                            "text": "Create a Python script that generates the first 20 Fibonacci numbers and saves them to fibonacci.txt",
                        },
                    ],
                },
            ],
        )

        # Process streaming events
        for event in stream:
            match event.type:
                case "agent.message":
                    for block in event.content:
                        print(block.text, end="")
                case "agent.tool_use":
                    print(f"\n[Using tool: {event.name}]")
                case "session.status_idle":
                    print("\n\nAgent finished.")
                    break


if __name__ == "__main__": 
    main() 