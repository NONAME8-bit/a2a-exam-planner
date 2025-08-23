import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from app.agent import ExamGeneratorAgent
from app.agent_executor import ExamGeneratorAgentExecutor
from dotenv import load_dotenv

# Set Google API key (for Gemini model)
os.environ["GOOGLE_API_KEY"] = "AIzaSyD8mJuls7tviOQSgBFF1cWHNd-QRKAf420"
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception raised when GOOGLE_API_KEY is not available."""


def main():
    """
    Starts ExamGeneratorn's Agent server.

    This agent is responsible for generating exam papers based on an approved exam plan.
    It listens for incoming requests (from the Host Agent) via A2A,
    executes tasks with the ExamGeneratorAgentExecutor, and returns generated exams.
    """
    host = "localhost"
    port = 10004
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

        # Define agent capabilities
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)

        # Define the primary skill for this agent
        skill = AgentSkill(
            id="exam_generator",
            name="Exam Generation Tool",
            description="Generates exams based on a given exam plan (topics, difficulty, grade level).",
            tags=["exam", "education", "generation"],
            examples=["Generate a 10-question exam from the following plan..."],
        )

        # Agent card metadata (advertises what this agent can do)
        agent_card = AgentCard(
            name="ExamGenerator Agent",
            description="Generates exams strictly based on approved exam plans provided by the Host Agent.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=ExamGeneratorAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=ExamGeneratorAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # Set up request handler and server
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=ExamGeneratorAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        # Run the agent server
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
