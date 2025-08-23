import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
import json
import os
from .remote_agent_connection import RemoteAgentConnections

from langfuse import Langfuse




langfuse = Langfuse(secret_key="sk-lf-a4ff8a03-5bbc-40eb-bc53-934d117585c3", public_key="pk-lf-bc8c75d2-f010-4972-9912-4944219c4e29", host="https://cloud.langfuse.com" )

REQUIRED_MCP_TOOLS = {
    "list_examhall_availabilities",
    "book_examhall",
    "get_exam_hall_summary",
}


GEMINI_API_KEY= "AIzaSyD8mJuls7tviOQSgBFF1cWHNd-QRKAf420"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY 

nest_asyncio.apply()


class HostAgent:
    """The Host agent."""

    def __init__(self):
        self.tools = None
        self.tool_map = {}
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""

        self.sse_connection = None
        self.session_manager = None
        self.mcp_session = None

        self._agent = self.create_agent()
        self._user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),##Stores artifacts in memory (RAM)
            session_service=InMemorySessionService(),##Keeps track of active sessions in memory.
            memory_service=InMemoryMemoryService(),##Provides longer-term conversational memory for agents
        )

  
    # MCP tools setup
   
    async def _setup_mcp_connection(self):
        self.sse_connection = sse_client("http://127.0.0.1:8787/sse")
        read_stream, write_stream = await self.sse_connection.__aenter__()
        self.session_manager = ClientSession(read_stream, write_stream)
        self.mcp_session = await self.session_manager.__aenter__()
        await self.mcp_session.initialize()
        return await load_mcp_tools(session=self.mcp_session)

    async def _ensure_tools_loaded(self):
        if self.tools is None:
            all_tools = await self._setup_mcp_connection()
            self.tools = [t for t in all_tools if t.name in REQUIRED_MCP_TOOLS]
            self.tool_map = {t.name: t for t in self.tools}

            

    async def _cleanup_mcp_connection(self):
        try:
            if self.session_manager:
                await self.session_manager.__aexit__(None, None, None)
            if self.sse_connection:
                await self.sse_connection.__aexit__(None, None, None)
        except Exception:
            pass

   
    async def _call_simple(self, tool_name: str, **kwargs) -> str:
        await self._ensure_tools_loaded()

        tool = self.tool_map.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool '{tool_name}' not found")

        # Create a span for the tool call using v3 API
        with langfuse.start_as_current_span(
            name=f"tool_call:{tool_name}",
            input=kwargs,
        ) as span:
            try:
                result = await tool.ainvoke(kwargs)
                text = result if isinstance(result, str) else str(result)
                span.update(output={"result": text})
                return text.strip() or "No output produced."
            except Exception as e:
                span.update(
                    output={"error": str(e)},
                    level="ERROR"
                )
                raise

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No friends found"

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]):
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

  
    # MCP Tool wrappers

    async def list_examhall_availabilities_mcp(self, date: str):
        """MCP: List available exam hall slots for a given date (YYYY-MM-DD)."""
        return await self._call_simple("list_examhall_availabilities", date=date)
    async def book_examhall_mcp(self, date: str, student_start: str, student_end: str, reservation_name: str):
     """MCP: Book the exam hall for a time range on the given date."""
     result = await self._call_simple(
        "book_examhall",
        date=date,
        student_start=student_start,
        student_end=student_end,
        reservation_name=reservation_name,
     )

    # Save booking immediately in memory
     self.last_booking = {
        "date": date,
        "start": student_start,
        "end": student_end,
        "reservation": reservation_name,
        "confirmation": result,
     }

     return result



    
    async def get_exam_hall_summary_mcp(self):
        """MCP: Summary of exam hall bookings for the next 7 days."""
        return await self._call_simple("get_exam_hall_summary")

   
    # ADK Agent definition
    
    def create_agent(self) -> Agent:
        return Agent(
            model="gemini-1.5-flash",
            name="Host_Agent",
            instruction=self.root_instruction,
            description="This Host agent orchestrates exam hall bookings and coordinates with friend agents.",
            tools=[
                self.send_message,
                self.list_examhall_availabilities_mcp,
                self.book_examhall_mcp,
                self.get_exam_hall_summary_mcp,
                self.build_final_output, 
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        return f"""
**Role:** You are the Host Agent.  
You orchestrate exam creation and scheduling by gathering exam requirements from the user and coordinating with three agents.  

**Important Rule:**  
 After each step, STOP and WAIT for explicit user approval before moving on.  
Do not automatically continue to the next step. Always confirm with the user first.  

**Step-by-Step Workflow:**

1. **Collect requirements from the user**  
   - Subject  
   - Topic(s)  
   - Grade level  
   - Preferred exam date  

2. **Send details to ExamPlanner Agent** using `send_message`.  
   - After ExamPlanner replies, summarize the plan to the user.  
   - Ask: *"Do you approve this exam plan?"*  
   - Wait for user response before continuing.  

3. **Send approved plan to ExamGenerator Agent** using `send_message`.  
   - After ExamGenerator replies, present the generated exam to the user.  
   - Ask: *"Do you approve this exam?"*  
   - Wait for user response before continuing.  

4. **Ask StudentAvailability Agent for student availability** using `send_message`.  
   - If the user already gave you a specific date for the exam (from Step 1), use that date directly when asking ExamGenerator. Do not ask the user again about the date unless they explicitly request to change or expand it.  
   - You may also ask StudentAvailability about availability in different ways if the user's request calls for it:  
     • A specific day (e.g., "What is StudentAvailability's availability on Tuesday?")  
     • A range of days (e.g., "What is the availability from Monday to Friday?")  
     • The entire week (e.g., "What are my available options this week?")  
   - After StudentAvailability replies, summarize the availability clearly.  
   - Ask the user: *"Is this availability acceptable?"*  
   - Wait for user response before continuing.  

5. **Find exam hall slots** using `list_examhall_availabilities`.  
   - Show available slots to the user.  
   - And find out if there is an available slot with the chosen date and the chosen hour
   - WAIT FOR USER RESPONSE and tell him if there is or no

6. **Reserve exam hall** using `book_examhall`, book it with the respected given date by the user.  
   - Confirm booking details back to the user.  

7. **Final output**  
   - Call build_final_output to return the final exam package IMPORTANT: PLS RESEND THE EXAM PAPER GENERATED FROM ExamPlanner aswell, re-include the the exam content again 

**Today's Date:** {datetime.now().strftime("%Y-%m-%d")}

<Available Agents>
{self.agents}
</Available Agents>
        """

 
    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
       ## Start a new Langfuse tracing span for this function call
        with langfuse.start_as_current_span(
            name="HostAgent.stream",
            input={"query": query},
        ) as root_span:
            # Set trace-level attributes using v3 API
            root_span.update_trace(
                session_id=session_id,
                metadata={"agent_type": "host_agent"},
                tags=["stream"]
            )
            
            try:
                session = await self._runner.session_service.get_session(
                    app_name=self._agent.name,
                    user_id=self._user_id,
                    session_id=session_id,
                )
                ## Wrap the user query in the format Gemini expects
                content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
 
                if session is None:
                    session = await self._runner.session_service.create_session(
                        app_name=self._agent.name,
                        user_id=self._user_id,
                        state={},
                        session_id=session_id,
                    )
                ##It runs Gemini + tools, and streams back events one by one (partial updates, tool calls, final output).
                async for event in self._runner.run_async(
                    user_id=self._user_id, session_id=session.id, new_message=content
                ):
                    if event.is_final_response():
                        response = ""
                        if event.content and event.content.parts and event.content.parts[0].text:
                            response = "\n".join([p.text for p in event.content.parts if p.text])

                        root_span.update(output={"response": response})
                        root_span.update_trace(output={"response": response})
                        
                        yield {
                            "is_task_complete": True,
                            "content": response,
                        }
                    else:
                        yield {
                            "is_task_complete": False,
                            "updates": "The host agent is thinking...",
                        }
            except Exception as e:
                root_span.update(
                    output={"error": str(e)},
                    level="ERROR"
                )
                root_span.update_trace(
                    output={"error": str(e)},
                    metadata={"error": True}
                )
                raise

    def build_final_output(self) -> str:
     date = getattr(self, "last_booking", {}).get("date", "N/A")
     start = getattr(self, "last_booking", {}).get("start", "N/A")
     end = getattr(self, "last_booking", {}).get("end", "N/A")
     hall_confirmation = getattr(self, "last_booking", {}).get("confirmation", "No hall booking")

     exam_paper = getattr(self, "last_ExamGenerator_exam", "[No exam content found from ExamGenerator]")

     return f"""
==============  FINAL EXAM PACKAGE =================

 Date: {date}
 Time: {start} - {end}
 Exam Hall: {hall_confirmation}

 Generated Exam Paper:

{exam_paper}
"""
    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
     with langfuse.start_as_current_span(
        name=f"send_message:{agent_name}",
        input={"task": task, "agent_name": agent_name},
     ) as span:
        try:
            if agent_name not in self.remote_agent_connections:
                error_msg = f"Agent {agent_name} not found"
                span.update(output={"error": error_msg}, level="ERROR")
                raise ValueError(error_msg)

            client = self.remote_agent_connections[agent_name]

            if not client:
                error_msg = f"Client not available for {agent_name}"
                span.update(output={"error": error_msg}, level="ERROR")
                raise ValueError(error_msg)

            
            state = tool_context.state
            task_id = state.get("task_id", str(uuid.uuid4()))
            context_id = state.get("context_id", str(uuid.uuid4()))
            message_id = str(uuid.uuid4())

           
            if agent_name == "ExamGenerator" and getattr(self, "last_ExamPlanner_plan", None):
                try:
                    plan_text = json.dumps(self.last_ExamPlanner_plan, indent=2)
                except Exception:
                    plan_text = str(self.last_ExamPlanner_plan)

                combined_text = f"{task}\n\nHere is ExamPlanner's exam plan:\n{plan_text}"
                parts = [{"type": "text", "text": combined_text}]
            else:
                parts = [{"type": "text", "text": task}]

            ## Wrap in SendMessageRequest which is an A2A request object.
            payload = {
                "message": {
                    "role": "user",
                    "parts": parts,
                    "messageId": message_id,
                    "taskId": task_id,
                    "contextId": context_id,
                },
            }

            # Debug: show payload before sending to ExamGenerator
            if agent_name == "ExamGenerator":
                print("Payload sent to ExamGenerator:")
                print(json.dumps(payload, indent=2))

            message_request = SendMessageRequest(
                id=message_id, params=MessageSendParams.model_validate(payload)
            )

            send_response: SendMessageResponse = await client.send_message(message_request)
            print("send_response", send_response)

            if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(
                send_response.root.result, Task
            ):
                err_msg = "Received a non-success or non-task response. Cannot proceed."
                print(err_msg)
                span.update(output={"error": err_msg}, level="ERROR")
                return

            response_content = send_response.root.model_dump_json(exclude_none=True)
            json_content = json.loads(response_content)
            ## unpack the response you got back from the remote agent.
            resp = []
            if json_content.get("result", {}).get("artifacts"):
                for artifact in json_content["result"]["artifacts"]:
                    if artifact.get("parts"):
                        resp.extend(artifact["parts"])

            # Cache ExamPlanner’s plan
            if agent_name == "ExamPlanner":
                self.last_ExamPlanner_plan = resp
            if agent_name == "ExamGenerator":
             text_parts = [p.get("text", "") for p in resp if isinstance(p, dict) and p.get("type") == "text"]
             self.last_ExamGenerator_exam = "\n".join(text_parts)
            span.update(output={"response": resp})
            return resp

        except Exception as e:
            span.update(
                output={"exception": str(e)},
                level="ERROR"
            )
            raise



