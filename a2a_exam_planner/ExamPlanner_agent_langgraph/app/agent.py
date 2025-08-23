# app/agent.py
import asyncio
import re
import logging
from typing import Any, AsyncIterable, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, END



# Logging setup

logger = logging.getLogger("ExamPlannerAgentManualLog")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                # console
        logging.FileHandler("ExamPlanner_agent.log")   # persistent log file
    ]
)


# ExamPlanner uses exam planner tool instead of generator
REQUIRED_MCP_TOOLS = {"generate_exam_plan"}



#  Graph State

class ExamState(TypedDict, total=False):
    query: str
    topic: Optional[str]
    draft: str
    output: str
    status: str



#  Node functions

def parse_topic(state: ExamState) -> ExamState:
    """Extract topic from the user query string."""
    q = state["query"].strip()
    topic = q

    if not q.startswith('{') and not 'plan:' in q.lower():
        m = re.search(r"topic\s*=\s*([^\n,]+)", q, re.IGNORECASE)
        if m:
            topic = m.group(1).strip()
        elif " about " in q.lower():
            topic = q.split(" about ", 1)[1].strip()
        elif " on " in q.lower():
            topic = q.split(" on ", 1)[1].strip()

    new_state = {**state, "topic": topic or q, "status": "parsed"}
    logger.info(f"[Parse Node] Input={q} → Extracted Topic={topic}")
    return new_state


def make_generate_node(tools):
    async def generate_node(state: ExamState) -> ExamState:
        try:
            tool = next(t for t in tools if t.name == "generate_exam_plan")
            payload = {
                "topic": state.get("topic", "")
            }
            logger.info(f"[Generate Node] Calling tool 'generate_exam_plan' with payload={payload}")

            result = await tool.ainvoke(payload)
            text = result if isinstance(result, str) else str(result)

            logger.info(f"[Generate Node] Tool result length={len(text)} chars")
            return {**state, "draft": text.strip(), "status": "planned"}
        except Exception as e:
            logger.error(f"[Generate Node] Error: {e}")
            return {**state, "draft": f"ERROR: {e}", "status": "error"}
    return generate_node


def format_output(state: ExamState) -> ExamState:
    """Format the exam plan into final output."""
    draft_text = state.get("draft", "")
    formatted = draft_text
    logger.info(f"[Format Node] Formatting plan (length={len(draft_text)})")
    return {**state, "output": formatted, "status": "formatted"}



#  Graph builder

def build_exam_graph(tools):
    graph = StateGraph(ExamState)

    graph.add_node("parse", parse_topic)
    graph.add_node("generate", make_generate_node(tools))
    graph.add_node("format", format_output)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "generate")
    graph.add_edge("generate", "format")
    graph.add_edge("format", END)

    return graph.compile()


#  ExamPlanner Agent

class ExamPlannerAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self.graph = None
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # unused but kept
        self.tools = None
        self.tool_map = {}
        self.mcp_session = None
        self.sse_connection = None
        self.session_manager = None

    
    async def _setup_mcp_connection(self):
        self.sse_connection = sse_client("http://127.0.0.1:8787/sse")
        read_stream, write_stream = await self.sse_connection.__aenter__()
        self.session_manager = ClientSession(read_stream, write_stream)
        self.mcp_session = await self.session_manager.__aenter__()
        await self.mcp_session.initialize()
        tools = await load_mcp_tools(session=self.mcp_session)
        return tools

    async def _ensure_tools_loaded(self):
        if self.tools is None:
            self.tools = await self._setup_mcp_connection()
            self.tool_map = {t.name: t for t in self.tools}
            if "generate_exam_plan" not in self.tool_map:
                raise RuntimeError(
                    f"'generate_exam_plan' not found. Available tools: {list(self.tool_map)}"
                )

    async def _call_simple(self, tool_name: str, **kwargs) -> str:
        """Call MCP tool safely."""
        await self._ensure_tools_loaded()
        if tool_name not in REQUIRED_MCP_TOOLS:
            raise ValueError(f"Tool '{tool_name}' is not allowed")
        tool = self.tool_map.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool '{tool_name}' not found in tool_map")

        logger.info(f"[Agent] Calling tool={tool_name} with kwargs={kwargs}")
        result = await tool.ainvoke(kwargs)
        text = result if isinstance(result, str) else str(result)
        logger.info(f"[Agent] Tool={tool_name} returned length={len(text)} chars")
        return text

    async def _cleanup_mcp_connection(self):
        try:
            if self.session_manager:
                await self.session_manager.__aexit__(None, None, None)
            if self.sse_connection:
                await self.sse_connection.__aexit__(None, None, None)
        except Exception:
            pass

    async def _ensure_graph(self):
        if self.graph is None:
            await self._ensure_tools_loaded()
            self.graph = build_exam_graph(self.tools)

    

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]: ## angGraph execution produces results step by step,
        """Yield updates as the LangGraph progresses through nodes."""
        try:
            await self._ensure_graph()
            initial_state: ExamState = {
                "query": query,
                "topic": "",
                "draft": "",
                "output": "",
                "status": "start",
            }
            final_state = initial_state

            logger.info(f"[Agent] Streaming run started query={query}")

            async for event in self.graph.astream(initial_state):
                if "node" in event:
                    node = event["node"]
                    final_state = event["state"]
                    logger.info(f"[Stream] Node={node}, State={final_state}")
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Step {node}: {final_state.get('status') or ''}",
                    }
                else:
                    for key, value in event.items():
                        if isinstance(value, dict) and "output" in value:
                            final_state = value
                            break

            final_content = final_state.get("output") or final_state.get("draft") or ""
            logger.info(f"[Stream] Final content length={len(final_content)}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_content,
            }

        except Exception as e:
            logger.error(f"[Stream] Error: {e}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": f"Error: {e!r}",
            }

    async def __aenter__(self):
        await self._ensure_tools_loaded()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup_mcp_connection()
