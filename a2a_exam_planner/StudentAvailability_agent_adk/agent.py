import asyncio
import json
from typing import List, Optional

import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

from a2a.client import A2ACardResolver
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client



REQUIRED_MCP_TOOLS = {
    "query_students_availability"
}


class StudentAvailabilityAgent:
    """The Student Availability agent."""

    def __init__(self):
        self.tools = None
        self.tool_map = {}
        self.sse_connection = None
        self.session_manager = None
        self.mcp_session = None

        # Build ADK LlmAgent wrapper
        self._agent = self.create_agent()
        

    async def _setup_mcp_connection(self):
        """Open SSE connection to MCP server and load tools."""
        self.sse_connection = sse_client("http://127.0.0.1:8787/sse")
        read_stream, write_stream = await self.sse_connection.__aenter__()
        self.session_manager = ClientSession(read_stream, write_stream)
        self.mcp_session = await self.session_manager.__aenter__()
        await self.mcp_session.initialize()
        return await load_mcp_tools(session=self.mcp_session)

    async def _ensure_tools_loaded(self):
        """Lazy load & filter only required MCP tools."""
        if self.tools is None:
            all_tools = await self._setup_mcp_connection()
            self.tools = [t for t in all_tools if t.name in REQUIRED_MCP_TOOLS]
            self.tool_map = {t.name: t for t in self.tools}

            missing = REQUIRED_MCP_TOOLS - set(self.tool_map.keys())
            if missing:
                raise RuntimeError(f"Missing required MCP tools: {missing}")

    async def _cleanup_mcp_connection(self):
        """Gracefully close MCP session."""
        try:
            if self.session_manager:
                await self.session_manager.__aexit__(None, None, None)
            if self.sse_connection:
                await self.sse_connection.__aexit__(None, None, None)
        except Exception:
            pass

    async def _call_simple(self, tool_name: str, **kwargs) -> str:
        """Convenience helper to call MCP tools safely."""
        await self._ensure_tools_loaded()

        if tool_name not in REQUIRED_MCP_TOOLS:
            raise ValueError(f"Tool '{tool_name}' is not allowed")

        tool = self.tool_map.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool '{tool_name}' not found in tool_map")

        result = await tool.ainvoke(kwargs)
        text = result if isinstance(result, str) else str(result)
        return text.strip() or "No output produced."

    # Agent factory  mirrors your HostAgent style
    def create_agent(self) -> LlmAgent:
     return LlmAgent(
        model="gemini-1.5-flash",
        name="ExamPLanner_Agent",   
        instruction=f"""
**Role:** You are *ExamPLanner Agent*. You are NOT a student.  
Your job is to answer availability questions for ALL students by using the tool below.

**Tool you can use:**
- `query_students_availability()`
   • Call this tool **once per session**.  
   • It always returns the free slots for ALL 10 students for the entire week in structured form.  
   • Store the result in memory and reuse it. Do not call the tool again.

**Your tasks:**
1. Always run `query_students_availability()` first (if not already done).
2. Remember the full week's availability in your context.
3. When the user asks about a specific date or day (e.g., "Monday, August 26th"),  
   - Convert that into a weekday (e.g., "Monday").  
   - Filter the stored weekly availability to only that day.  
   - Respond with just that slot, not the full list.  
4. If the user later asks about another day (e.g., "Wednesday"), do **not** re-run the tool.  
   Use the stored weekly results and filter again.
5. If the user asks "When are all students free together?", summarize the full weekly availability.

**Examples:**
- User: "When are all students free together?"  
  → Call tool once → Return summary for all days.

- User: "I want a test on Monday, generate me one."  
  → Call tool once → Filter and respond:  
    "On Monday, all 10 students are free from 11:00 to 12:00."

- User: "How about Wednesday?"  
  → Do NOT re-run tool → Use memory →  
    "On Wednesday, all 10 students are free from 12:00 to 13:00."

- User: "What is ExamPLanner's availability on Tuesday, August 26th, 2025?"  
  → Recognize ExamPLanner here means YOU (the agent).  
  → Convert August 26, 2025 into "Tuesday".  
  → Filter Tuesday slot →  
    "On Tuesday, all 10 students are free from 15:00 to 16:00."
""",
        tools=[self.query_students_availability],
    )



   
    async def query_students_availability(self) -> str:
     return await self._call_simple("query_students_availability")


    
