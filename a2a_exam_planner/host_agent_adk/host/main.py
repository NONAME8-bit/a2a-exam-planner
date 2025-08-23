import asyncio
import sys
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.agent import HostAgent


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


app = FastAPI(title="HostAgent API", version="1.0")


host_agent = None

@app.on_event("startup")
async def startup_event():
    global host_agent
    friend_agent_urls = [
        "http://localhost:10002",  
        "http://localhost:10003",  
        "http://localhost:10004",  
    ]
    host_agent = await HostAgent.create(friend_agent_urls)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default-session"

class ChatResponse(BaseModel):
    is_task_complete: bool
    content: Optional[str] = None
    updates: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    responses = []
    async for event in host_agent.stream(request.query, request.session_id):
        responses.append(event)

    if responses:
        return responses[-1]
    return {"is_task_complete": False, "updates": "No response generated."}


@app.get("/")
async def root():
    return {"status": "HostAgent is running"}


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
