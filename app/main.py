"""
main.py
-------
FastAPI application. Exposes two endpoints:

    GET  /health  — returns {"status": "ok"} with HTTP 200
    POST /chat    — takes conversation history, returns agent response

This file only handles HTTP concerns (request validation, response formatting).
All agent logic lives in agent.py. All retrieval logic lives in retrieval.py.

How to run locally:
    uvicorn app.main:app --reload --port 8000

Then test with:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import get_agent_response

# FastAPI app
app = FastAPI(title="SHL Assessment Recommender")

# Request and response models
# Pydantic models do two things:
#   1. Validate incoming JSON automatically — wrong types return HTTP 422
#   2. Document the schema in the auto-generated /docs page

class Message(BaseModel):
    role: str       # must be "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str


class ChatResponse(BaseModel):
    reply:               str
    recommendations:     list[Recommendation]
    end_of_conversation: bool

# Endpoints

@app.get("/health")
def health():
    """
    Readiness check. The evaluator calls this before running conversations.
    Must return HTTP 200 with {"status": "ok"}.
    Cold start services get up to 2 minutes before this is checked.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint. Takes the full conversation history and returns the
    agent's next reply plus optional recommendations.

    The API is stateless — the client sends the entire history every time.
    We store nothing on the server between calls.
    """

    # Convert Pydantic models back to plain dicts for agent.py
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Basic validation — empty message list is not actionable
    if not messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty")

    # Delegate to agent — all logic lives there
    result = get_agent_response(messages)

    return result