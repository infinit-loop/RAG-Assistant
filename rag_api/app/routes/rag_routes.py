"""HTTP routes. Each route delegates to a controller method."""
from fastapi import APIRouter, UploadFile, File, Form
from app.dto.schemas import (
    AskRequest, AskResponse, StructuredRequest, StructuredResponse, HealthResponse,
    UploadResponse, SuggestResponse, SessionAskRequest,
    AgentAskRequest, AgentAskResponse,
)
from app.controllers.rag_controller import RAGController

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse, tags=["agent"])
def agent_ask(payload: AgentAskRequest):
    """Smart routing: classifies the message and routes it to the structured,
    document, chitchat, or off-topic handler. `source` selects base corpus vs
    the session's uploaded documents for document questions."""
    return RAGController.ask_agent(payload)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return RAGController.health()


@router.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest):
    """Ask a grounded question against the base document corpus."""
    return RAGController.ask(payload)


@router.post("/structured", response_model=StructuredResponse, tags=["rag"])
def structured(payload: StructuredRequest):
    """Run a predefined analytical query over the structured (CSV) data."""
    return RAGController.structured(payload)


# ----- uploaded-document (session) endpoints -----
@router.post("/upload", response_model=UploadResponse, tags=["session"])
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a .txt/.md/.csv file into a session-scoped index (persisted)."""
    data = await file.read()
    return RAGController.upload(session_id, file.filename, data)


@router.get("/suggest", response_model=SuggestResponse, tags=["session"])
def suggest(session_id: str):
    """Get auto-generated suggested questions about the session's documents."""
    return RAGController.suggest(session_id)


@router.post("/session/ask", response_model=AskResponse, tags=["session"])
def session_ask(payload: SessionAskRequest):
    """Ask a question grounded ONLY in the session's uploaded documents."""
    return RAGController.ask_session(payload)