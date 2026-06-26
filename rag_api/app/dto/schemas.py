"""Request / response schemas (data transfer objects)."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          examples=["What is the escalation process for delayed shipments?"])


class SourceRef(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    type: Literal["answer", "clarify", "abstain", "blocked"]
    answer: str
    sources: list[SourceRef] = []
    confidence: Optional[str] = None
    score: Optional[float] = None
    mode: Optional[str] = None  # "generated" (LLM) or "extractive" (snippets)
    query: Optional[str] = None  # the pandas expression, when mode == "table"


class StructuredRequest(BaseModel):
    intent: Literal["top_branch_sales", "avg_aging",
                    "top5_aging", "aging_over_target"]


class StructuredResponse(BaseModel):
    intent: str
    result: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    structured_rows: int
    llm_enabled: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    pii_engine: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    files: list[str]
    indexed_chunks: int
    warning: Optional[str] = None


class SuggestResponse(BaseModel):
    session_id: str
    questions: list[str]


class SessionAskRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=1000)


class AgentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    source: Literal["base", "uploads"] = "base"


class AgentAskResponse(AskResponse):
    intent: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str