"""Controllers contain request-handling logic and map engine output to DTOs.
They stay thin: validation is in DTOs, business logic is in the core engine."""
from app.dto.schemas import (
    AskRequest, AskResponse, StructuredRequest, StructuredResponse, HealthResponse,
    UploadResponse, SuggestResponse, SessionAskRequest,
    AgentAskRequest, AgentAskResponse,
)
from app.services.rag_service import get_engine
from app.services.session_service import get_session_manager
from app.common.constants import AppError


class RAGController:
    @staticmethod
    def ask(payload: AskRequest) -> AskResponse:
        engine = get_engine()
        result = engine.answer(payload.question)
        return AskResponse(**result)

    @staticmethod
    def structured(payload: StructuredRequest) -> StructuredResponse:
        engine = get_engine()
        try:
            result = engine.query_structured(payload.intent)
        except ValueError as e:
            raise AppError(str(e), status_code=400)
        return StructuredResponse(intent=payload.intent, result=result)

    @staticmethod
    def health() -> HealthResponse:
        engine = get_engine()
        h = engine.health()
        return HealthResponse(status="ok", **h)

    # ----- session / uploaded documents -----
    @staticmethod
    def upload(session_id: str, filename: str, data: bytes) -> UploadResponse:
        mgr = get_session_manager()
        try:
            res = mgr.add_file(session_id, filename, data)
        except ValueError as e:
            raise AppError(str(e), status_code=400)
        return UploadResponse(**res)

    @staticmethod
    def suggest(session_id: str) -> SuggestResponse:
        mgr = get_session_manager()
        return SuggestResponse(session_id=session_id,
                               questions=mgr.suggest(session_id))

    @staticmethod
    def ask_session(payload: SessionAskRequest) -> AskResponse:
        mgr = get_session_manager()
        return AskResponse(**mgr.answer(payload.session_id, payload.question))

    @staticmethod
    def ask_agent(payload: AgentAskRequest) -> AgentAskResponse:
        from app.agent.graph import run_agent
        res = run_agent(payload.question, payload.session_id, payload.source)
        return AgentAskResponse(**res)