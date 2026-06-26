"""LangGraph agent that routes each message to the right handler.

Flow:
    START -> guard -> classify -> (structured | document | chitchat | offtopic) -> END

Nodes:
  guard      : prompt-injection block (safety, always first)
  classify   : decide intent (LLM if available, else keyword heuristic)
  structured : grounded analytic query over the CSV
  document   : grounded RAG over the base corpus OR the session's uploads
  chitchat   : PII-mask the message, then a friendly LLM reply
  offtopic   : polite decline (NOT grounded -> we don't answer)

The grounded promise is preserved: 'document'/'structured' answers come only
from the indexed data; 'offtopic' is declined; only greetings get a free LLM
reply (and the message is PII-masked before being sent to the LLM).
"""
from __future__ import annotations
import re
from typing import Optional, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from app.services.rag_service import get_engine
from app.services.session_service import get_session_manager
from app.common.constants import INJECTION_PATTERNS
from app.common.logging import get_logger
from app.utils.pii import mask

log = get_logger("agent")

# Keyword fallback for classification when no LLM is configured.
STRUCTURED_HINTS = ("branch", "sales", "aging", "sku", "inventory",
                    "highest", "average", "top ", "lowest", "how many")
GREETING_HINTS = ("hi", "hello", "hey", "yo", "thanks", "thank you",
                  "good morning", "good evening", "how are you", "sup")


class AgentState(TypedDict, total=False):
    question: str
    session_id: Optional[str]      # if set, document mode uses uploaded docs
    source: str                    # "base" or "uploads"
    intent: str
    result: dict                   # final response dict (same shape as /ask)


# ---------------- nodes ----------------
def guard_node(state: AgentState) -> dict:
    q = state["question"]
    if any(p.search(q) for p in INJECTION_PATTERNS):
        log.warning("guard: BLOCKED (prompt-injection pattern matched)")
        return {"intent": "blocked",
                "result": {
                    "type": "blocked",
                    "answer": ("This request looks like an attempt to override "
                               "instructions or extract restricted data. I only "
                               "help with the business documents."),
                    "sources": [], "confidence": None, "score": None,
                    "mode": None}}
    return {}


def classify_node(state: AgentState) -> dict:
    if state.get("intent") == "blocked":
        return {}
    q = state["question"].strip()
    engine = get_engine()

    # Prefer the LLM classifier when available; fall back to keywords.
    label, how = "", "keyword"
    if engine.llm.enabled:
        try:
            label = engine.llm.classify(q)
            how = "llm"
        except Exception as e:  # noqa: BLE001
            log.warning("classify: LLM classifier failed (%s), using keywords", e)
            label = ""
    if not label:
        how = "keyword"
        ql = q.lower()
        words = re.findall(r"\w+", ql)
        if any(h in ql for h in STRUCTURED_HINTS):
            label = "structured"
        elif len(words) <= 3 and any(ql.startswith(g) or ql == g
                                     for g in GREETING_HINTS):
            label = "chitchat"
        else:
            # Default to a grounded document lookup; the document node will
            # abstain if retrieval finds nothing (so off-topic still won't be
            # answered from thin air).
            label = "document"
    log.info("classify: intent=%s (via %s)", label, how)
    return {"intent": label}


def route(state: AgentState) -> Literal[
        "blocked", "structured", "document", "chitchat", "offtopic"]:
    intent = state.get("intent", "document")
    if intent == "blocked":
        return "blocked"
    if intent in ("structured", "document", "chitchat", "offtopic"):
        return intent
    return "document"


_STRUCTURED_MAP = [
    (("highest", "top branch", "branch", "sales"), "top_branch_sales"),
    (("average", "avg", "mean"), "avg_aging"),
    (("top 5", "top five", "top skus", "by aging"), "top5_aging"),
    (("over", "exceed", "target"), "aging_over_target"),
]


def structured_node(state: AgentState) -> dict:
    engine = get_engine()
    q = state["question"]

    # Resolve which table to query: a session's uploaded CSV takes precedence
    # over the base corpus CSV.
    df, table_name = None, None
    if state.get("source") == "uploads" and state.get("session_id"):
        frames = get_session_manager().get_frames(state["session_id"])
        if frames:
            table_name, df = frames[0][0], frames[0][1]
    if df is None and engine.df is not None:
        df, table_name = engine.df, engine.csv_name or "operational_report.csv"

    if df is None:
        # No table to query (e.g. the classifier saw a number and routed here,
        # but the answer actually lives in the documents). Don't dead-end —
        # fall back to grounded document retrieval, which abstains on its own
        # if nothing matches.
        log.info("structured: no tabular data -> falling back to document retrieval")
        return document_node(state)

    # Preferred path: text-to-pandas (works for ANY schema). Requires an LLM to
    # write the pandas expression; pandas computes the actual number/table.
    if engine.llm.enabled:
        log.info("structured: text-to-pandas on '%s' (%d rows)",
                 table_name, len(df))
        return {"result": engine.answer_table(q, df, table_name)}
    log.info("structured: no LLM -> canned-intent fallback on '%s'", table_name)

    # No-LLM fallback: the hardcoded analytic intents (base schema only).
    ql = q.lower()
    intent_key = "avg_aging"
    for keys, val in _STRUCTURED_MAP:
        if any(k in ql for k in keys):
            intent_key = val
            break
    try:
        result_text = engine.query_structured(intent_key)
    except Exception as e:  # noqa: BLE001
        result_text = (f"Could not run that analytic query: {e}. "
                       "(No LLM configured, so only the base report's preset "
                       "questions are supported.)")
    return {"result": {
        "type": "answer", "answer": result_text,
        "sources": [{"source": table_name, "score": 1.0}],
        "confidence": "high", "score": 1.0, "mode": "structured"}}


def document_node(state: AgentState) -> dict:
    engine = get_engine()
    q = state["question"]
    if state.get("source") == "uploads" and state.get("session_id"):
        result = get_session_manager().answer(state["session_id"], q)
    else:
        result = engine.answer(q, skip_ambiguity=True)
    return {"result": result}


def chitchat_node(state: AgentState) -> dict:
    engine = get_engine()
    safe = mask(state["question"])  # PII-mask before sending to the LLM
    if engine.llm.enabled:
        try:
            reply = engine.llm.chat(safe)
        except Exception:
            reply = ("Hello! I can answer questions grounded in your business "
                     "documents - try asking about SOPs, procurement, KPIs, or "
                     "returns.")
    else:
        reply = ("Hello! I can answer questions grounded in your business "
                 "documents - try asking about SOPs, procurement, KPIs, or "
                 "returns.")
    return {"result": {"type": "answer", "answer": reply, "sources": [],
                       "confidence": None, "score": None, "mode": "chitchat"}}


def offtopic_node(state: AgentState) -> dict:
    return {"result": {
        "type": "abstain",
        "answer": ("That looks outside the scope of the business documents I "
                   "work with, so I can't answer it reliably. I can help with "
                   "SOPs, policies, KPIs, procurement, returns, the operational "
                   "data, or any documents you upload."),
        "sources": [], "confidence": None, "score": None, "mode": "offtopic"}}


# ---------------- graph ----------------
def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("guard", guard_node)
    g.add_node("classify", classify_node)
    g.add_node("structured", structured_node)
    g.add_node("document", document_node)
    g.add_node("chitchat", chitchat_node)
    g.add_node("offtopic", offtopic_node)
    g.add_node("blocked", lambda s: {})  # passthrough; result already set

    g.add_edge(START, "guard")
    g.add_edge("guard", "classify")
    g.add_conditional_edges("classify", route, {
        "blocked": "blocked",
        "structured": "structured",
        "document": "document",
        "chitchat": "chitchat",
        "offtopic": "offtopic",
    })
    for n in ("blocked", "structured", "document", "chitchat", "offtopic"):
        g.add_edge(n, END)
    return g.compile()


_graph = None


def get_agent():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_agent(question: str, session_id: str | None = None,
              source: str = "base") -> dict:
    log.info("ASK  source=%s session=%s q=%r", source, session_id, question)
    state = {"question": question, "session_id": session_id, "source": source}
    out = get_agent().invoke(state)
    res = out.get("result") or {
        "type": "abstain", "answer": "No response produced.",
        "sources": [], "confidence": None, "score": None, "mode": None}
    res["intent"] = out.get("intent", "document")
    log.info("DONE intent=%s type=%s mode=%s confidence=%s",
             res.get("intent"), res.get("type"), res.get("mode"),
             res.get("confidence"))
    return res