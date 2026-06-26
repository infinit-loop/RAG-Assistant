"""Streamlit chat UI for the RAG API.

Talks to the FastAPI backend over HTTP (set API_URL). Keeps the UI a thin
client: all retrieval, guardrails and generation happen server-side.

Two modes:
  - Base corpus : ask the built-in business documents (/ask)
  - My uploads  : upload your own files, get suggested questions, and ask
                  questions answered ONLY from those files (/upload, /suggest,
                  /session/ask). Uploads are session-scoped and persisted.
"""
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")  # only needed if the backend gate is on

# Header sets: JSON requests vs multipart upload (must NOT set Content-Type).
JSON_HEADERS = {"Content-Type": "application/json"}
AUTH_HEADERS = {}
if API_KEY:
    JSON_HEADERS["x-api-key"] = API_KEY
    AUTH_HEADERS["x-api-key"] = API_KEY

TYPE_BADGE = {
    "answer": ("✅", "Grounded answer"),
    "clarify": ("❓", "Needs clarification"),
    "abstain": ("🚫", "No supporting evidence"),
    "blocked": ("🛡️", "Blocked by guardrail"),
}

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 Document Assistant")

# Stable per-browser-session id (used to isolate uploads server-side).
if "session_id" not in st.session_state:
    st.session_state.session_id = "ui-" + uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

SID = st.session_state.session_id


def post_json(path, payload):
    return requests.post(f"{API_URL}{path}", headers=JSON_HEADERS,
                         json=payload, timeout=120).json()


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Backend status")
    try:
        h = requests.get(f"{API_URL}/health", timeout=10).json()
        st.success("Connected")
        st.caption(f"Indexed chunks: {h['indexed_chunks']}")
        st.caption(f"Structured rows: {h['structured_rows']}")
        st.caption(f"LLM: {h.get('llm_provider')} · {h.get('llm_model')}"
                   if h.get("llm_enabled")
                   else "LLM: off (extractive mode)")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach API at {API_URL}")
        st.caption(str(e))

    st.divider()
    mode = st.radio("Question source",
                    ["Base corpus", "My uploads"],
                    help="Base corpus = built-in docs. My uploads = your files.")
    st.caption("Tip: for analytics just ask in chat, e.g. "
               "\"which branch has the highest sales?\"")

# ---------------- upload panel (only in uploads mode) ----------------
if mode == "My uploads":
    with st.expander("📤 Upload documents (.txt, .md, .csv, .pdf)", expanded=True):
        up = st.file_uploader("Add a file to this session",
                              type=["txt", "md", "csv", "pdf"],
                              accept_multiple_files=False)
        col1, col2 = st.columns(2)
        if col1.button("Upload & index", disabled=up is None):
            try:
                files = {"file": (up.name, up.getvalue())}
                r = requests.post(f"{API_URL}/upload", headers=AUTH_HEADERS,
                                  data={"session_id": SID}, files=files, timeout=120)
                if r.status_code == 200:
                    info = r.json()
                    if info.get("warning"):
                        st.warning(info["warning"])
                    else:
                        st.success(f"Indexed '{up.name}' "
                                   f"({info['indexed_chunks']} chunks). "
                                   f"Files: {', '.join(info['files'])}")
                    # fetch suggested questions
                    s = requests.get(f"{API_URL}/suggest", headers=AUTH_HEADERS,
                                     params={"session_id": SID}, timeout=120).json()
                    st.session_state.suggestions = s.get("questions", [])
                else:
                    st.error(r.json().get("error", r.text))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
        if col2.button("Refresh suggestions"):
            try:
                s = requests.get(f"{API_URL}/suggest", headers=AUTH_HEADERS,
                                 params={"session_id": SID}, timeout=120).json()
                st.session_state.suggestions = s.get("questions", [])
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    if st.session_state.suggestions:
        st.caption("💡 Suggested questions (click to ask):")
        for i, q in enumerate(st.session_state.suggestions):
            if st.button(q, key=f"sugg-{i}"):
                st.session_state.pending_q = q

# ---------------- chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])


def format_meta(emoji, label, r):
    """Build the small grey footnote under an answer: status badges on the
    first line, then each cited source on its own line for readability."""
    badges = [f"{emoji} {label}"]
    if r.get("intent"):
        badges.append(f"route: {r['intent']}")
    if r.get("confidence"):
        badges.append(f"confidence: {r['confidence']}")
    if r.get("mode"):
        badges.append(f"mode: {r['mode']}")
    lines = [" · ".join(badges)]
    if r.get("query"):
        lines.append(f"pandas: `{r['query']}`")
    if r.get("sources"):
        lines.append("**Sources**")
        for s in r["sources"]:
            lines.append(f"- {s['source']} · score {s['score']}")
    return "  \n".join(lines)


def handle_question(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    path = "/agent/ask"
    payload = {"question": prompt,
               "source": "uploads" if mode == "My uploads" else "base",
               "session_id": SID if mode == "My uploads" else None}
    with st.chat_message("assistant"):
        try:
            r = post_json(path, payload)
        except Exception as e:  # noqa: BLE001
            err = f"Error contacting API: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            return
        emoji, label = TYPE_BADGE.get(r["type"], ("", r["type"]))
        st.markdown(r["answer"])
        meta = format_meta(emoji, label, r)
        st.caption(meta)
        st.session_state.messages.append(
            {"role": "assistant", "content": r["answer"], "meta": meta})


# a clicked suggestion
if st.session_state.pending_q:
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    handle_question(q)

# free-form input
placeholder = ("Ask about your uploaded documents..." if mode == "My uploads"
               else "Ask about SOPs, policies, KPIs, returns...")
if prompt := st.chat_input(placeholder):
    handle_question(prompt)