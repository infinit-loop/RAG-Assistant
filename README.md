# Document Assistant

A small but complete document assistant that answers questions **grounded in the
documents you give it** — and is honest when it can't. It does plain document
Q&A, runs simple analytics over spreadsheets, and lets you upload your own files
to ask about them in an isolated session. Under the hood it's a layered FastAPI
service with a LangGraph agent for routing, FAISS for retrieval, and a
provider-agnostic LLM client (OpenRouter, OpenAI, Cloudflare Workers AI, or
Google Gemini — pick one with a single environment variable).

If no LLM key is configured it still works: it falls back to **extractive**
answers (returning the relevant source snippets) so the grounding promise holds
even offline.

---

## Setup

### 1. Create your `.env`
Copy the example file and fill in your values (at minimum, the key for the LLM
provider you want to use — see "Choosing an LLM" below):
```bash
cp .env.example .env
```
`docker compose` reads this `.env` automatically, and it's gitignored so your
keys never get committed. You can skip the keys entirely to run in extractive
mode (offline, no LLM).

### 2. With Docker (recommended)
```bash
docker compose up --build
```
- API + Swagger UI → http://localhost:8000/docs
- Streamlit chat UI → http://localhost:8501

### 3. Locally (alternative to Docker)
```bash
pip install -r requirements.txt
# The base corpus lives in docs/base_corpus; docs/examples holds sample files
# for testing the upload flow and is intentionally NOT indexed.
DOCS_DIR=./docs/base_corpus uvicorn app.main:app --reload
# in another terminal, for the UI:
streamlit run ui/streamlit_app.py
```

### Choosing an LLM
The assistant talks to any OpenAI-compatible endpoint, so switching providers is
one line in `.env`:

```env
LLM_PROVIDER=cloudflare        # openrouter | openai | cloudflare | gemini

OPENROUTER_API_KEY=...         OPENROUTER_MODEL=google/gemma-4-31b-it:free
OPENAI_API_KEY=...             OPENAI_MODEL=gpt-4o-mini
CLOUDFLARE_API_TOKEN=...       CLOUDFLARE_ACCOUNT_ID=...   CLOUDFLARE_MODEL=@cf/qwen/qwen2.5-coder-32b-instruct
GEMINI_API_KEY=...             GEMINI_MODEL=gemini-2.0-flash
```
Only the selected provider's key is needed; you can leave the others filled in
and just flip `LLM_PROVIDER`. Leave every key empty to run in extractive mode.
Other useful knobs: `LOG_LEVEL` (INFO/DEBUG), `API_KEY` (turns on the access
gate), and the retrieval thresholds (`MIN_RETRIEVAL_SCORE`, `CONFIDENT_SCORE`).
See `.env.example` for the full list.

---

## Architecture

The codebase is layered so each piece has one job and is easy to swap:

```
rag_api/
  app/
    routes/        HTTP endpoints -> controllers
    controllers/   thin request handlers, map engine output to DTOs
    middleware/    API-key access gate + error handling
    dto/           pydantic request/response schemas (validation)
    agent/         LangGraph agent: routes each message to the right handler
    services/      shared engine, session manager, LLM client
    core/          RAG engine, FAISS store, text-to-pandas analytics
    utils/         stateless helpers (chunking, PII masking)
    common/        config, constants, logging
    main.py        app factory & startup wiring
  docs/
    base_corpus/   the documents indexed at startup
    examples/      sample files to test the upload flow (not indexed)
  eval/            offline evaluation harness + gold set + rubric
  ui/              Streamlit chat client
```

Every request flows through the same layers:

```
HTTP -> route -> controller -> service -> core engine (FAISS / pandas) -> DTO
```

The interesting part is the agent. Instead of treating every message the same,
it classifies the message first and sends it down the right path:

```
            ┌──────────┐
  message → │  guard   │  prompt-injection check (always first)
            └────┬─────┘
                 ▼
            ┌──────────┐
            │ classify │  LLM classifier, with a keyword fallback
            └────┬─────┘
     ┌───────────┼───────────────┬───────────────┐
     ▼           ▼               ▼               ▼
 structured   document        chitchat        offtopic
 (analytics) (RAG answer)   (greeting)     (polite decline)
     │           ▲
     └───────────┘  if there's no table to query, fall back to document retrieval
```

- **document** runs grounded RAG against the base corpus, or against a session's
  uploaded files when you're in upload mode.
- **structured** answers analytical questions over a spreadsheet. The LLM writes
  a single pandas expression from the table's schema; **pandas does the actual
  math**. If there's no table available it doesn't dead-end — it falls back to
  document retrieval, which abstains on its own if nothing matches.
- **chitchat** handles greetings (the message is PII-masked before it ever
  reaches the LLM), and **offtopic** politely declines anything outside the
  documents rather than answering from general knowledge.

Retrieval is FAISS over TF-IDF vectors (L2-normalized, inner-product index =
cosine similarity). The base corpus and each upload session get their own index,
so uploaded files are isolated per session and never leak into the base corpus.

### Endpoints
| Method | Path            | Purpose |
|--------|-----------------|---------|
| POST   | `/agent/ask`    | main entry — classifies and routes the message (used by the UI) |
| POST   | `/ask`          | grounded Q&A against the base corpus (also runs the clarification gate) |
| POST   | `/structured`   | preset analytical queries over the base CSV |
| POST   | `/upload`       | add a file to a session-scoped index |
| GET    | `/suggest`      | suggested questions for a session's uploads |
| POST   | `/session/ask`  | ask questions answered only from a session's uploads |
| GET    | `/health`       | index size, row count, active LLM provider/model |

---

## Key design choices & tradeoffs

**Retrieval is intentionally lightweight (TF-IDF + FAISS).** It keeps the service
fully offline, small, and fast — no model download, no GPU. The honest tradeoff
is that TF-IDF matches on words, not meaning, so a question phrased very
differently from the source can score low. The retrieval layer is deliberately
isolated behind `FaissVectorStore._embed`, so moving to a proper embedding model
to capture the **semantic space** (sentence-transformers, or a hosted embedding
API) is a localized change — the rest of the stack doesn't move. For larger or
PDF-heavy corpora, the natural next step is to add a **reranker** (a cross-encoder
that reorders the top-N hits by true relevance); it costs an extra pass, so it
only earns its keep once first-stage recall starts pulling in a lot of
loosely-related chunks.

**The LLM writes code and prose, but never computes the numbers.** For analytics
the model only produces a pandas expression; pandas evaluates it. This is the
single most important reliability decision in the project — it means figures like
totals and averages are computed deterministically and can't be hallucinated. It
also makes answers reproducible and easy to audit (the exact expression is
returned alongside the answer).

**Analytics use pandas rather than SQL/DuckDB.** For the scale this assistant
targets (spreadsheets up to roughly ten thousand rows) in-memory pandas is more
than fast enough and far simpler to operate. The generated expression is parsed
and checked against an allow-list of safe, read-only operations before it's
evaluated in a restricted namespace with no imports and no file or network
access — so a bad or hostile expression is rejected rather than run. If the data
ever grew to millions of rows or needed cross-table joins, a SQL engine like
DuckDB would be the better tool.

**Message routing is a hybrid.** A quick keyword check handles the obvious cases
for free; the LLM classifier handles the ambiguous ones. This keeps common
questions cheap and fast while still being flexible — and the system degrades
gracefully if the classifier is unavailable.

**One environment variable picks the LLM provider.** Because every supported
provider speaks the OpenAI-compatible protocol, the client only varies the base
URL, key, and model. Switching between OpenRouter, OpenAI, Cloudflare, and Gemini
is a config change, not a code change, which makes it easy to dodge a provider's
rate limits or outages.

**Evaluation is deterministic by default.** The eval harness computes its core
metrics with plain string/number checks and no LLM calls, so it's reproducible
and doesn't burn tokens or depend on a provider being up. An LLM-as-judge tier is
available, but it's opt-in.

---

## Assumptions

- The corpus is small-to-medium and fits comfortably in memory; the index is
  built once at startup.
- Spreadsheets for analytics are on the order of a few thousand rows.
- Documents are text-based and in English. PDFs are expected to have a real text
  layer — scanned/image-only PDFs aren't read (there's no OCR).
- Sessions are identified by an opaque id from the client; isolation is by id,
  not by authenticated user.
- Generated answers and analytics need a working LLM key; without one the system
  runs in extractive mode.

---

## Guardrails implemented

Each guardrail below lists why it exists, the risk it addresses, and where it
falls short.

**1. Abstain on weak evidence (minimum retrieval threshold + coverage check).**
If the best match scores below a threshold, or the top chunk shares no meaningful
word with the question, the assistant says it couldn't find support instead of
guessing.
*Risk:* hallucinated answers to out-of-corpus questions.
*Limitation:* the score is lexical, so the threshold is corpus-dependent — short
or unusually-worded questions can under-score even when the answer exists.

**2. Confidence / uncertainty handling.** Answers are labelled high or low
confidence based on retrieval strength, and low-confidence answers carry a note
about what might be missing.
*Risk:* users over-trusting a weakly-supported answer.
*Limitation:* "confidence" is a retrieval-score bucket, not a calibrated
probability of correctness.

**3. Prompt-injection detection.** A guard step checks each message for known
override/exfiltration patterns ("ignore previous instructions", "reveal the
system prompt", …) and blocks them; the system prompt also tells the model to
treat retrieved text as data, not instructions.
*Risk:* jailbreaks, instruction override, data exfiltration.
*Limitation:* pattern matching catches known phrasings and is bypassable by
paraphrase or obfuscation — it's a first line of defense, not a complete one.

**4. Sensitive-information masking.** PII is redacted from retrieved snippets
before generation, from anything the model echoes back, and from chitchat before
it's sent to the provider. A regex engine handles structured secrets (PIN, SSN,
card numbers) out of the box, with optional Presidio + spaCy NER for names,
emails, phones, and locations.
*Risk:* leaking secrets or personal data into answers or to a third-party LLM.
*Limitation:* the default regex engine only catches structured patterns; broader
entity coverage needs the NER engine enabled.

**5. Access restriction.** Lines tagged `RESTRICTED` are dropped at load time so
they're never indexed or retrievable, and an optional API-key gate protects the
endpoints.
*Risk:* exposure of restricted content or unauthenticated use.
*Limitation:* restriction is line-level and all-or-nothing, and the gate is a
single shared key with no per-user roles.

**6. Grounded, cited answers.** The model is instructed to answer only from the
provided context and to say so when it can't; every answer returns its sources
with scores, and extractive answers quote the snippets with per-source
attribution.
*Risk:* unverifiable claims.
*Limitation:* citations are at the chunk/source level, not sentence-level, and in
generated mode grounding is enforced by instruction rather than hard-verified.

---

## Evaluation approach

There's a small offline harness under `eval/` that runs a hand-curated gold set
(`eval/golden_truth.yaml`) through the real engine and reports answer quality.

It's deterministic by default — no LLM tokens — so it's reproducible and survives
provider outages. For each question it checks:

- **Retrieval relevance** — did the expected source document show up in the
  citations (recall@k)?
- **Citation coverage** — does every answer carry at least one source?
- **Groundedness** — how much of the answer's wording actually appears in the
  retrieved context (a low overlap flags a possible hallucination), plus a
  numeric check that any figures in the answer also appear in the source/data.
- **Correctness** — are the expected facts present in the answer?
- **Behaviour** — do the guardrail cases do the right thing (off-topic abstains,
  injection is blocked)?

It also runs a couple of direct safety probes (PII masking redacts an SSN/card;
the injection detector flags an override attempt), and writes a CSV for human
review scored against `eval/RUBRIC.md` (relevance, groundedness, completeness,
correctness, safety on a 1–5 scale). An optional LLM-as-judge pass adds a second
opinion with `--judge`.

```bash
docker compose exec api python -m eval.run_eval
docker compose exec api python -m eval.run_eval --judge
```

---

## Limitations

- **Lexical retrieval.** TF-IDF misses paraphrases and synonyms a semantic model
  would catch.
- **Clarification flow is dormant in the chat UI.** The ambiguity → clarify path
  exists and runs on the direct `/ask` endpoint, but the agent skips it (it has
  already decided the message is a document question), so it won't trigger in the
  Streamlit flow.
- **The base corpus is text-only right now.** The sample spreadsheet was moved to
  `docs/examples/`, so analytics in base-corpus mode will abstain — upload the CSV
  to ask analytical questions about it.
- **PDFs are only read on upload**, not from the base corpus, and only if they
  have a real text layer (no OCR).
- **Default PII masking is regex-based** and won't catch free-form names without
  the NER engine enabled.
- **Access control is simulated** — a single shared key, no per-user roles.
- **Free-tier LLMs are flaky.** Daily caps and retired model slugs happen; the
  provider switch and a small retry help, but they don't eliminate it.
- **The gold set is small** — indicative of quality, not a statistical benchmark.

---

## Future improvements

- **Semantic embeddings.** Swap TF-IDF for a sentence-transformer or hosted
  embedding model so retrieval works on meaning, not just shared words.
- **Hybrid retrieval.** Combine a keyword signal like BM25 with the dense
  (embedding) search and fuse the two rankings — keyword matching nails exact
  terms, codes, and rare names, while embeddings handle paraphrases, so together
  they recall more than either alone.
- **A reranker for large / PDF-heavy corpora.** Once many PDFs are indexed and
  first-stage recall gets noisy, a cross-encoder reranker over the top-N would
  sharpen precision a lot.
- **Multimodal support.** Use a multimodal embedding model (such as Nomic Embed
  or a CLIP-style model) so images, charts, diagrams, and scanned pages become
  searchable alongside text — letting the assistant answer from documents whose
  meaning lives in pictures, not just in the text layer.
- **An answer-review layer.** Add a verification step where a second LLM (ideally
  a different model or provider) reviews the drafted answer against the retrieved
  evidence and either approves it or sends it back — a cheap way to catch
  ungrounded claims before they reach the user.
- **A knowledge graph for numerous PDFs.** Building a graph of entities and
  relationships extracted from the documents would enable entity-centric and
  multi-hop lookups that plain vector search struggles with at scale.
- **Stronger guardrails.** A policy framework like NeMo Guardrails for richer
  input/output rules, and entity detection with GLiNER or spaCy's large model
  (`en_core_web_lg`) for more thorough PII and entity coverage than regex.
- **Provider fallback** — try the selected provider and automatically fall
  through to the next configured one on a quota or availability error.
- **Observability** — wire in Langfuse to track token usage, cost, and latency
  per request, and add per-endpoint monitoring (request rates, error rates,
  response times) so we can see how each API is behaving in production.
- **Secrets management.** Load API keys from a secrets manager (Vault, AWS/GCP
  Secrets Manager, Doppler) injected at runtime instead of a plaintext `.env`,
  keep them encrypted at rest and out of version control, redact them from logs,
  and rotate them regularly. Keys already stay server-side and travel over TLS;
  this hardens how they're stored and handled.
- **Streaming responses.** Stream tokens back as the LLM generates them (SSE /
  chunked responses) so the UI shows the answer as it's written instead of
  waiting for the full reply — a big perceived-latency win on longer answers.
- **Operational polish** — response caching and sentence-level citations.
