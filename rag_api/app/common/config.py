"""Central configuration. Values can be overridden by environment variables
(see docker-compose.yml)."""
import os
from functools import lru_cache


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "grounded-rag-api")
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    # Log verbosity: INFO shows the agent's decisions; DEBUG also shows
    # retrieval scores and raw generated expressions.
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Path to the document corpus (mounted in Docker)
    DOCS_DIR: str = os.getenv("DOCS_DIR", "/app/docs/base_corpus")
    # Where session-uploaded files are persisted (mounted volume)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/app/uploads")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

    # Retrieval / guardrail thresholds
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.08"))
    CONFIDENT_SCORE: float = float(os.getenv("CONFIDENT_SCORE", "0.18"))
    AMBIGUITY_MIN_WORDS: int = int(os.getenv("AMBIGUITY_MIN_WORDS", "4"))

    # Simple API-key gate (access-restriction simulation). Empty = disabled.
    API_KEY: str = os.getenv("API_KEY", "")

    # ----- PII masking -----
    # "regex"    = fast, offline, structured patterns only (default)
    # "presidio" = Presidio + spaCy NER (also catches names, emails, phones,
    #              locations, etc.). Requires presidio + a spaCy model installed.
    PII_ENGINE: str = os.getenv("PII_ENGINE", "regex")
    PII_SPACY_MODEL: str = os.getenv("PII_SPACY_MODEL", "en_core_web_sm")
    # Comma-separated Presidio entity types to redact (empty = Presidio defaults)
    PII_ENTITIES: str = os.getenv("PII_ENTITIES", "")

    # ----- LLM generation (optional) -----
    # Provider is selected by LLM_PROVIDER ("openrouter" or "openai"). Each
    # provider has its own key + default model; the client picks the right
    # base_url/key/model from the choice. If no key resolves, the engine falls
    # back to extractive answers.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")

    # Per-provider keys (only the selected provider's key is needed).
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # Cloudflare Workers AI: needs a token AND the account id (in the URL path).
    CLOUDFLARE_API_TOKEN: str = os.getenv("CLOUDFLARE_API_TOKEN", "")
    CLOUDFLARE_ACCOUNT_ID: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    # Google Gemini via its OpenAI-compatible endpoint (Google AI Studio key).
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Per-provider default models (used unless LLM_MODEL overrides globally).
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    CLOUDFLARE_MODEL: str = os.getenv("CLOUDFLARE_MODEL", "@cf/qwen/qwen2.5-coder-32b-instruct")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Optional global overrides (win over provider defaults when set). Empty =
    # use the selected provider's default. LLM_API_KEY is a generic fallback
    # key for backward compatibility.
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))


@lru_cache
def get_settings() -> "Settings":
    return Settings()