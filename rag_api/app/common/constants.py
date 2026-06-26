"""Shared constants, regex patterns and exception types."""
import re

# ----- sensitive-info masking -----
SENSITIVE_PATTERNS = [
    (re.compile(r"\bPIN\s*(is)?\s*\d{3,6}\b", re.I), "[REDACTED-PIN]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED-CARD]"),
]
RESTRICTED_TAG = re.compile(r"\bRESTRICTED\b", re.I)

# ----- prompt-injection markers -----
INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|the above) (instructions|prompts?)", re.I),
    re.compile(r"disregard .* (instructions|rules|guardrails)", re.I),
    re.compile(r"reveal .* (system prompt|hidden|confidential|restricted)", re.I),
    re.compile(r"you are now .* (dan|jailbreak|unrestricted)", re.I),
    re.compile(r"print .* (api key|password|pin|secret)", re.I),
]

VAGUE_TERMS = {"bad", "wrong", "broken", "good", "issue", "problem",
               "it", "this", "that"}
GENERIC_NOUNS = {"report", "data", "number", "numbers", "thing", "stuff",
                 "file", "document", "system"}

# response "types"
TYPE_ANSWER = "answer"
TYPE_CLARIFY = "clarify"
TYPE_ABSTAIN = "abstain"
TYPE_BLOCKED = "blocked"


class AppError(Exception):
    """Base error carrying an HTTP status code."""
    status_code = 500

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code:
            self.status_code = status_code


class UnauthorizedError(AppError):
    status_code = 401
