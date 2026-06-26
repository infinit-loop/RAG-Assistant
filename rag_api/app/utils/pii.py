"""Pluggable PII masking.

Two engines, selected by the PII_ENGINE setting:

  - "regex"    : the original fast/offline structured-pattern masker. No extra
                 dependencies. Catches PIN/SSN/card-style patterns only.
  - "presidio" : Microsoft Presidio with a spaCy NER model. Catches the regex
                 patterns AND unstructured PII (PERSON, EMAIL_ADDRESS,
                 PHONE_NUMBER, LOCATION, IBAN, etc.). The original regex
                 patterns are registered as Presidio pattern-recognizers so
                 nothing the regex engine caught is lost.

Public entry point is `mask()`. The engine is built once and cached. If
Presidio is selected but its dependencies/model are missing, it logs a warning
and falls back to regex so the service still starts.
"""
from __future__ import annotations
import re
import logging

from app.common.config import get_settings
from app.common.constants import SENSITIVE_PATTERNS

logger = logging.getLogger("pii")


# ---------------- regex engine (original behaviour) ----------------
def _mask_regex(text: str) -> str:
    for pat, repl in SENSITIVE_PATTERNS:
        text = pat.sub(repl, text)
    return text


# ---------------- presidio engine ----------------
class _PresidioMasker:
    """Lazily builds a Presidio analyzer+anonymizer using spaCy NER, augmented
    with the project's existing regex patterns as recognizers."""

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        cfg = get_settings()

        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": cfg.PII_SPACY_MODEL}],
        }).create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                                       supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()

        # Register the original regex patterns as Presidio recognizers so the
        # custom PIN rule (not a Presidio default) is still enforced.
        pin_rec = PatternRecognizer(
            supported_entity="CUSTOM_PIN",
            patterns=[Pattern(name="pin", regex=r"\bPIN\s*(is)?\s*\d{3,6}\b",
                              score=0.9)],
        )
        self.analyzer.registry.add_recognizer(pin_rec)

        # Optional explicit entity allow-list from config.
        ents = [e.strip() for e in cfg.PII_ENTITIES.split(",") if e.strip()]
        self.entities = ents or None  # None => Presidio analyzes all supported

    def mask(self, text: str) -> str:
        if not text:
            return text
        # Always apply the deterministic regex first: Presidio's NER-based
        # US_SSN/card detection is context-dependent and can miss a bare number
        # that the regex reliably catches. Regex placeholders are inert text
        # that Presidio simply ignores, so the two compose safely.
        text = _mask_regex(text)
        results = self.analyzer.analyze(text=text, language="en",
                                        entities=self.entities)
        if not results:
            return text
        # Replace each detected span with a typed placeholder, e.g. <PERSON>.
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


# ---------------- engine selection (cached) ----------------
_engine = None
_engine_kind = None


def _get_engine():
    global _engine, _engine_kind
    if _engine_kind is not None:
        return _engine, _engine_kind

    cfg = get_settings()
    kind = (cfg.PII_ENGINE or "regex").lower()
    if kind == "presidio":
        try:
            _engine = _PresidioMasker()
            _engine_kind = "presidio"
        except BaseException as e:  # noqa: BLE001
            # Note: a missing spaCy model makes Presidio's provider raise
            # SystemExit (a BaseException), so we catch broadly and fall back
            # to regex rather than letting the service crash on startup.
            logger.warning("Presidio unavailable (%s); falling back to regex PII "
                           "masking.", e)
            _engine = None
            _engine_kind = "regex"
    else:
        _engine = None
        _engine_kind = "regex"
    return _engine, _engine_kind


def mask(text: str) -> str:
    """Mask PII in `text` using the configured engine."""
    engine, kind = _get_engine()
    if kind == "presidio" and engine is not None:
        try:
            return engine.mask(text)
        except Exception:  # never let masking failure leak raw text downstream
            return _mask_regex(text)
    return _mask_regex(text)


def active_engine() -> str:
    """Return the engine actually in use ('regex' or 'presidio')."""
    return _get_engine()[1]