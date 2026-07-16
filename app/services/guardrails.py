"""Input guardrails applied to user questions before they reach retrieval/generation:

1. Prompt-injection defense — regex/keyword pattern matching against known jailbreak
   and instruction-override phrasing. This is a first line of defense, not a complete
   solution: production systems layer this with an ML classifier and/or an LLM-based
   self-critique step. Cheap and fast enough to run on every request though, so there's
   no reason not to have it.

2. PII redaction — Microsoft Presidio (spaCy NER + pattern recognizers) detects and
   redacts things like emails, phone numbers, credit cards, and names before the raw
   question is sent to embeddings / the LLM (both third-party APIs).

Both degrade gracefully: if Presidio/spaCy isn't installed or fails to load, PII
redaction is skipped (logged once) rather than breaking the request.
"""

import logging
import re

logger = logging.getLogger(__name__)

_analyzer = None
_anonymizer = None
_presidio_unavailable = False


class PromptInjectionDetected(Exception):
    pass


# --- Prompt injection defense -------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above) instructions",
    r"disregard (all )?(the )?(previous|prior|above)",
    r"reveal (your |the )?system prompt",
    r"forget (all )?(your )?(previous )?instructions",
    r"new instructions\s*:",
    r"you are now (a|an)\s+\w+",  # e.g. "you are now DAN"
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"pretend (you have no|to have no) (restrictions|rules|guidelines)",
    r"act as if you (have no|had no) (restrictions|filters|rules)",
]
_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_prompt_injection(text: str) -> None:
    """Raise PromptInjectionDetected if the text matches a known injection pattern."""
    match = _INJECTION_REGEX.search(text)
    if match:
        logger.warning("Blocked possible prompt injection: matched %r", match.group(0))
        raise PromptInjectionDetected(
            f"Input blocked: matches a known prompt-injection pattern ({match.group(0)!r})"
        )


# --- PII redaction (Presidio) -------------------------------------------------------


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        _analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["en"]
        )
    return _analyzer


def _get_anonymizer():
    global _anonymizer
    if _anonymizer is None:
        from presidio_anonymizer import AnonymizerEngine

        _anonymizer = AnonymizerEngine()
    return _anonymizer


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Detect and redact PII. Returns (possibly-redacted text, entity types found).
    Falls back to returning the original text unchanged if Presidio/spaCy is
    unavailable — PII redaction is a defense-in-depth layer, not a hard dependency."""
    global _presidio_unavailable
    if _presidio_unavailable:
        return text, []

    try:
        analyzer = _get_analyzer()
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return text, []
        anonymizer = _get_anonymizer()
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        entity_types = sorted({result.entity_type for result in results})
        return anonymized.text, entity_types
    except Exception as exc:
        _presidio_unavailable = True
        logger.warning("PII redaction unavailable (%s); continuing without it", exc)
        return text, []
