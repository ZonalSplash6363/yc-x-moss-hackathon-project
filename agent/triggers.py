"""
Mid-call trigger checks — Zone A.

Two kinds of triggers decide whether the agent should break from the fixed
script and run a Moss retrieval + follow-up question:

1. Deterministic wrong-answer check — compares a RECALL_CHECK answer
   (orientation questions like day-of-week/season, or a personal-recall
   question) against a known correct/expected value.
2. Semantic symptom-flag check — scans free-form OPEN_QA answers for
   language suggesting a flagged symptom, worth a clarifying follow-up.
   Implemented as a lightweight keyword match against a small Parkinson's
   symptom vocabulary (word-boundary matching, not substring), with an
   embedding-similarity upgrade path noted below for when that's worth the
   extra latency/dependency.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Union

# Category -> surface forms. Keep this small and hand-curated for the demo;
# swap/extend once real call transcripts show what patients actually say.
SYMPTOM_VOCABULARY: dict[str, list[str]] = {
    "tremor": ["tremor", "tremors", "shaking", "shake", "shakes", "trembling", "tremble"],
    "stiffness": ["stiff", "stiffness", "rigid", "rigidity"],
    "balance": ["balance", "unsteady", "fell", "falling", "falls", "dizzy", "dizziness"],
    "freezing": ["freezing", "froze", "frozen", "stuck", "locked up"],
    "fatigue": ["tired", "fatigue", "fatigued", "exhausted", "no energy", "low energy", "worn out"],
    "medication issues": [
        "medication", "medications", "meds", "pills", "dose", "dosage",
        "side effect", "side effects", "forgot to take", "missed a dose", "ran out of",
    ],
}

_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Word-boundary regex for a keyword/short phrase (avoids matching
    'stiff' inside an unrelated word, unlike plain substring search)."""
    if phrase not in _WORD_RE_CACHE:
        _WORD_RE_CACHE[phrase] = re.compile(r"\b" + re.escape(phrase) + r"\b")
    return _WORD_RE_CACHE[phrase]


@dataclass
class TriggerResult:
    fired: bool
    reason: Optional[str] = None
    query_text: Optional[str] = None  # text to hand to moss_client.query_patient_history


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


# Accepted alternatives for answers that have more than one correct wording.
# Without these a patient saying "autumn" when we expect "fall" is marked
# wrong and gets an unnecessary follow-up — a false alarm that wastes the
# patient's time and pollutes the flags a clinician sees.
_ANSWER_SYNONYMS: dict[str, set[str]] = {
    "fall": {"fall", "autumn"},
    "autumn": {"fall", "autumn"},
}

# Negations: "I had toast, NOT eggs" contains "eggs", so a plain substring
# check would score it correct. If the expected answer only appears negated,
# treat it as a miss.
_NEGATORS = ("not", "no", "never", "didnt", "didn t", "dont", "don t", "wasnt", "isnt")


def _mentions_expected(answer: str, expected: str) -> bool:
    """Is `expected` (or an accepted synonym) genuinely present, not negated?"""
    candidates = set(_ANSWER_SYNONYMS.get(expected, {expected}))
    for cand in candidates:
        match = _phrase_pattern(cand).search(answer)
        if not match:
            continue
        # Look at the few words immediately before the match for a negator.
        preceding = answer[: match.start()].split()[-3:]
        if any(tok in _NEGATORS for tok in preceding):
            continue  # present, but negated — keep looking
        return True
    return False


def check_wrong_answer(
    answer_text: str,
    expected_answer: Optional[Union[str, Iterable[str]]],
) -> TriggerResult:
    """Deterministic check: does this answer contradict what's expected?

    `expected_answer` may be one value or several acceptable ones (e.g. both
    seasons during the weeks when the meteorological and astronomical ones
    disagree). Synonyms are accepted and negated mentions rejected, so
    ordinary phrasing ("autumn" for "fall", "not eggs" for "eggs") isn't
    mis-scored.
    """
    if expected_answer is None:
        return TriggerResult(fired=False)

    accepted = [expected_answer] if isinstance(expected_answer, str) else list(expected_answer)
    if not accepted:
        return TriggerResult(fired=False)

    answer = _normalize(answer_text)
    if any(_mentions_expected(answer, _normalize(exp)) for exp in accepted):
        return TriggerResult(fired=False)

    expected_str = accepted[0] if len(accepted) == 1 else " or ".join(sorted(accepted))
    return TriggerResult(
        fired=True,
        reason=f"answer did not match expected value: {expected_str!r}",
        query_text=answer_text,
    )


def _word_was_said(answer: str, word: str) -> bool:
    """Did the patient say this word, allowing for speech-to-text slips?

    Exact matching punished the transcriber, not the patient: on a live call
    "book and garden" came back as "Booking garden", scoring 1 of 3 instead
    of 2 and flagging a memory problem that didn't happen. A recalled word is
    accepted when a spoken token matches it exactly, starts with it (booking
    -> book), or is near-identical (penny -> penney).
    """
    if _phrase_pattern(word).search(answer):
        return True
    for token in answer.split():
        if len(word) >= 4 and (token.startswith(word) or word.startswith(token) and len(token) >= 4):
            return True
        if difflib.SequenceMatcher(None, token, word).ratio() >= 0.85:
            return True
    return False


def recalled_words(answer_text: str, words: Iterable[str]) -> list[str]:
    """Which of `words` this one answer brought back."""
    answer = _normalize(answer_text)
    return [w for w in words if _word_was_said(answer, _normalize(w))]


def check_word_recall(
    answer_text: str,
    words: Iterable[str],
    min_required: int = 2,
    already_recalled: Iterable[str] = (),
) -> TriggerResult:
    """Delayed recall: how many of the words said earlier come back?

    This has a correct answer the agent actually knows, because it chose the
    words — unlike "what did you have for breakfast?", which it replaced.

    Fires when fewer than `min_required` are recalled. One miss out of three
    is common in healthy older adults, so flagging every miss would be noise;
    the count is reported either way for tracking over time.

    `already_recalled` carries the words produced on earlier attempts at this
    same question, because the agent asks again after a miss. Scoring each
    attempt alone recorded "recalled 0 of 3" for a patient who had in fact
    produced two of the three across the exchange: the final attempt — cued,
    partial, and the weakest of the three — became the whole record.
    """
    words = list(words)
    found = set(already_recalled) | set(recalled_words(answer_text, words))
    recalled = [w for w in words if w in found]  # keep the caller's order
    summary = f"recalled {len(recalled)} of {len(words)} words"
    if len(recalled) >= min_required:
        return TriggerResult(fired=False, reason=summary)
    missed = [w for w in words if w not in recalled]
    return TriggerResult(
        fired=True,
        reason=f"{summary} (missed: {', '.join(missed)})",
        query_text=answer_text,
    )


# Words describing a symptom as over. Only consulted in the two tokens right
# after a match, and only when no negator shares that window, so "the tremor
# is not gone" still flags.
_RESOLVED = ("gone", "resolved", "stopped", "cleared", "disappeared")


def _phrase_token_spans(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    """Every [start, end) token span where `phrase` occurs."""
    want = phrase.split()
    n = len(want)
    return [(i, i + n) for i in range(len(tokens) - n + 1) if tokens[i:i + n] == want]


def _mention_is_denied(tokens: list[str], start: int, end: int) -> bool:
    """Does the patient deny this symptom, or report it as over?

    Bare keyword matching flagged every denial: "I've not noticed any
    tremor", "no stiffness" and "the tremor is gone" all fired. A patient
    reporting they were *better* was recorded as symptomatic, and the agent
    then asked an LLM follow-up probing a symptom they had just ruled out.
    """
    if any(t in _NEGATORS for t in tokens[max(0, start - 3):start]):
        return True
    after = tokens[end:end + 2]
    return any(t in _RESOLVED for t in after) and not any(t in _NEGATORS for t in after)


def check_symptom_flag(answer_text: str) -> TriggerResult:
    """Semantic check: does this free-form answer report a flagged symptom?

    Lightweight keyword match (word-boundary, case-insensitive) against
    SYMPTOM_VOCABULARY, skipping mentions the patient denies or describes as
    resolved. TODO(agent): upgrade path is an embedding-similarity check
    against a small reference set of symptom sentences instead of exact
    keywords — worth it once false negatives on paraphrased symptoms (e.g.
    "my hands won't stop moving" for tremor) start to matter; keyword
    matching is intentionally the cheap/fast first pass so this never adds
    retrieval-blocking latency mid-call.

    Denial is judged per category, so "no tremor, but the stiffness is bad"
    still flags stiffness. Nothing is lost when a mention is skipped: every
    answer goes to Moss whether or not a trigger fires.
    """
    tokens = _normalize(answer_text).split()
    hits: list[str] = []
    for category, phrases in SYMPTOM_VOCABULARY.items():
        for phrase in phrases:
            spans = _phrase_token_spans(tokens, _normalize(phrase))
            if any(not _mention_is_denied(tokens, s, e) for s, e in spans):
                hits.append(f"{category}:{phrase}")
                break  # one hit per category is enough to flag it

    if hits:
        return TriggerResult(
            fired=True,
            reason=f"symptom keywords matched: {hits}",
            query_text=answer_text,
        )
    return TriggerResult(fired=False)
