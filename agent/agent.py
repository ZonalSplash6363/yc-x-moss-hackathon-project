"""
LiveKit voice agent — Zone A.

Pipeline: STT (Deepgram, word-level timestamps) -> trigger check -> Moss
retrieval (only if triggered) -> LLM -> TTS (edge-tts today; Kokoro-82M
stubbed, see _synthesize_kokoro).

Uses low-level LiveKit Agents primitives (JobContext + manual STT stream +
manual AudioSource publish), not the high-level VoicePipelineAgent wrapper,
because a trigger firing needs to inject retrieved Moss context into the LLM
call before it runs — the high-level wrapper doesn't expose that seam.

Run `python -m agent.agent --dry-run` to exercise the turn logic with no
external services, or `python -m agent.agent start` to run the live worker
(needs LIVEKIT_*/DEEPGRAM_API_KEY in .env). The live path has been run
against LiveKit Cloud end-to-end: STT with word timestamps, trigger checks,
Moss retrieval, the dead-air bridging phrase, TTS playback, and clean
shutdown with audio + transcript persisted.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib
import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from agent.call_script import (
    CallScript,
    CallState,
    CaptureMode,
    orientation_expected_answer,
    patient_now,
    recall_words_for_call,
)
from agent.triggers import (
    TriggerResult,
    check_symptom_flag,
    check_word_recall,
    check_wrong_answer,
    recalled_words,
)
from db.contracts import MossQARecord
from agent.moss_worker import MossBridge
from db.moss_client import push_patient_session

logger = logging.getLogger("agent")

# --- env / config ----------------------------------------------------------

LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Kokoro tried first per spec; defaults to edge-tts because Kokoro isn't
# wired yet (see _synthesize_kokoro for why) — flip this once it is.
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "edge-tts")  # "kokoro" | "edge-tts"
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")

SAMPLE_RATE = 16000
NUM_CHANNELS = 1

CALLS_DIR = Path(os.environ.get("CALLS_DIR", "calls"))
DB_PATH = os.environ.get("DB_PATH", "data/app.db")

# No dead air: if retrieval + LLM follow-up generation takes longer than
# this, speak the bridging phrase first, then the real follow-up once ready.
# The LLM alone takes ~2s median (0.95-3.5s measured), so the bridge fires
# on most follow-ups. It must therefore be an acknowledgement, not a question:
# the old "Can you tell me a bit more about that?" was immediately followed
# by the real follow-up question, so patients heard two questions in a row.
FALLBACK_TIMEOUT_S = float(os.environ.get("FALLBACK_TIMEOUT_S", "1.2"))
FALLBACK_PHRASE = "Okay, thank you for telling me."
# Used only when no follow-up can be generated at all (no key / LLM error).
TEMPLATED_FOLLOWUP = "Can you say a little more about that? I want to make sure I understand."
# Longest we'll wait on a Moss lookup before asking the follow-up without history.
RETRIEVAL_TIMEOUT_S = float(os.environ.get("RETRIEVAL_TIMEOUT_S", "3"))

# Most follow-ups to ask about any single question before moving on, so a
# repeatedly-tripped trigger can't trap the call on one question.
MAX_FOLLOWUPS_PER_QUESTION = int(os.environ.get("MAX_FOLLOWUPS_PER_QUESTION", "2"))

# How long to wait for a final transcript before moving the call on anyway.
# Essential, not just defensive: the SUSTAINED_PHONATION task asks for "ahhh",
# which is not speech and which STT may never emit a final transcript for, and
# a patient may simply stay silent. Without this the call waits forever.
NO_ANSWER_TIMEOUT_S = float(os.environ.get("NO_ANSWER_TIMEOUT_S", "12"))

# Sustained "ahh" (#3) is recorded from the microphone, not via STT: wait up
# to START_TIMEOUT for the patient to begin, stop once they've been silent for
# END_SILENCE, never record longer than MAX.
PHONATION_START_TIMEOUT_S = float(os.environ.get("PHONATION_START_TIMEOUT_S", "6"))
PHONATION_END_SILENCE_S = float(os.environ.get("PHONATION_END_SILENCE_S", "1.2"))
PHONATION_MAX_S = float(os.environ.get("PHONATION_MAX_S", "20"))

# Turn-taking (#4). Deepgram finalizes on very short pauses, so one spoken
# answer arrives as several transcripts. Keep listening until the patient has
# been quiet this long (no new transcript and no voice on the microphone).
#
# 1.0s cut real patients off mid-sentence: on the first human call, "a bit of
# stiffness in my" and "It's more it's" were both truncated where the patient
# paused to think. Parkinson's speech pauses more, not less, so the window has
# to tolerate a thinking pause. Costs up to a second of reply latency per turn.
ANSWER_SETTLE_S = float(os.environ.get("ANSWER_SETTLE_S", "2.0"))
ANSWER_MAX_S = float(os.environ.get("ANSWER_MAX_S", "30"))
LONG_SPEECH_SETTLE_S = float(os.environ.get("LONG_SPEECH_SETTLE_S", "2.5"))
LONG_SPEECH_MAX_S = float(os.environ.get("LONG_SPEECH_MAX_S", "60"))

# Data-message topic the call page listens on for the live transcript.
TRANSCRIPT_TOPIC = os.environ.get("TRANSCRIPT_TOPIC", "transcript")

# Microphone level (int16 RMS) that counts as voice. Browser microphones with
# echo cancellation and auto-gain put speech well above this, room noise below.
VOICE_RMS_THRESHOLD = float(os.environ.get("VOICE_RMS_THRESHOLD", "400"))

# Who gets called and under which call id is now resolved at runtime — see
# resolve_patient() and start_call_row(). PATIENT_ID pins a specific patient;
# otherwise the most recent signup is called.
FALLBACK_PATIENT_NAME = "Test Patient"  # only used if the patients table is empty


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# --- turn context / handling ------------------------------------------------

@dataclass
class TurnContext:
    patient_id: int
    call_id: int
    # question_id -> known-correct answer text, for RECALL_CHECK prompts
    # whose expected answer isn't computed dynamically (i.e. personal
    # recall, not orientation — see call_script.orientation_expected_answer).
    expected_answers: dict[str, str] = field(default_factory=dict)
    # The patient's timezone, so "what day is it?" is judged where they are.
    timezone: Optional[str] = None
    # question_id -> how many follow-ups we've already asked on it. A fired
    # trigger deliberately does NOT advance the script (so the patient can
    # answer the follow-up), which means a patient who keeps tripping the
    # check would loop on one question forever. See MAX_FOLLOWUPS_PER_QUESTION.
    followup_counts: dict[str, int] = field(default_factory=dict)
    # Every word the patient has recalled across all attempts at the delayed
    # recall question. The agent re-asks after a miss, and scoring each attempt
    # in isolation recorded "recalled 0 of 3" for someone who had produced two
    # of the three words over the course of the exchange.
    recalled_so_far: set[str] = field(default_factory=set)
    # This call's answers so far. Moss holds previous calls; these are handed
    # to the LLM directly so a follow-up can also reference earlier in *this*
    # call without waiting for (or re-loading) the index.
    call_history: list[MossQARecord] = field(default_factory=list)
    # Moss access. Live calls pass a bridge that runs Moss in child processes
    # (see agent/moss_worker.py); the in-process default suits tests.
    moss: MossBridge = field(default_factory=lambda: MossBridge(isolated=False))

    def save_in_background(self, record: MossQARecord) -> None:
        self.moss.save(record)

    async def flush_writes(self, timeout: float = 20.0) -> None:
        """Wait for outstanding Moss writes so the last answers aren't lost."""
        await self.moss.flush(timeout)


@dataclass
class TurnOutcome:
    reply_text: str
    trigger: TriggerResult
    used_fallback: bool = False


def _resolve_expected_answer(script: CallScript, ctx: TurnContext):
    qid = script.current_question_id()
    dynamic = orientation_expected_answer(qid, timezone=ctx.timezone)
    if dynamic is not None:
        return dynamic
    return ctx.expected_answers.get(qid)


async def _retrieve_and_generate_followup(
    script: CallScript, ctx: TurnContext, record: MossQARecord, trigger: TriggerResult
) -> str:
    """The Moss retrieval + LLM follow-up step that runs when a trigger
    fires. The patient's index is preloaded at call start, so the query is a
    local lookup; this call's earlier answers come straight from memory."""
    try:
        past_calls = await asyncio.wait_for(
            ctx.moss.query(
                patient_id=ctx.patient_id,
                query_text=trigger.query_text or record.answer_text,
                question_topic=record.question_topic,
            ),
            timeout=RETRIEVAL_TIMEOUT_S,
        )
    except Exception:
        # A slow or failed lookup must not stall the call: ask the follow-up
        # without history rather than not at all.
        logger.warning("Moss lookup failed or timed out; following up without history", exc_info=True)
        past_calls = []
    this_call = [r for r in ctx.call_history if r is not record]
    return await generate_followup_llm(
        trigger, past_calls, this_call,
        question=script.current_prompt(),
        answer=record.answer_text,
    )


async def speak_with_fallback(
    generate: Awaitable[str], speak: Callable[[str], Awaitable[None]]
) -> str:
    """Race `generate` against FALLBACK_TIMEOUT_S. If it's not ready in
    time, speak the bridging phrase first (so there's never dead air), then
    speak the real result once it lands. Returns the text actually used as
    the final reply.
    """
    task = asyncio.ensure_future(generate)
    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=FALLBACK_TIMEOUT_S)
        return result
    except asyncio.TimeoutError:
        await speak(FALLBACK_PHRASE)
        result = await task  # already in flight, just await completion
        return result


async def handle_turn(
    script: CallScript,
    ctx: TurnContext,
    answer_text: str,
    speak: Callable[[str], Awaitable[None]],
    word_timestamps: Optional[list] = None,
) -> TurnOutcome:
    """Given one captured answer: ingest it into Moss, run the relevant
    trigger check for the current state, and return what the agent should
    say next — either an informed follow-up (trigger fired) or the next
    scripted prompt (trigger didn't fire / call complete).

    `speak` is called directly (not just returned) when the fallback phrase
    is needed mid-generation, so the caller doesn't need to know that
    happened to avoid dead air.
    """
    record = MossQARecord(
        patient_id=ctx.patient_id,
        call_id=ctx.call_id,
        question_id=script.current_question_id(),
        question_topic=script.state.name.lower(),
        answer_text=answer_text,
        timestamp=_now_iso(),
        extra={"word_timestamps": word_timestamps} if word_timestamps else {},
    )
    trigger: TriggerResult = TriggerResult(fired=False)
    if script.state == CallState.RECALL_CHECK:
        expected = _resolve_expected_answer(script, ctx)
        if expected is not None:
            trigger = check_wrong_answer(answer_text, expected)
        else:
            # The word-registration prompt. Repeating words back is attention,
            # not memory — the memory check is DELAYED_RECALL at the end — so
            # note the count but don't interrupt the call over it.
            repeated = check_word_recall(answer_text, script.recall_words, min_required=0)
            record.extra["words_repeated"] = repeated.reason
            logger.info("%s: registration — %s", record.question_id, repeated.reason)
    elif script.state == CallState.DELAYED_RECALL:
        # Credit words from earlier attempts at this same question: the agent
        # re-asks after a miss, and a cued second answer ("Table. Chair.")
        # must not erase what the first one already produced ("Apple ...").
        ctx.recalled_so_far.update(recalled_words(answer_text, script.recall_words))
        trigger = check_word_recall(
            answer_text, script.recall_words, already_recalled=ctx.recalled_so_far
        )
        record.extra["delayed_recall"] = trigger.reason
        logger.info("%s: %s", record.question_id, trigger.reason)
    elif script.state == CallState.OPEN_QA:
        trigger = check_symptom_flag(answer_text)

    # Save in the background, after the checks above have annotated the
    # record. Awaiting the write put a 3-5s Moss call between every answer and
    # the agent's reply (measured 5.8-7.7s of silence per turn). The reply
    # never depends on it: retrieval reads previous calls, and this call's
    # answers are kept in call_history.
    ctx.save_in_background(record)
    ctx.call_history.append(record)

    qid = record.question_id
    already_followed_up = ctx.followup_counts.get(qid, 0)

    if trigger.fired and already_followed_up >= MAX_FOLLOWUPS_PER_QUESTION:
        # Don't ask a third time about the same question — move the call on.
        # Without this a patient who keeps tripping the check (or whose
        # answer keeps mis-transcribing) loops on one question forever and
        # the call never reaches the later tasks.
        logger.info(
            "trigger fired on %s but follow-up limit (%d) reached — advancing",
            qid, MAX_FOLLOWUPS_PER_QUESTION,
        )
    elif trigger.fired:
        ctx.followup_counts[qid] = already_followed_up + 1
        logger.info("trigger fired on %s: %s", qid, trigger.reason)
        generate = _retrieve_and_generate_followup(script, ctx, record, trigger)
        before = asyncio.get_event_loop().time()
        reply = await speak_with_fallback(generate, speak)
        used_fallback = (asyncio.get_event_loop().time() - before) >= FALLBACK_TIMEOUT_S
        return TurnOutcome(reply_text=reply, trigger=trigger, used_fallback=used_fallback)

    next_prompt = script.advance()
    if script.is_complete():
        return TurnOutcome(reply_text="That's everything for today — thanks for checking in!", trigger=trigger)
    return TurnOutcome(reply_text=next_prompt, trigger=trigger)


# --- LLM follow-up generation ------------------------------------------------

_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI  # local import: optional dep until an LLM key exists
        _llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _llm_client


async def generate_followup_llm(
    trigger: TriggerResult,
    context_records: list[MossQARecord],
    this_call: Optional[list[MossQARecord]] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> str:
    """Generate one short, informed follow-up question using retrieved Moss
    context (previous calls) plus this call's earlier answers. Degrades to a
    templated question (no network call) when OPENAI_API_KEY isn't set, so
    the whole loop stays runnable without an LLM key.
    """
    past_lines = [f"- ({r.timestamp[:10]}) {r.answer_text}" for r in context_records]
    now_lines = [f"- {r.answer_text}" for r in (this_call or [])]
    context_str = (
        "From previous calls:\n" + ("\n".join(past_lines) or "(none on file)")
        + "\n\nEarlier in this call:\n" + ("\n".join(now_lines) or "(nothing yet)")
    )

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set — using templated follow-up instead of a real LLM call.")
        return TEMPLATED_FOLLOWUP

    # The model must be told what was actually just asked. Without it, it
    # picked whatever stood out in the retrieved history: after a patient
    # failed the word-recall it asked about their hands, because a tremor was
    # mentioned earlier in the call.
    system = (
        "You are a warm, brief phone check-in assistant for a Parkinson's patient.\n"
        f"You just asked: {question or '(unknown)'}\n"
        f"They answered: {answer or '(unknown)'}\n"
        f"That was flagged because: {trigger.reason}\n\n"
        "Ask ONE short, natural, non-alarming follow-up that stays on THIS "
        "question — do not change the subject to something else in the "
        "history. If they couldn't recall the words, you may offer a gentle "
        "cue (such as what kind of thing one of them is) but never say the "
        "words themselves. Use the history only to make the question more "
        "specific. Do not diagnose or give medical advice. One sentence."
    )
    try:
        client = _get_llm_client()
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=100,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Retrieved context:\n{context_str}\n\nGenerate the follow-up question.",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip() or TEMPLATED_FOLLOWUP
    except Exception:
        # Never let an LLM outage drop the call — fall back to a safe line.
        logger.warning("LLM follow-up generation failed; using templated follow-up", exc_info=True)
        return TEMPLATED_FOLLOWUP


# --- TTS ---------------------------------------------------------------------

async def synthesize_speech(text: str) -> bytes:
    """Returns PCM16 mono audio at SAMPLE_RATE for `text`, via whichever
    provider TTS_PROVIDER selects."""
    if TTS_PROVIDER == "edge-tts":
        return await _synthesize_edge_tts(text)
    if TTS_PROVIDER == "kokoro":
        return await _synthesize_kokoro(text)
    raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER!r}")


async def _synthesize_edge_tts(text: str) -> bytes:
    """edge-tts: free, no API key, works today. Streams mp3, decoded to
    PCM16 via PyAV (pip-installable, no system ffmpeg/espeak-ng needed)."""
    import edge_tts

    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    mp3_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes.extend(chunk["data"])
    # Decode off the event loop: PyAV is synchronous CPU work, and LiveKit
    # warns ("event loop blocked for 1280ms") that blocking here delays
    # outgoing audio and turn handling for the whole call.
    return await asyncio.to_thread(_decode_to_pcm16, bytes(mp3_bytes))


async def _synthesize_kokoro(text: str) -> bytes:
    """Kokoro-82M self-hosted TTS — tried first per spec, NOT wired yet.

    Deferred because getting it running needs, beyond a pip install:
      - the `espeak-ng` phonemizer backend, a system-level package
        (`brew install espeak-ng` on macOS) — exactly the kind of install
        this project asks to confirm before running
      - ~327MB of model weights to download
      - likely `torch` as a dependency (heavier install than edge-tts)

    That's more setup than fits a "get one round trip proven" first step,
    so TTS_PROVIDER defaults to edge-tts for now. To switch: confirm the
    espeak-ng install, `pip install kokoro`, set TTS_PROVIDER=kokoro and
    KOKORO_MODEL_PATH in .env, and implement synthesis here (e.g. via the
    `kokoro` package's KPipeline, matching the return contract above:
    PCM16 mono bytes at SAMPLE_RATE).
    """
    raise NotImplementedError("Kokoro TTS not wired yet — see docstring. Using TTS_PROVIDER=edge-tts instead.")


def _decode_to_pcm16(compressed_audio: bytes) -> bytes:
    """Decode mp3 (or anything PyAV's ffmpeg build reads) to raw PCM16 mono
    at SAMPLE_RATE, for LiveKit AudioFrame publishing."""
    import av

    container = av.open(io.BytesIO(compressed_audio))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()
    stream = container.streams.audio[0]
    for packet in container.demux(stream):
        for frame in packet.decode():
            for resampled in resampler.resample(frame):
                pcm.extend(bytes(resampled.planes[0]))
    return bytes(pcm)


# --- call recording (audio + transcript with word-level timestamps) --------

class CallRecorder:
    """Records a call to <CALLS_DIR>/<call_id>/:

      patient.wav      the patient's microphone, continuous from the moment
                       their audio arrives. This is what the ML pipeline
                       analyses and what calls.audio_path points at.
      agent.wav        what the agent said (TTS), kept for review only.
      transcript.json  the turns, with Deepgram word timestamps on patient
                       turns and each turn's position in patient.wav.

    Kept as two files on purpose. The original recorder wrote only the
    agent's TTS into a single audio.wav and never captured the patient, so
    every acoustic feature and UPDRS score described the synthetic voice.
    """

    def __init__(self, call_id: int, patient_id: int, calls_dir: Path = CALLS_DIR):
        self.call_id = call_id
        self.patient_id = patient_id
        self.call_dir = calls_dir / str(call_id)
        self.call_dir.mkdir(parents=True, exist_ok=True)
        self.audio_path = self.call_dir / "patient.wav"
        self.agent_audio_path = self.call_dir / "agent.wav"
        self.transcript_path = self.call_dir / "transcript.json"
        self.turns: list[dict] = []
        self.patient_samples = 0
        self._closed = False
        # Voice activity on the patient's microphone, used for turn-taking and
        # to time the phonation task. Position (for cutting segments) and
        # wall-clock (for "how long have they been quiet") are kept separately.
        self.last_voice_at_s: Optional[float] = None
        self._last_voice_monotonic: Optional[float] = None
        self.segments: list[dict] = []
        # Triggers that fired during the call, for the report.
        self.flags: list[dict] = []

        self._patient_wav = self._open_wav(self.audio_path)
        self._agent_wav = self._open_wav(self.agent_audio_path)

    @staticmethod
    def _open_wav(path: Path):
        w = wave.open(str(path), "wb")
        w.setnchannels(NUM_CHANNELS)
        w.setsampwidth(2)  # PCM16
        w.setframerate(SAMPLE_RATE)
        return w

    @property
    def patient_seconds(self) -> float:
        """Current length of patient.wav — a position marker into the recording."""
        return self.patient_samples / SAMPLE_RATE

    def write_patient_frame(self, pcm16_bytes: bytes) -> None:
        if self._closed:
            return  # a late frame after hang-up; the file is already closed
        self._patient_wav.writeframes(pcm16_bytes)
        self.patient_samples += len(pcm16_bytes) // 2
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if samples.size and float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) >= VOICE_RMS_THRESHOLD:
            self.last_voice_at_s = self.patient_seconds
            self._last_voice_monotonic = time.monotonic()

    def seconds_since_voice(self) -> Optional[float]:
        """Wall-clock seconds since the patient's microphone last carried
        voice, or None if it never has."""
        if self._last_voice_monotonic is None:
            return None
        return time.monotonic() - self._last_voice_monotonic

    def log_flag(self, question_id: str, reason: str) -> None:
        """Record a trigger that fired, so the after-call report can list it."""
        self.flags.append({
            "question_id": question_id,
            "reason": reason,
            "patient_audio_at_s": round(self.patient_seconds, 3),
        })

    def resolve_flags(self, question_id: str, note: str) -> int:
        """Withdraw this question's flags after a later answer disproved them.

        Marked rather than deleted, so transcript.json still shows what fired
        and why it was withdrawn; only the report skips resolved flags.
        """
        resolved = 0
        for flag in self.flags:
            if flag["question_id"] == question_id and not flag.get("resolved"):
                flag["resolved"] = note
                resolved += 1
        return resolved

    def mark_segment(
        self, name: str, start_s: float, end_s: float,
        text: Optional[str] = None, words: Optional[list] = None,
    ) -> None:
        """Mark a stretch of patient.wav to be cut out for the ML pipeline."""
        self.segments.append({
            "name": name,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "text": text,
            "words": words or [],
        })

    def write_agent_audio(self, pcm16_bytes: bytes) -> None:
        if self._closed:
            return
        self._agent_wav.writeframes(pcm16_bytes)

    def log_patient_turn(self, text: str, word_timestamps: Optional[list] = None) -> None:
        self.turns.append({
            "speaker": "patient",
            "text": text,
            "words": word_timestamps or [],
            "timestamp": _now_iso(),
            "patient_audio_at_s": round(self.patient_seconds, 3),
        })

    def log_agent_turn(self, text: str) -> None:
        self.turns.append({
            "speaker": "agent",
            "text": text,
            "timestamp": _now_iso(),
            "patient_audio_at_s": round(self.patient_seconds, 3),
        })

    def finalize(self, db_path: str = DB_PATH) -> None:
        if self._closed:
            return
        self._closed = True
        self._patient_wav.close()
        self._agent_wav.close()
        self._cut_segments()
        with open(self.transcript_path, "w") as f:
            json.dump(
                {
                    "call_id": self.call_id,
                    "patient_id": self.patient_id,
                    "audio": {"patient": str(self.audio_path), "agent": str(self.agent_audio_path)},
                    "patient_audio_seconds": round(self.patient_seconds, 3),
                    "segments": self.segments,
                    "flags": self.flags,
                    "turns": self.turns,
                },
                f, indent=2,
            )

        _record_call_row(db_path, self.call_id, self.patient_id, str(self.audio_path), str(self.transcript_path))
        logger.info(
            "call %s recorded: %s (%.1fs of patient audio, %d segment(s)), %s",
            self.call_id, self.audio_path, self.patient_seconds, len(self.segments), self.transcript_path,
        )

    def _cut_segments(self) -> None:
        """Write each marked segment to segments/<name>.wav from patient.wav."""
        if not self.segments:
            return
        seg_dir = self.call_dir / "segments"
        seg_dir.mkdir(exist_ok=True)
        seen: dict[str, int] = {}
        with wave.open(str(self.audio_path), "rb") as src:
            total = src.getnframes()
            for seg in self.segments:
                a = max(0, min(total, int(seg["start_s"] * SAMPLE_RATE)))
                b = max(a, min(total, int(seg["end_s"] * SAMPLE_RATE)))
                if b - a < int(0.25 * SAMPLE_RATE):
                    seg["path"], seg["skipped"] = None, "shorter than 0.25s"
                    continue
                n = seen.get(seg["name"], 0)
                seen[seg["name"]] = n + 1
                out = seg_dir / (f"{seg['name']}.wav" if n == 0 else f"{seg['name']}_{n + 1}.wav")
                src.setpos(a)
                w = self._open_wav(out)
                w.writeframes(src.readframes(b - a))
                w.close()
                seg["path"] = str(out)
                seg["duration_s"] = round((b - a) / SAMPLE_RATE, 3)


def _connect_db(db_path: str) -> sqlite3.Connection:
    """Open the call database, applying db/schema.sql (read-only use — this
    never modifies the shared schema file)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_sql = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql.read_text())
    return conn


@dataclass
class PatientRef:
    id: int
    name: str
    timezone: Optional[str] = None  # from the signup form; used to judge orientation answers


def _patient_timezone(conn, patient_id: int) -> Optional[str]:
    """The patient's timezone from their signup, if they have one. The table
    belongs to /signup and may not exist on a database the agent created."""
    try:
        row = conn.execute(
            "SELECT timezone FROM patient_signup_details WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
            (patient_id,),
        ).fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.Error:
        return None


def _requested_patient_id(participant) -> Optional[int]:
    """Which patient the caller asked for, taken from the participant
    metadata on their token ({"patient_id": 3}) — set by the call page's
    picker. None if absent or malformed, in which case resolve_patient falls
    back to its usual order."""
    raw = getattr(participant, "metadata", None)
    if not raw:
        return None
    try:
        value = json.loads(raw).get("patient_id")
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("participant metadata isn't JSON, ignoring it: %r", str(raw)[:100])
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("participant metadata patient_id isn't a number: %r", value)
        return None


def resolve_patient(db_path: str = DB_PATH, patient_id: Optional[int] = None) -> PatientRef:
    """Decide who this call is for.

    An explicit patient_id wins (the browser picks one when starting a call),
    then PATIENT_ID, then the most recently signed-up patient; failing all
    that a test patient is created so the agent still runs on an empty
    database.
    """
    conn = _connect_db(db_path)
    try:
        for candidate, source in ((patient_id, "requested"), (os.environ.get("PATIENT_ID"), "PATIENT_ID")):
            if not candidate:
                continue
            try:
                wanted = int(candidate)
            except (TypeError, ValueError):
                logger.warning("%s=%r is not a patient id; ignoring it", source, candidate)
                continue
            row = conn.execute("SELECT id, name FROM patients WHERE id = ?", (wanted,)).fetchone()
            if row:
                return PatientRef(int(row[0]), row[1], _patient_timezone(conn, int(row[0])))
            logger.warning("%s=%s not found in %s; falling back", source, wanted, db_path)

        row = conn.execute("SELECT id, name FROM patients ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            return PatientRef(int(row[0]), row[1], _patient_timezone(conn, int(row[0])))

        cur = conn.execute("INSERT INTO patients (name) VALUES (?)", (FALLBACK_PATIENT_NAME,))
        conn.commit()
        return PatientRef(int(cur.lastrowid), FALLBACK_PATIENT_NAME, None)
    finally:
        conn.close()


def start_call_row(patient_id: int, db_path: str = DB_PATH) -> int:
    """Create this call's row up front and return its id.

    Allocating a fresh id per call (instead of a hardcoded one) is what keeps
    repeat calls from overwriting each other's audio, transcript and row.
    """
    conn = _connect_db(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO calls (patient_id, timestamp) VALUES (?, ?)",
            (patient_id, _now_iso()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _record_call_row(db_path: str, call_id: int, patient_id: int, audio_path: str, transcript_path: str) -> None:
    """Attach the recorded artefacts to the call row created at call start."""
    conn = _connect_db(db_path)
    try:
        conn.execute(
            "UPDATE calls SET audio_path = ?, transcript_path = ? WHERE id = ?",
            (audio_path, transcript_path, call_id),
        )
        conn.commit()
    finally:
        conn.close()


# --- dry run (no external keys needed) --------------------------------------

async def dry_run() -> None:
    """Exercises the full turn-handling loop — call script, both triggers,
    stub Moss retrieval, fallback bridging, LLM step (templated if
    OPENAI_API_KEY is unset) — with scripted fake patient answers.
    Zero LiveKit/Deepgram/Moss/LLM keys required. This is the "prove the
    pipeline works" check that's actually runnable in this environment.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    patient = resolve_patient()
    patient_id = patient.id
    call_id = start_call_row(patient_id)
    print(f"[dry-run] patient {patient_id} ({patient.name}), timezone {patient.timezone}, call {call_id}\n")
    moss = MossBridge()
    await moss.start()
    await moss.preload(patient_id)

    script = CallScript(recall_words=recall_words_for_call(call_id))
    ctx = TurnContext(
        patient_id=patient_id,
        call_id=call_id,
        timezone=patient.timezone,
        moss=moss,
    )
    recorder = CallRecorder(call_id, patient_id)

    async def speak(text: str) -> None:
        print(f"AGENT: {text}")
        recorder.log_agent_turn(text)

    # Scripted answers per question_id, as a queue: a triggering first answer
    # (wrong recall / symptom mention) followed by a clean follow-up answer,
    # so the follow-up round the agent asks actually gets resolved instead
    # of re-triggering on the same wrong answer forever. Everything else is
    # a single plausible "correct" answer.
    words = script.recall_words
    scripted_answers: dict[str, list[str]] = {
        "recall_check:0": [patient_now(ctx.timezone).strftime("%A")],           # correct — no trigger
        "recall_check:1": ["definitely not a season", "sorry, I meant fall"],   # WRONG, then corrected — triggers once
        "recall_check:2": [", ".join(words)],                                   # repeats the words back
        "sustained_phonation:0": ["ahhhhh"],
        "reading_task:0": ["The old dog stretched slowly..."],
        "open_qa:0": ["I've been okay, but my hands have had a bit of a tremor lately.", "it's mild, comes and goes"],  # triggers symptom flag once
        "open_qa:1": ["Nothing major otherwise."],
        "counting_task:0": ["one two three four five six seven eight nine ten"],
        "delayed_recall:0": [words[0], f"{words[0]} and {words[1]}"],           # 1 of 3 -> flags, then 2 of 3 -> clears
    }
    answer_iters = {qid: iter(answers) for qid, answers in scripted_answers.items()}

    def next_answer(qid: str) -> str:
        try:
            return next(answer_iters.get(qid, iter(())))
        except StopIteration:
            return "that's all, thank you"  # never re-triggers — safe default once a qid's queue is exhausted

    await speak(script.current_prompt())
    max_turns = 30  # safety net against any future scripting bug causing an infinite loop
    for _ in range(max_turns):
        if script.is_complete():
            break
        qid = script.current_question_id()
        answer = next_answer(qid)
        print(f"PATIENT: {answer}")
        recorder.log_patient_turn(answer)

        outcome = await handle_turn(script, ctx, answer, speak=speak)
        await speak(outcome.reply_text)
        if outcome.trigger.fired:
            print(f"  (trigger: {outcome.trigger.reason} | fallback used: {outcome.used_fallback})")
    else:
        print(f"[dry-run] hit max_turns={max_turns} safety cap without completing — check for a trigger loop.")

    await ctx.flush_writes()
    await moss.close()
    recorder.finalize()
    await push_patient_session(ctx.patient_id)  # persist this patient's Moss session for their next call
    print(f"\nTranscript: {recorder.transcript_path}")
    print(f"Patient audio (empty in a dry run — there's no microphone): {recorder.audio_path}")


# --- live LiveKit entrypoint --------------------------------------------------
# NOT YET RUNNABLE without LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET and
# DEEPGRAM_API_KEY in .env — see module docstring. Written against the
# current livekit-agents / livekit-plugins-deepgram APIs; verify field/method
# names against your installed versions once those keys are in place.

async def _forward_audio(track, stt_stream, recorder: Optional[CallRecorder] = None) -> None:
    """Feed the patient's audio to STT *and* to the patient recording.

    This used to push frames to STT only, so the patient's voice was never
    saved. Frames are requested at SAMPLE_RATE mono (LiveKit resamples
    natively) so the same frame is valid for both consumers.
    """
    from livekit import rtc
    audio_stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
    async for event in audio_stream:
        frame = event.frame
        if recorder is not None:
            recorder.write_patient_frame(bytes(frame.data))
        stt_stream.push_frame(frame)


async def _play_pcm16(audio_source, pcm16_bytes: bytes, recorder: Optional[CallRecorder] = None) -> None:
    from livekit import rtc

    frame_ms = 20
    bytes_per_frame = int(SAMPLE_RATE * frame_ms / 1000) * 2  # PCM16
    for i in range(0, len(pcm16_bytes), bytes_per_frame):
        chunk = pcm16_bytes[i:i + bytes_per_frame]
        if not chunk:
            continue
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(chunk) // 2,
        )
        await audio_source.capture_frame(frame)
    if recorder is not None:
        recorder.write_agent_audio(pcm16_bytes)


def _create_stt():
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY not set — see .env.example")
    from livekit.plugins import deepgram

    # Deepgram returns word-level timestamps by default (the `words` array
    # on each alternative); consumed below via alt.words.
    return deepgram.STT(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2",
        language="en-US",
        punctuate=True,
        # Off on purpose. Smart formatting rewrites spoken words into
        # "readable" text, which destroys the counting task: on a live call it
        # turned "one two three ... ten" into "+1 (234) 567-8910" — 3 tokens
        # instead of 10 words, and a wrong speech rate. The features need the
        # words as spoken.
        smart_format=False,
        interim_results=True,
        # The plugin default is 25ms of silence, which splits ordinary speech
        # into many fragments. 300ms still ends a turn quickly; the Listener
        # merges whatever fragments remain.
        endpointing_ms=300,
    )


# --- listening: turning the STT stream into complete answers ----------------

@dataclass
class Utterance:
    text: str
    words: list[dict]


def _word_dicts(alt) -> list[dict]:
    """Deepgram words as plain dicts. In livekit-agents >=1.x they're
    TimedString — a str subclass carrying start_time/end_time, with no
    `.word` attribute; the word itself IS the string."""
    return [
        {"word": str(w), "start": getattr(w, "start_time", None), "end": getattr(w, "end_time", None)}
        for w in (getattr(alt, "words", None) or [])
    ]


def _trim_to_words(start_s: float, end_s: float, words: list[dict], pad_s: float = 0.3) -> tuple[float, float]:
    """Narrow a segment to where the words actually are, dropping the lead-in
    silence before the patient starts (which would skew pause features).
    Word times are positions in the audio sent to STT — the same frames that
    go into patient.wav — but if they don't line up with the markers, the
    markers are kept rather than trusting them."""
    times = [(w["start"], w["end"]) for w in words
             if isinstance(w.get("start"), (int, float)) and isinstance(w.get("end"), (int, float))]
    if not times:
        return start_s, end_s
    first, last = min(t[0] for t in times), max(t[1] for t in times)
    if first < start_s - 1.0 or last > end_s + 1.0:
        return start_s, end_s
    return max(start_s, first - pad_s), min(end_s, last + pad_s)


class Listener:
    """Collects the patient's final transcripts in the background and hands
    them out as complete answers.

    Replaces awaiting the STT stream directly under a timeout. That cancelled
    the stream's __anext__ on every timeout, and took each Deepgram final as
    a whole answer — but Deepgram finalizes on very short pauses, so a
    sentence arrives in pieces and only its first fragment was scored.
    """

    def __init__(self, stt_stream, recorder: "CallRecorder", disconnected: asyncio.Event):
        self._finals: asyncio.Queue[Utterance] = asyncio.Queue()
        self._recorder = recorder
        self._disconnected = disconnected
        self._task = asyncio.create_task(self._pump(stt_stream))

    async def _pump(self, stt_stream) -> None:
        from livekit.agents import stt as lk_stt
        async for event in stt_stream:
            if event.type != lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                continue
            alt = event.alternatives[0]
            if alt.text and alt.text.strip():
                await self._finals.put(Utterance(alt.text.strip(), _word_dicts(alt)))

    def discard_pending(self) -> int:
        """Drop transcripts nobody asked for (e.g. "ahh" fragments during the
        phonation task) so they aren't read as the next answer."""
        dropped = 0
        while not self._finals.empty():
            self._finals.get_nowait()
            dropped += 1
        return dropped

    async def _next(self, timeout: float) -> Optional[Utterance]:
        """The next final transcript, or None on timeout or hang-up."""
        if self._disconnected.is_set():
            return None
        get = asyncio.ensure_future(self._finals.get())
        hang_up = asyncio.ensure_future(self._disconnected.wait())
        done, _ = await asyncio.wait({get, hang_up}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        hang_up.cancel()
        if get in done:
            return get.result()
        get.cancel()  # safe: a cancelled Queue.get leaves any item in the queue
        return None

    async def answer(
        self, first_timeout: float, settle_s: float, max_s: float,
    ) -> tuple[Optional[Utterance], float, float]:
        """Wait for the patient to start, then keep collecting until they've
        been quiet for `settle_s` or `max_s` passes.

        Returns (utterance or None, start, end) — positions in patient.wav.
        """
        loop = asyncio.get_running_loop()
        start_s = self._recorder.patient_seconds
        first = await self._next(first_timeout)
        if first is None:
            return None, start_s, self._recorder.patient_seconds

        parts = [first]
        deadline = loop.time() + max_s
        voiced_waits = 0
        while loop.time() < deadline and not self._disconnected.is_set():
            more = await self._next(min(settle_s, max(0.05, deadline - loop.time())))
            if more is not None:
                parts.append(more)
                voiced_waits = 0
                continue
            since_voice = self._recorder.seconds_since_voice()
            # Voice still on the mic but no transcript yet: STT lags speech,
            # so give it a couple more windows. Capped, so a noisy room can't
            # hold every answer open until max_s.
            if since_voice is not None and since_voice < settle_s and voiced_waits < 2:
                voiced_waits += 1
                continue
            break

        return (
            Utterance(" ".join(p.text for p in parts), [w for p in parts for w in p.words]),
            start_s,
            self._recorder.patient_seconds,
        )

    async def close(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


async def _record_phonation(
    recorder: "CallRecorder", listener: Listener, disconnected: asyncio.Event,
) -> tuple[float, float, bool]:
    """Record the sustained "ahh" from the microphone.

    STT produces no transcript for a sustained vowel (confirmed on a live
    call: a 4-second vowel yielded none), so this stage used to wait out the
    no-answer timeout and capture nothing. Now: wait for voicing to begin,
    record until the patient has been silent for PHONATION_END_SILENCE_S,
    capped at PHONATION_MAX_S. Returns (start, end, voice_detected).
    """
    loop = asyncio.get_running_loop()
    start_s = recorder.patient_seconds
    t0 = loop.time()
    voiced = False
    while not disconnected.is_set():
        elapsed = loop.time() - t0
        if recorder.last_voice_at_s is not None and recorder.last_voice_at_s > start_s:
            voiced = True
            since = recorder.seconds_since_voice()
            if since is not None and since >= PHONATION_END_SILENCE_S:
                break
        elif elapsed >= PHONATION_START_TIMEOUT_S:
            break
        if elapsed >= PHONATION_MAX_S:
            break
        await asyncio.sleep(0.1)
    listener.discard_pending()
    return start_s, recorder.patient_seconds, voiced


def _spawn_post_call_analysis(call_id: int) -> None:
    """Start the after-call analysis in its own detached process.

    Not inline: extracting features is heavy CPU that would stall the call,
    and the LiveKit job is torn down as soon as the call ends, so anything
    still running here would be killed. Output goes to the call's own log.
    """
    repo_root = Path(__file__).resolve().parent.parent
    log_path = CALLS_DIR / str(call_id) / "analysis.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "agent.post_call", "--call-id", str(call_id)],
            cwd=str(repo_root), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, env=os.environ.copy(),
        )
    logger.info("after-call analysis started for call %s (log: %s)", call_id, log_path)


async def _publish_transcript_line(room, speaker: str, text: str) -> None:
    """Send one transcript line to whoever has the call page open.

    The page was listening for LiveKit transcription events, which this agent
    never emits, so its transcript panel stayed empty for the whole call.
    Failures are swallowed on purpose: a display feed must never break a call.
    """
    if not text:
        return
    try:
        payload = json.dumps({"speaker": speaker, "text": text, "at": _now_iso()})
        await room.local_participant.publish_data(
            payload.encode(), topic=TRANSCRIPT_TOPIC, reliable=True
        )
    except Exception:
        logger.debug("could not publish transcript line", exc_info=True)


async def _start_moss_and_preload(moss: MossBridge, patient_id: int) -> None:
    """Start the Moss child processes and load the patient's history.

    Failures are logged, not raised: a call without history is still a call.
    Writes queued before start finishes simply wait for it.
    """
    try:
        await moss.start()
        await moss.preload(patient_id)
    except Exception:
        logger.warning("Moss start/preload failed; call continues without history", exc_info=True)


async def entrypoint(ctx) -> None:
    from livekit import rtc
    from livekit.agents import AutoSubscribe

    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET not set — see .env.example")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # Resolved per call, off the main loop (sqlite is blocking): who we're
    # calling, and a fresh call id so repeat calls don't overwrite each other.
    patient = await asyncio.to_thread(resolve_patient, DB_PATH, _requested_patient_id(participant))
    patient_id = patient.id
    call_id = await asyncio.to_thread(start_call_row, patient_id)
    logger.info(
        "starting call %s for patient %s (%s), timezone %s",
        call_id, patient_id, patient.name, patient.timezone or "unknown (using server clock)",
    )
    # Moss runs in its own processes so its native core can't freeze this
    # call's audio (see agent/moss_worker.py). Start them and load this
    # patient's history while the agent says its first line.
    moss = MossBridge()
    preload_task = asyncio.create_task(_start_moss_and_preload(moss, patient_id))

    script = CallScript(recall_words=recall_words_for_call(call_id))
    turn_ctx = TurnContext(
        patient_id=patient_id,
        call_id=call_id,
        timezone=patient.timezone,
        moss=moss,
    )
    recorder = CallRecorder(call_id, patient_id)

    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", audio_source)
    await ctx.room.local_participant.publish_track(track)

    async def speak(text: str) -> None:
        pcm = await synthesize_speech(text)
        # Publish before playing so the line appears as the agent starts talking.
        await _publish_transcript_line(ctx.room, "agent", text)
        await _play_pcm16(audio_source, pcm, recorder=recorder)
        recorder.log_agent_turn(text)

    stt_stream = _create_stt().stream()

    # Without this the job never ends when the patient hangs up: the STT
    # stream just blocks, and LiveKit eventually force-cancels the entrypoint
    # ("entrypoint did not exit in time"), so the call shuts down dirtily.
    disconnected = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(*_args):
        disconnected.set()

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_args):
        disconnected.set()

    # One patient-audio forwarder at a time: it feeds both STT and the
    # recording, and two running at once would interleave frames in the WAV.
    forwarder: dict[str, Optional[asyncio.Task]] = {"task": None}

    def _start_forwarding(track_) -> None:
        if forwarder["task"] is not None and not forwarder["task"].done():
            forwarder["task"].cancel()
        forwarder["task"] = asyncio.create_task(_forward_audio(track_, stt_stream, recorder))

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(track_, *_args):
        if track_.kind == rtc.TrackKind.KIND_AUDIO:
            _start_forwarding(track_)

    # The patient's track can already be subscribed by this point: subscribing
    # starts at connect, and resolving the patient takes a moment. Then
    # track_subscribed fired before the handler above existed, and STT would
    # never receive any audio. Pick up an already-subscribed track explicitly.
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track is not None and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                _start_forwarding(publication.track)
                break

    await speak(script.current_prompt())

    # The whole conversation runs under try/finally: a call that ends early —
    # patient hangs up, network drops, the room closes — must still write its
    # transcript and persist to Moss. Without this, anything short of a fully
    # completed 5-stage call loses the transcript entirely (the audio survives,
    # since wave writes incrementally, but every word and timestamp is gone),
    # which is exactly the data the ML pipeline needs.
    listener = Listener(stt_stream, recorder, disconnected)

    async def move_on() -> bool:
        """Advance to the next prompt and say it; False once the call is over."""
        next_prompt = script.advance()
        if script.is_complete():
            await speak("That's everything for today — thanks for checking in!")
            return False
        await speak(next_prompt)
        return True

    try:
        while not script.is_complete():
            if disconnected.is_set():
                logger.info("participant left — ending call %s", turn_ctx.call_id)
                break

            qid = script.current_question_id()
            mode = script.capture_mode()

            if mode is CaptureMode.TIMED_AUDIO:
                # #3: the sustained vowel isn't speech, so record it from the
                # microphone instead of waiting for a transcript that never comes.
                start_s, end_s, voiced = await _record_phonation(recorder, listener, disconnected)
                if voiced:
                    recorder.mark_segment(script.segment_name() or qid, start_s, end_s)
                    logger.info("%s: recorded %.1fs of patient audio", qid, end_s - start_s)
                else:
                    # Don't mark silence as a phonation segment: the after-call
                    # analysis would extract jitter/shimmer from nothing, write
                    # NaN features, and still print a confident severity.
                    logger.info("%s: no voice detected in %.1fs — no phonation segment",
                                qid, end_s - start_s)
                if not await move_on():
                    break
                continue

            if mode is CaptureMode.LONG_SPEECH:
                settle_s, max_s = LONG_SPEECH_SETTLE_S, LONG_SPEECH_MAX_S
            else:
                settle_s, max_s = ANSWER_SETTLE_S, ANSWER_MAX_S

            utterance, start_s, end_s = await listener.answer(NO_ANSWER_TIMEOUT_S, settle_s, max_s)
            if utterance is None:
                if disconnected.is_set():
                    break
                # A silent patient: keep the call moving instead of hanging.
                logger.info("no answer within %ss on %s — advancing", NO_ANSWER_TIMEOUT_S, qid)
                if not await move_on():
                    break
                continue

            recorder.log_patient_turn(utterance.text, utterance.words)
            await _publish_transcript_line(ctx.room, "patient", utterance.text)
            if mode is CaptureMode.LONG_SPEECH and script.segment_name():
                # #4: keep the reading/counting audio and its words together,
                # trimmed to the speech, for speech-rate and pause features.
                seg_start, seg_end = _trim_to_words(start_s, end_s, utterance.words)
                recorder.mark_segment(script.segment_name(), seg_start, seg_end,
                                      utterance.text, utterance.words)

            outcome = await handle_turn(
                script, turn_ctx, utterance.text, speak=speak, word_timestamps=utterance.words
            )
            if outcome.trigger.fired and outcome.trigger.reason:
                if qid.startswith("delayed_recall:"):
                    # Each attempt's flag is superseded by the next, which now
                    # counts cumulatively. Keeping all of them left the record
                    # ending on the lowest and least true of the three.
                    recorder.resolve_flags(qid, f"superseded by: {outcome.trigger.reason}")
                recorder.log_flag(qid, outcome.trigger.reason)
            elif qid.startswith(("recall_check:", "delayed_recall:")):
                # Same qid as the answer that flagged: the script only advances
                # once the question passes, so reaching here means a later
                # attempt succeeded and the standing flag is wrong. Orientation:
                # Deepgram rendered "autumn" as "awesome", leaving a healthy
                # patient flagged as disoriented until the clarifier put it
                # right. Delayed recall: the words came back over successive
                # attempts, so the cumulative count reached the threshold.
                withdrawn = recorder.resolve_flags(
                    qid, f"answered correctly on follow-up: {utterance.text!r}"
                )
                if withdrawn:
                    logger.info("%s: withdrew %d flag(s) — the follow-up answer was correct",
                                qid, withdrawn)
            await speak(outcome.reply_text)
            if script.is_complete():
                break
    finally:
        await listener.close()
        try:
            # Answers are saved in the background during the call; give the
            # last few a chance to land before the job exits.
            await turn_ctx.flush_writes()
        except Exception:
            logger.exception("failed waiting for Moss writes on call %s", turn_ctx.call_id)
        finally:
            if not preload_task.done():
                preload_task.cancel()
            await moss.close()
        if forwarder["task"] is not None:
            forwarder["task"].cancel()  # stop writing patient audio before the file closes
        try:
            recorder.finalize()
        except Exception:
            logger.exception("failed to finalize call recording for call %s", turn_ctx.call_id)
        else:
            try:
                _spawn_post_call_analysis(turn_ctx.call_id)
            except Exception:
                logger.exception("could not start after-call analysis for call %s", turn_ctx.call_id)
        try:
            # Persist this patient's Moss session for their next call.
            await push_patient_session(turn_ctx.patient_id)
        except Exception:
            logger.exception("failed to push Moss session for patient %s", turn_ctx.patient_id)


def prewarm(proc) -> None:
    """Runs once per worker process, before it takes any call.

    Pays the slow, synchronous start-up costs (native SDK imports, client
    construction) here instead of on the first call's event loop, where they
    froze audio for seconds.
    """
    # Imported for the side effect of paying their (slow) import cost here.
    for module in ("av", "edge_tts", "livekit.plugins.deepgram"):
        importlib.import_module(module)

    # Moss is deliberately not imported here: it runs in child processes
    # (agent/moss_worker.py) so its native core can't freeze calls.
    if OPENAI_API_KEY:
        # Touching .chat matters: everything under it is a lazy cached_property
        # that imports ~30 modules on first use. Left until the first live
        # follow-up, that import blocked the event loop for 248ms mid-call.
        _ = _get_llm_client().chat


def run_worker() -> None:
    from livekit.agents import WorkerOptions, cli
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    if "--dry-run" in sys.argv or len(sys.argv) == 1:
        asyncio.run(dry_run())
    else:
        run_worker()
