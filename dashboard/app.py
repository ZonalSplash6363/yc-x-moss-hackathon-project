"""
Clinician dashboard — /dashboard zone.

What a clinician would actually open between visits: every check-in call for a
patient, the voice measurements taken from it, what the agent flagged, and the
full conversation. Reads the committed snapshot built by build_snapshot.py
(see that module for why it isn't the live database).

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "demo_data.json"

# The measurements worth plotting, with the units the report uses. Jitter and
# shimmer are stored as ratios and shown as percentages, as clinicians read them.
MEASURES = {
    "jitter_local": ("Jitter (local)", "%", 100),
    "shimmer_local": ("Shimmer (local)", "%", 100),
    "hnr": ("Harmonic-to-noise ratio", "dB", 1),
    "rpde": ("RPDE", "", 1),
    "dfa": ("DFA", "", 1),
    "ppe": ("PPE", "", 1),
    "speech_rate": ("Speech rate", "words/min", 1),
    "pause_freq": ("Pause frequency", "pauses/s", 1),
}

st.set_page_config(page_title="Voice Check-in — Clinician View",
                   page_icon="🎙️", layout="wide")


@st.cache_data
def load_data() -> dict:
    if not DATA_PATH.exists():
        return {"patients": [], "calls": [], "generated_at": None}
    return json.loads(DATA_PATH.read_text())


data = load_data()
if not data["calls"]:
    st.error("No snapshot found. Run `python -m dashboard.build_snapshot` first.")
    st.stop()

st.title("🎙️ Parkinson's Voice Check-in")
st.caption("Between-visit monitoring from a short phone conversation.")

# --- the caveat goes at the top, not buried in a footnote --------------------
st.warning(
    "**The UPDRS severity estimate on this page is not a clinical measurement.** "
    "The model was trained on the UCI Telemonitoring dataset and does not "
    "generalise to patients it has never heard: held-out R² is −0.25, where "
    "simply predicting the average scores −0.06. It is shown because it is part "
    "of the pipeline, not because it is trustworthy. The *measured* voice "
    "features above it are real."
)

# --- patient selection -------------------------------------------------------
patients = {p["id"]: p["name"] for p in data["patients"]}
call_counts = {}
for c in data["calls"]:
    call_counts[c["patient_id"]] = call_counts.get(c["patient_id"], 0) + 1

with st.sidebar:
    st.header("Patient")
    options = [pid for pid in patients if call_counts.get(pid)]
    if not options:
        st.error("No patient has any recorded calls.")
        st.stop()
    patient_id = st.selectbox(
        "Select", options,
        format_func=lambda pid: f"{patients[pid]} — {call_counts.get(pid, 0)} calls",
    )
    if data.get("generated_at"):
        st.caption(f"Snapshot: {data['generated_at'][:10]}")

calls = [c for c in data["calls"] if c["patient_id"] == patient_id]
calls.sort(key=lambda c: c["id"])
scored = [c for c in calls if c["scored"]]

# --- summary -----------------------------------------------------------------
standing_flags = sum(
    1 for c in calls for f in c["flags"] if not f["withdrawn"]
)
withdrawn_flags = sum(1 for c in calls for f in c["flags"] if f["withdrawn"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Calls completed", len(calls))
col2.metric("Produced measurements", len(scored))
col3.metric("Flags raised", standing_flags)
col4.metric("Flags withdrawn", withdrawn_flags,
            help="Raised mid-call, then disproved by a later answer — a "
                 "mis-transcribed word, or words recalled after a cue.")

tab_trend, tab_calls, tab_detail = st.tabs(
    ["Voice trend", "Call history", "Call detail"]
)

# --- trends ------------------------------------------------------------------
with tab_trend:
    if len(scored) < 2:
        st.info("At least two scored calls are needed to show a trend.")
    else:
        rows = []
        for c in scored:
            row = {"call": f"#{c['id']}", "date": c["timestamp"][:10]}
            row.update(c["features"])
            rows.append(row)
        df = pd.DataFrame(rows)

        st.subheader("Measured from the sustained vowel and reading passage")
        picked = st.multiselect(
            "Measures", list(MEASURES),
            default=["jitter_local", "shimmer_local", "hnr"],
            format_func=lambda k: MEASURES[k][0],
        )
        cols = st.columns(min(len(picked), 3)) if picked else []
        for i, key in enumerate(picked):
            label, unit, scale = MEASURES[key]
            with cols[i % len(cols)]:
                st.caption(f"{label}{f' ({unit})' if unit else ''}")
                plot = df[["call", key]].copy()
                plot[key] = plot[key] * scale
                st.line_chart(plot.set_index("call"), height=200)

        st.info(
            "These swing considerably between calls on the same person — "
            "recording level moves shimmer and HNR more than short-term "
            "physiology does. Comparisons are only meaningful once capture is "
            "calibrated, which is the next piece of work.",
            icon="⚠️",
        )

# --- history -----------------------------------------------------------------
with tab_calls:
    rows = []
    for c in calls:
        standing = [f for f in c["flags"] if not f["withdrawn"]]
        rows.append({
            "Call": f"#{c['id']}",
            "Date": c["timestamp"][:10],
            "Outcome": "measured" if c["scored"] else "declined — no usable voice",
            "Flags": len(standing),
            "Withdrawn": len([f for f in c["flags"] if f["withdrawn"]]),
            "UPDRS (experimental)": (
                c["prediction"]["predicted_score"] if c.get("prediction") else None
            ),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "A call that captured no usable voice — silence, or a hang-up — is "
        "declined rather than scored. Inventing a measurement from silence "
        "would put a number in this patient's history that nothing supports."
    )

# --- detail ------------------------------------------------------------------
with tab_detail:
    chosen = st.selectbox(
        "Call", [c["id"] for c in calls],
        format_func=lambda cid: next(
            f"#{cid} — {c['timestamp'][:10]}" for c in calls if c["id"] == cid
        ),
        index=len(calls) - 1,
    )
    call = next(c for c in calls if c["id"] == chosen)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Conversation")
        for turn in call["turns"]:
            who = "🩺 Agent" if turn["speaker"] == "agent" else "🗣️ Patient"
            st.markdown(f"**{who}** · {turn['text']}")

    with right:
        st.subheader("Raised during the call")
        standing = [f for f in call["flags"] if not f["withdrawn"]]
        withdrawn = [f for f in call["flags"] if f["withdrawn"]]
        if not standing:
            st.success("Nothing flagged.")
        for f in standing:
            # Must be a real emoji: Streamlit validates this argument and
            # raises on dingbats like U+2691, which halts the whole script.
            st.error(f["reason"], icon="🚩")
        for f in withdrawn:
            st.caption(f"~~{f['reason']}~~ — withdrawn: {f['withdrawn_because']}")

        st.subheader("Recorded")
        if call["segments"]:
            st.dataframe(
                pd.DataFrame([
                    {"Task": s["name"], "Seconds": s["duration_s"]}
                    for s in call["segments"]
                ]),
                width="stretch", hide_index=True,
            )
        else:
            st.caption("No task audio captured.")

        if call["scored"]:
            st.subheader("Measured")
            feats = call["features"]
            st.markdown(
                f"- Jitter **{feats['jitter_local'] * 100:.2f}%**\n"
                f"- Shimmer **{feats['shimmer_local'] * 100:.2f}%**\n"
                f"- HNR **{feats['hnr']:.1f} dB**\n"
                f"- Speech rate **{feats['speech_rate']:.0f} words/min**"
            )

    if call.get("report"):
        with st.expander("Full report as generated after the call"):
            st.code(call["report"], language=None)
