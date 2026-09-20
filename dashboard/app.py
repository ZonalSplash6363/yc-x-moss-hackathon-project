"""
Clinician dashboard — /dashboard zone.

What a clinician would open between visits: every check-in call for a patient,
the voice measurements taken from it, what the agent flagged, and the full
conversation. Reads the committed snapshot built by build_snapshot.py (see
that module for why it isn't the live database).

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "demo_data.json"

# Reference ranges are the commonly cited MDVP/Praat values for sustained
# phonation in healthy adults. They are here to make a number legible at a
# glance — "is this near normal?" — and are NOT thresholds this system has
# validated on anyone. Shown as a shaded band, never as a pass/fail verdict.
REFERENCE = {
    "jitter_local": (None, 1.04, "%", 100, "Jitter (local)"),
    "shimmer_local": (None, 3.81, "%", 100, "Shimmer (local)"),
    "hnr": (20.0, None, "dB", 1, "Harmonic-to-noise ratio"),
    "rpde": (None, None, "", 1, "RPDE"),
    "dfa": (None, None, "", 1, "DFA"),
    "ppe": (None, None, "", 1, "PPE"),
    "speech_rate": (None, None, "words/min", 1, "Speech rate"),
    "pause_freq": (None, None, "pauses/s", 1, "Pause frequency"),
}

TASK_LABELS = {
    "phonation": "Sustained vowel",
    "reading": "Reading passage",
    "counting": "Counting 1–20",
}

st.set_page_config(
    page_title="Voice Check-in — Clinician View",
    page_icon="+",
    layout="wide",
)

# Clinical styling. Streamlit's defaults look like a notebook; medical records
# software is dense, quiet, and typographically flat — information carries the
# emphasis, not colour or shadow.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      h1, h2, h3 { letter-spacing: -0.01em; color: #12202B; font-weight: 600; }
      h1 { font-size: 1.45rem !important; }
      h2 { font-size: 1.08rem !important; margin-top: 0.4rem; }
      h3 { font-size: 0.95rem !important; }
      /* Numbers line up when they're tabular. */
      .clin-num, [data-testid="stMetricValue"] {
          font-variant-numeric: tabular-nums;
      }
      .masthead {
          border-bottom: 2px solid #12202B; padding-bottom: 0.5rem;
          margin-bottom: 1.1rem; display: flex; align-items: baseline; gap: 0.6rem;
      }
      .masthead .mark {
          font-weight: 700; color: #1B6B8A; letter-spacing: 0.09em;
          font-size: 0.72rem; text-transform: uppercase;
      }
      .masthead .sub { color: #5C6B77; font-size: 0.78rem; margin-left: auto; }
      .banner {
          background: #F4F6F8; border: 1px solid #DDE3E8; border-left: 3px solid #1B6B8A;
          padding: 0.85rem 1.1rem; margin-bottom: 1.1rem;
      }
      .banner .name { font-size: 1.12rem; font-weight: 650; color: #12202B; }
      .banner .meta { color: #5C6B77; font-size: 0.79rem; margin-top: 0.28rem; }
      .banner .meta b { color: #33414D; font-weight: 600; }
      .tiles { display: flex; gap: 0; border: 1px solid #DDE3E8; margin-bottom: 1.2rem; }
      .tile { flex: 1; padding: 0.7rem 0.95rem; border-right: 1px solid #DDE3E8; }
      .tile:last-child { border-right: none; }
      .tile .k {
          font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em;
          color: #6B7883; font-weight: 600;
      }
      .tile .v {
          font-size: 1.5rem; font-weight: 600; color: #12202B;
          font-variant-numeric: tabular-nums; line-height: 1.25;
      }
      .tile .v.warn { color: #A8500F; }
      .tile .v.mute { color: #6B7883; }
      .disclaimer {
          border: 1px solid #E4CBA6; background: #FDF8F0; padding: 0.8rem 1rem;
          font-size: 0.83rem; color: #4A3A22; margin-bottom: 1.1rem;
      }
      .flag-row {
          border-left: 3px solid #B3261E; background: #FCF3F2;
          padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; font-size: 0.85rem;
          color: #3D1512;
      }
      .flag-row.ok { border-left-color: #1E6B45; background: #F1F8F4; color: #14341F; }
      .flag-row.gone {
          border-left-color: #9AA5AE; background: #F7F8F9; color: #6B7883;
      }
      .flag-row .qid {
          font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
          color: #8A959E; display: block; margin-bottom: 0.12rem;
      }
      .flag-row s { color: #8A959E; }
      .turn { margin: 0 0 0.5rem 0; font-size: 0.87rem; line-height: 1.5; }
      /* A speaker label longer than min-width would otherwise butt straight
         against the utterance ("CLINICIAN AGENTWhat day..."), so the gap is a
         margin rather than relying on the width alone. */
      .turn .who {
          display: inline-block; min-width: 5.2rem; margin-right: 0.6rem;
          font-weight: 650; font-size: 0.68rem; text-transform: uppercase;
          letter-spacing: 0.06em; color: #6B7883; vertical-align: top;
      }
      .turn.pt .who { color: #1B6B8A; }
      .turn .said { display: inline; color: #26333D; }
      .measure-row {
          display: flex; justify-content: space-between; padding: 0.38rem 0;
          border-bottom: 1px solid #EDF0F2; font-size: 0.86rem;
      }
      .measure-row .lab { color: #5C6B77; }
      .measure-row .val { font-variant-numeric: tabular-nums; font-weight: 600; color: #12202B; }
      .measure-row .val.out { color: #A8500F; }
      .ref { color: #8A959E; font-size: 0.72rem; font-weight: 400; }
      footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(snapshot_mtime: float) -> dict:
    """The parsed snapshot, keyed on the file's modification time.

    The mtime argument is the point: a no-argument cache would hold the first
    read for the life of the process, so regenerating the snapshot — after a
    new call, say — would leave the page showing stale data until it was
    restarted.
    """
    if not DATA_PATH.exists():
        return {"patients": [], "calls": [], "generated_at": None}
    return json.loads(DATA_PATH.read_text())


data = load_data(DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)

st.markdown(
    '<div class="masthead"><span class="mark">Voice Check-in</span>'
    '<span style="color:#5C6B77;font-size:0.82rem;">Remote monitoring for '
    'Parkinson&rsquo;s disease</span>'
    f'<span class="sub">Data as of {(data.get("generated_at") or "—")[:10]}</span></div>',
    unsafe_allow_html=True,
)

if not data["calls"]:
    st.error("No snapshot found. Run `python -m dashboard.build_snapshot` first.")
    st.stop()

# --- patient selection -------------------------------------------------------
patients = {p["id"]: p for p in data["patients"]}
call_counts: dict[int, int] = {}
for c in data["calls"]:
    call_counts[c["patient_id"]] = call_counts.get(c["patient_id"], 0) + 1

with st.sidebar:
    st.markdown("### Patient list")
    options = [pid for pid in patients if call_counts.get(pid)]
    if not options:
        st.error("No patient has any recorded calls.")
        st.stop()
    patient_id = st.selectbox(
        "Select a patient", options,
        format_func=lambda pid: f"{patients[pid]['name']} · {call_counts.get(pid, 0)} check-ins",
        label_visibility="collapsed",
    )
    st.caption(
        "Check-ins are placed automatically on the patient's schedule. "
        "This view is read-only."
    )

patient = patients[patient_id]
calls = sorted(
    [c for c in data["calls"] if c["patient_id"] == patient_id],
    key=lambda c: c["id"],
)
scored = [c for c in calls if c["scored"]]
latest = calls[-1]

# --- patient banner ----------------------------------------------------------
sched = ""
if patient.get("call_frequency"):
    times = ", ".join(patient.get("preferred_call_times") or [])
    sched = f"{patient['call_frequency'].replace('_', ' ')}{f' at {times}' if times else ''}"

meta_bits = [f"<b>MRN</b> VCI-{patient['id']:05d}"]
if patient.get("timezone"):
    meta_bits.append(f"<b>Timezone</b> {patient['timezone']}")
if sched:
    meta_bits.append(f"<b>Schedule</b> {sched}")
if patient.get("enrolled"):
    meta_bits.append(f"<b>Enrolled</b> {patient['enrolled'][:10]}")
if patient.get("caregiver_name"):
    meta_bits.append(f"<b>Care contact</b> {patient['caregiver_name']}")

st.markdown(
    f'<div class="banner"><div class="name">{patient["name"]}</div>'
    f'<div class="meta">{" &nbsp;·&nbsp; ".join(meta_bits)}</div></div>',
    unsafe_allow_html=True,
)

# --- summary tiles -----------------------------------------------------------
standing_total = sum(1 for c in calls for f in c["flags"] if not f["withdrawn"])
withdrawn_total = sum(1 for c in calls for f in c["flags"] if f["withdrawn"])
declined_total = len(calls) - len(scored)

st.markdown(
    f"""
    <div class="tiles">
      <div class="tile"><div class="k">Check-ins</div><div class="v">{len(calls)}</div></div>
      <div class="tile"><div class="k">Yielded measurements</div><div class="v">{len(scored)}</div></div>
      <div class="tile"><div class="k">No usable voice</div>
        <div class="v {'mute' if not declined_total else ''}">{declined_total}</div></div>
      <div class="tile"><div class="k">Findings open</div>
        <div class="v {'warn' if standing_total else 'mute'}">{standing_total}</div></div>
      <div class="tile"><div class="k">Findings withdrawn</div><div class="v mute">{withdrawn_total}</div></div>
      <div class="tile"><div class="k">Most recent</div>
        <div class="v" style="font-size:1rem;padding-top:0.32rem;">{latest['timestamp'][:10]}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_trend, tab_history, tab_detail = st.tabs(
    ["Voice measurements", "Check-in history", "Check-in detail"]
)


def _chart(df: pd.DataFrame, key: str) -> alt.LayerChart | alt.Chart:
    """One measurement over time, with its reference band behind it."""
    low, high, unit, scale, label = REFERENCE[key]
    plot = df[["label", key]].copy()
    plot[key] = plot[key] * scale
    axis_title = f"{label}{f' ({unit})' if unit else ''}"

    line = (
        alt.Chart(plot)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), strokeWidth=2)
        .encode(
            x=alt.X("label:N", title=None, sort=None,
                    axis=alt.Axis(labelAngle=0, labelColor="#6B7883", domainColor="#C9D2D9")),
            y=alt.Y(f"{key}:Q", title=axis_title, scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelColor="#6B7883", titleColor="#5C6B77",
                                  gridColor="#EDF0F2", domainColor="#C9D2D9")),
            color=alt.value("#1B6B8A"),
            tooltip=[alt.Tooltip("label:N", title="Check-in"),
                     alt.Tooltip(f"{key}:Q", title=axis_title, format=".3f")],
        )
        .properties(height=190)
    )

    if low is None and high is None:
        return line

    lo = low if low is not None else float(plot[key].min()) - 1
    hi = high if high is not None else float(plot[key].max()) + 1
    band = (
        alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]}))
        .mark_rect(opacity=0.10, color="#1E6B45")
        .encode(y="lo:Q", y2="hi:Q")
    )
    return band + line


# --- voice measurements ------------------------------------------------------
with tab_trend:
    if not scored:
        st.info("No check-in has yielded a usable voice sample yet.")
    else:
        rows = []
        for c in scored:
            # Label by check-in number, not date. Several check-ins can land on
            # the same day — four of these did — and a date-only label collapses
            # them onto one category, drawing a trend line as a vertical bar.
            row = {"label": f"#{c['id']}", "date": c["timestamp"][:10], "call": c["id"]}
            row.update(c["features"])
            rows.append(row)
        df = pd.DataFrame(rows)

        st.markdown("## Acoustic measures")
        st.caption(
            "Taken from the sustained vowel. The shaded band is the commonly "
            "cited healthy range from the phonation literature — context for "
            "reading a number, not a threshold this system has validated."
        )
        if len(df) < 2:
            st.info("A trend needs at least two measured check-ins.")
        else:
            c1, c2, c3 = st.columns(3)
            for col, key in zip((c1, c2, c3), ("jitter_local", "shimmer_local", "hnr")):
                with col:
                    st.altair_chart(_chart(df, key), width="stretch")

            st.markdown("## Nonlinear and timing measures")
            st.caption(
                "RPDE, DFA and PPE describe how regular the vocal fold cycle is. "
                "Speech timing comes from the reading passage."
            )
            c4, c5, c6 = st.columns(3)
            for col, key in zip((c4, c5, c6), ("ppe", "speech_rate", "pause_freq")):
                with col:
                    st.altair_chart(_chart(df, key), width="stretch")

        st.markdown(
            '<div class="disclaimer"><b>Interpret the spread with care.</b> '
            'These measures move substantially between check-ins on the same '
            'person, and recording conditions — microphone distance and gain — '
            'shift shimmer and harmonic-to-noise ratio more than short-term '
            'physiology does. Calibrated capture is required before '
            'check-in-to-check-in comparison carries weight.</div>',
            unsafe_allow_html=True,
        )

# --- history -----------------------------------------------------------------
with tab_history:
    st.markdown("## Check-in history")
    rows = []
    for c in reversed(calls):
        standing = [f for f in c["flags"] if not f["withdrawn"]]
        feats = c["features"] or {}
        rows.append({
            "Date": c["timestamp"][:10],
            "Check-in": f"#{c['id']}",
            "Result": "Measured" if c["scored"] else "No usable voice — not scored",
            "Findings": len(standing),
            "Withdrawn": len([f for f in c["flags"] if f["withdrawn"]]),
            "Jitter %": round(feats["jitter_local"] * 100, 2) if feats else None,
            "Shimmer %": round(feats["shimmer_local"] * 100, 2) if feats else None,
            "HNR dB": round(feats["hnr"], 1) if feats else None,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "A check-in that captured no usable voice — silence, or the patient "
        "hanging up — is recorded as unscored rather than estimated. Producing "
        "a measurement from silence would place a number in this patient's "
        "record that nothing supports."
    )

# --- detail ------------------------------------------------------------------
with tab_detail:
    chosen = st.selectbox(
        "Check-in", [c["id"] for c in reversed(calls)],
        format_func=lambda cid: next(
            f"{c['timestamp'][:10]} · check-in #{cid}" for c in calls if c["id"] == cid
        ),
    )
    call = next(c for c in calls if c["id"] == chosen)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("## Conversation")
        for turn in call["turns"]:
            is_pt = turn["speaker"] != "agent"
            who = "Patient" if is_pt else "Clinician agent"
            st.markdown(
                f'<div class="turn {"pt" if is_pt else ""}">'
                f'<span class="who">{who}</span>'
                f'<span class="said">{turn["text"]}</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("## Findings")
        standing = [f for f in call["flags"] if not f["withdrawn"]]
        withdrawn = [f for f in call["flags"] if f["withdrawn"]]
        if not standing and not withdrawn:
            st.markdown('<div class="flag-row ok">Nothing flagged during this '
                        'check-in.</div>', unsafe_allow_html=True)
        for f in standing:
            st.markdown(
                f'<div class="flag-row"><span class="qid">{f["question_id"]}</span>'
                f'{f["reason"]}</div>', unsafe_allow_html=True,
            )
        for f in withdrawn:
            st.markdown(
                f'<div class="flag-row gone"><span class="qid">{f["question_id"]} '
                f'· withdrawn</span><s>{f["reason"]}</s><br>'
                f'{f["withdrawn_because"]}</div>', unsafe_allow_html=True,
            )
        if withdrawn:
            st.caption(
                "A finding is withdrawn when something later in the same "
                "check-in disproves it — a mis-transcribed word, or words "
                "recalled after a cue. The original is kept, not deleted."
            )

        st.markdown("## Tasks completed")
        if call["segments"]:
            for s in call["segments"]:
                st.markdown(
                    f'<div class="measure-row"><span class="lab">'
                    f'{TASK_LABELS.get(s["name"], s["name"])}</span>'
                    f'<span class="val">{s["duration_s"]:.1f}s</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="measure-row"><span class="lab">No task audio '
                        'captured</span><span class="val">—</span></div>',
                        unsafe_allow_html=True)

        if call["scored"]:
            st.markdown("## Measurements")
            f = call["features"]
            for key in ("jitter_local", "shimmer_local", "hnr"):
                low, high, unit, scale, label = REFERENCE[key]
                value = f[key] * scale
                out = (high is not None and value > high) or (low is not None and value < low)
                if high is not None:
                    ref = f"ref &lt; {high} {unit}"
                elif low is not None:
                    ref = f"ref &gt; {low} {unit}"
                else:
                    ref = ""
                st.markdown(
                    f'<div class="measure-row"><span class="lab">{label} '
                    f'<span class="ref">{ref}</span></span>'
                    f'<span class="val {"out" if out else ""}">{value:.2f} {unit}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="measure-row"><span class="lab">Speech rate</span>'
                f'<span class="val">{f["speech_rate"]:.0f} words/min</span></div>',
                unsafe_allow_html=True,
            )

        if call.get("prediction"):
            st.markdown("## Research model output")
            st.markdown(
                '<div class="disclaimer">'
                f'<b>UPDRS {call["prediction"]["predicted_score"]}</b> — not a '
                'clinical measurement, and not to be acted on. This model does '
                'not generalise to patients it was not trained on: held-out '
                'R&sup2; is &minus;0.25, where predicting the cohort average '
                'scores &minus;0.06. It is surfaced because it is part of the '
                'pipeline, not because it is reliable. The measurements above '
                'are real; this number is not.</div>',
                unsafe_allow_html=True,
            )

    if call.get("report"):
        with st.expander("Report generated after this check-in"):
            st.code(call["report"], language=None)
