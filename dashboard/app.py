"""
Clinician dashboard — /dashboard zone.

What a clinician would open between visits: every check-in call for a patient,
the voice measurements taken from it, what the agent flagged, and the full
conversation. Reads the committed snapshot built by build_snapshot.py (see
that module for why it isn't the live database).

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "demo_data.json"

# (low, high, unit, display scale, label). Reference values are the commonly
# cited MDVP/Praat figures for sustained phonation in healthy adults. They are
# here so a number can be read at a glance — is this near normal? — and are NOT
# thresholds this system has validated on any patient. Measures with no
# published consensus range carry None and are reported without a flag.
REFERENCE = {
    "jitter_local": (None, 1.04, "%", 100, "Jitter (local)"),
    "jitter_rap": (None, 0.68, "%", 100, "Jitter (RAP)"),
    "shimmer_local": (None, 3.81, "%", 100, "Shimmer (local)"),
    "shimmer_apq5": (None, 3.07, "%", 100, "Shimmer (APQ5)"),
    "hnr": (20.0, None, "dB", 1, "Harmonic-to-noise ratio"),
    "rpde": (None, None, "", 1, "RPDE"),
    "dfa": (None, None, "", 1, "DFA"),
    "ppe": (None, None, "", 1, "PPE"),
    "speech_rate": (None, None, "/min", 1, "Speech rate"),
    "pause_freq": (None, None, "/s", 1, "Pause frequency"),
    "pause_avg_duration": (None, None, "s", 1, "Mean pause duration"),
}

TREND_KEYS = ("jitter_local", "shimmer_local", "hnr", "ppe", "speech_rate", "pause_freq")

TASK_LABELS = {
    "phonation": "Sustained vowel /a/",
    "reading": "Reading passage",
    "counting": "Serial counting 1–20",
}

st.set_page_config(
    page_title="Voice Check-in — Clinician View",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Records software is dense, drab and typographically flat: small type, hairline
# rules, square corners, numbers right-aligned and tabular, colour spent only on
# an abnormal result. Streamlit's defaults are the opposite of all of that.
st.markdown(
    """
    <style>
      /* Streamlit's header is a 60px absolute bar. The default top padding
         exists to clear it; anything smaller slides content underneath. */
      .block-container { padding-top: 4.9rem !important; padding-bottom: 3rem;
                         max-width: 1320px; }
      html, body, [class*="css"] { font-size: 13px; }
      .stApp { background: #FFFFFF; }
      h1,h2,h3,h4 { color:#101C25; font-weight:600; letter-spacing:-0.005em; }
      h2 { font-size:0.95rem !important; margin:1.1rem 0 0.35rem !important;
           text-transform:uppercase; letter-spacing:0.07em; color:#4A5A66 !important;
           border-bottom:1px solid #D8DEE3; padding-bottom:0.3rem; }
      /* Square everything off. */
      .stTabs [data-baseweb="tab"], .stSelectbox div, .stDataFrame,
      div[data-testid="stExpander"] { border-radius:0 !important; }
      .stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid #D8DEE3; }
      .stTabs [data-baseweb="tab"] {
          height:32px; padding:0 16px; font-size:0.8rem; font-weight:600;
          text-transform:uppercase; letter-spacing:0.05em; color:#5C6B77;
      }
      .stTabs [aria-selected="true"] { color:#0E4C63 !important; background:#EDF1F4; }

      /* Masthead */
      .mast { display:flex; align-items:baseline; gap:0.55rem;
              border-bottom:2px solid #101C25; padding-bottom:0.35rem; }
      .mast .m1 { font-weight:700; font-size:0.74rem; letter-spacing:0.12em;
                  text-transform:uppercase; color:#0E4C63; }
      .mast .m2 { font-size:0.76rem; color:#6B7883; }
      .mast .m3 { margin-left:auto; font-size:0.7rem; color:#8A959E;
                  font-variant-numeric:tabular-nums; }

      /* Patient banner — one dense strip, EHR style */
      .pbar { background:#EDF1F4; border:1px solid #D8DEE3; border-top:3px solid #0E4C63;
              padding:0.5rem 0.8rem; margin:0.55rem 0 0.2rem;
              display:flex; align-items:baseline; flex-wrap:wrap; gap:0 1.2rem; }
      .pbar .nm { font-size:1rem; font-weight:700; color:#101C25;
                  text-transform:uppercase; letter-spacing:0.02em; }
      .pbar .f { font-size:0.74rem; color:#4A5A66; font-variant-numeric:tabular-nums; }
      /* inline-block, or the margin collapses in the flex row and the label
         runs straight into its value ("MRNVCI-00002"). */
      .pbar .f span { color:#8A959E; text-transform:uppercase; letter-spacing:0.06em;
                      font-size:0.66rem; display:inline-block; margin-right:0.4rem; }

      /* Lab-panel table */
      table.lab { width:100%; border-collapse:collapse; font-size:0.8rem;
                  font-variant-numeric:tabular-nums; }
      table.lab th { text-align:left; font-size:0.64rem; text-transform:uppercase;
                     letter-spacing:0.07em; color:#6B7883; font-weight:600;
                     border-bottom:1px solid #A9B4BD; padding:0.3rem 0.5rem; }
      table.lab td { padding:0.28rem 0.5rem; border-bottom:1px solid #E8ECEF;
                     color:#1A2530; }
      table.lab td.an { color:#33414D; }
      table.lab .num, table.lab th.num { text-align:right; font-weight:600; }
      table.lab .u { color:#8A959E; font-weight:400; }
      table.lab .ref { color:#8A959E; font-size:0.74rem; }
      table.lab .flag, table.lab th.flag { text-align:center; width:3.2rem; color:#C3CBD2; }
      table.lab .flag.hi { color:#A8500F; font-weight:700; }
      table.lab tr.abn td { background:#FDF8F0; }

      /* Findings */
      .fd { border-left:3px solid #B3261E; background:#FCF3F2; padding:0.4rem 0.6rem;
            margin-bottom:0.3rem; font-size:0.78rem; color:#3D1512; }
      .fd.ok { border-left-color:#1E6B45; background:#F1F8F4; color:#14341F; }
      .fd.gone { border-left-color:#A9B4BD; background:#F5F7F8; color:#6B7883; }
      .fd .q { display:block; font-size:0.62rem; text-transform:uppercase;
               letter-spacing:0.07em; color:#8A959E; margin-bottom:0.1rem; }
      .fd s { color:#98A2AA; }

      /* Transcript */
      .tr { margin-bottom:0.3rem; font-size:0.79rem; line-height:1.45;
            display:flex; gap:0.7rem; }
      .tr .w { flex:0 0 5.6rem; font-size:0.62rem; text-transform:uppercase;
               letter-spacing:0.06em; font-weight:700; color:#8A959E; padding-top:0.16rem; }
      .tr.pt .w { color:#0E4C63; }
      .tr .s { color:#26333D; }
      .tr.pt .s { font-weight:500; color:#101C25; }

      /* Sidebar context rail */
      section[data-testid="stSidebar"] { background:#F5F7F8; border-right:1px solid #D8DEE3; }
      section[data-testid="stSidebar"] .block-container { padding-top:1.4rem !important; }
      .rail { font-size:0.74rem; }
      .rail .k { font-size:0.62rem; text-transform:uppercase; letter-spacing:0.07em;
                 color:#8A959E; font-weight:600; margin-top:0.55rem; }
      .rail .v { color:#1A2530; font-variant-numeric:tabular-nums; }
      .rail hr { border:none; border-top:1px solid #D8DEE3; margin:0.7rem 0 0.2rem; }

      .note { border:1px solid #E4CBA6; background:#FDF8F0; padding:0.6rem 0.75rem;
              font-size:0.76rem; color:#4A3A22; margin:0.6rem 0; }
      .cap { font-size:0.72rem; color:#8A959E; margin-top:0.3rem; }
      footer, #MainMenu { visibility:hidden; }
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


def _ehr_name(name: str) -> str:
    """SURNAME, Given — how a patient is listed in a record system."""
    parts = (name or "").split()
    if len(parts) < 2:
        return (name or "Unknown").upper()
    return f"{parts[-1].upper()}, {' '.join(parts[:-1])}"


def _flagged(key: str, raw: float) -> tuple[float, str, str, str]:
    """Value, unit, reference text and H/L flag for one measure."""
    low, high, unit, scale, _ = REFERENCE[key]
    value = raw * scale
    if high is not None:
        return value, unit, f"&lt; {high:g}", "H" if value > high else ""
    if low is not None:
        return value, unit, f"&gt; {low:g}", "L" if value < low else ""
    return value, unit, "—", ""


def results_table(features: dict) -> str:
    rows = []
    for key in REFERENCE:
        if features.get(key) is None:
            continue
        value, unit, ref, flag = _flagged(key, features[key])
        label = REFERENCE[key][4]
        rows.append(
            f'<tr class="{"abn" if flag else ""}">'
            f'<td class="an">{label}</td>'
            f'<td class="num">{value:,.2f}<span class="u"> {unit}</span></td>'
            f'<td class="ref">{ref}</td>'
            f'<td class="flag {"hi" if flag else ""}">{flag or "—"}</td></tr>'
        )
    return (
        '<table class="lab"><thead><tr><th>Measure</th><th class="num">Result</th>'
        '<th>Reference</th><th class="flag">Flag</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def trend_chart(df: pd.DataFrame, key: str) -> alt.LayerChart | alt.Chart:
    low, high, unit, scale, label = REFERENCE[key]
    plot = df[["label", key]].copy()
    plot[key] = plot[key] * scale
    title = f"{label}{f' ({unit})' if unit else ''}"
    line = (
        alt.Chart(plot)
        .mark_line(point=alt.OverlayMarkDef(size=36, filled=True), strokeWidth=1.5)
        .encode(
            x=alt.X("label:N", title=None, sort=None,
                    axis=alt.Axis(labelAngle=0, labelColor="#8A959E", labelFontSize=9,
                                  domainColor="#C3CBD2", tickColor="#C3CBD2")),
            y=alt.Y(f"{key}:Q", title=title, scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelColor="#8A959E", labelFontSize=9,
                                  titleColor="#6B7883", titleFontSize=10,
                                  gridColor="#EDF0F2", domainColor="#C3CBD2")),
            color=alt.value("#0E4C63"),
            tooltip=[alt.Tooltip("label:N", title="Check-in"),
                     alt.Tooltip(f"{key}:Q", title=title, format=".3f")],
        )
        .properties(height=150)
    )
    if low is None and high is None:
        return line
    lo = low if low is not None else float(plot[key].min()) - 1
    hi = high if high is not None else float(plot[key].max()) + 1
    band = (alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]}))
            .mark_rect(opacity=0.08, color="#1E6B45").encode(y="lo:Q", y2="hi:Q"))
    return band + line


data = load_data(DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)

st.markdown(
    '<div class="mast"><span class="m1">Voice Check-in</span>'
    '<span class="m2">Remote monitoring &mdash; Parkinson&rsquo;s disease</span>'
    f'<span class="m3">Data as of {(data.get("generated_at") or "—")[:10]}</span></div>',
    unsafe_allow_html=True,
)

if not data["calls"]:
    st.error("No snapshot found. Run `python -m dashboard.build_snapshot` first.")
    st.stop()

patients = {p["id"]: p for p in data["patients"]}
counts: dict[int, int] = {}
for c in data["calls"]:
    counts[c["patient_id"]] = counts.get(c["patient_id"], 0) + 1

with st.sidebar:
    options = [pid for pid in patients if counts.get(pid)]
    if not options:
        st.error("No patient has any recorded check-ins.")
        st.stop()
    patient_id = st.selectbox(
        "Patient", options,
        format_func=lambda pid: f"{_ehr_name(patients[pid]['name'])} ({counts.get(pid,0)})",
    )

patient = patients[patient_id]
calls = sorted([c for c in data["calls"] if c["patient_id"] == patient_id],
               key=lambda c: c["id"])
scored = [c for c in calls if c["scored"]]
latest = calls[-1]
latest_scored = scored[-1] if scored else None
open_findings = sum(1 for c in calls for f in c["flags"] if not f["withdrawn"])
withdrawn = sum(1 for c in calls for f in c["flags"] if f["withdrawn"])

with st.sidebar:
    times = ", ".join(patient.get("preferred_call_times") or [])
    freq = (patient.get("call_frequency") or "—").replace("_", " ")
    st.markdown(
        f"""
        <div class="rail">
          <hr>
          <div class="k">Identifiers</div>
          <div class="v">MRN VCI-{patient['id']:05d}</div>
          <div class="k">Enrolled</div><div class="v">{(patient.get('enrolled') or '—')[:10]}</div>
          <div class="k">Timezone</div><div class="v">{patient.get('timezone') or '—'}</div>
          <div class="k">Call schedule</div><div class="v">{freq}{f' &middot; {times}' if times else ''}</div>
          <div class="k">Care contact</div><div class="v">{patient.get('caregiver_name') or '—'}</div>
          <hr>
          <div class="k">Check-ins on file</div><div class="v">{len(calls)}</div>
          <div class="k">Yielded measurements</div><div class="v">{len(scored)}</div>
          <div class="k">Findings open</div><div class="v">{open_findings}</div>
          <div class="k">Findings withdrawn</div><div class="v">{withdrawn}</div>
          <hr>
          <div class="cap">Read-only view. Check-ins are placed automatically
          on the patient&rsquo;s schedule.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fields = [
    ("MRN", f"VCI-{patient['id']:05d}"),
    ("TZ", patient.get("timezone") or "—"),
    ("Enrolled", (patient.get("enrolled") or "—")[:10]),
    ("Last check-in", latest["timestamp"][:10]),
    ("Open findings", str(open_findings)),
]
st.markdown(
    f'<div class="pbar"><span class="nm">{html.escape(_ehr_name(patient["name"]))}</span>'
    + "".join(f'<span class="f"><span>{k}</span>{html.escape(v)}</span>' for k, v in fields)
    + "</div>",
    unsafe_allow_html=True,
)

tab_results, tab_history, tab_record = st.tabs(
    ["Results", "Check-in history", "Check-in record"]
)

# --- results -----------------------------------------------------------------
with tab_results:
    if latest_scored is None:
        st.info("No check-in has yielded a usable voice sample yet.")
    else:
        st.markdown(
            f"## Acoustic panel &mdash; check-in #{latest_scored['id']}, "
            f"{latest_scored['timestamp'][:10]}"
        )
        st.markdown(results_table(latest_scored["features"]), unsafe_allow_html=True)
        st.markdown(
            '<div class="cap">Measures taken from the sustained vowel; timing '
            'measures from the reading passage. Reference values are the commonly '
            'cited healthy ranges from the phonation literature &mdash; context for '
            'reading a result, not thresholds validated by this system.</div>',
            unsafe_allow_html=True,
        )

        if len(scored) >= 2:
            rows = []
            for c in scored:
                row = {"label": f"#{c['id']}", "date": c["timestamp"][:10]}
                row.update(c["features"])
                rows.append(row)
            df = pd.DataFrame(rows)
            st.markdown("## Serial measurements")
            cols = st.columns(3)
            for i, key in enumerate(TREND_KEYS):
                with cols[i % 3]:
                    st.altair_chart(trend_chart(df, key), width="stretch")
            st.markdown(
                '<div class="note"><b>Interpret the spread with care.</b> These '
                'measures move substantially between check-ins on the same patient, '
                'and recording conditions &mdash; microphone distance and gain &mdash; '
                'shift shimmer and harmonic-to-noise ratio more than short-term '
                'physiology does. Calibrated capture is required before '
                'check-in-to-check-in comparison carries weight.</div>',
                unsafe_allow_html=True,
            )

# --- history -----------------------------------------------------------------
with tab_history:
    st.markdown("## Check-in history")
    # Built row by row rather than with a conditional expression inside the
    # f-string: a declined check-in has no features, so the branches produce
    # different column counts and the ternary form silently emitted short rows.
    rows = []
    for c in reversed(calls):
        f = c["features"] or {}
        standing = len([x for x in c["flags"] if not x["withdrawn"]])
        gone = len([x for x in c["flags"] if x["withdrawn"]])
        if f:
            jit, shim, hnr = f"{f['jitter_local']*100:.2f}", f"{f['shimmer_local']*100:.2f}", f"{f['hnr']:.1f}"
        else:
            jit = shim = hnr = "—"
        rows.append(
            f'<tr><td class="an">{c["timestamp"][:10]}</td><td class="an">#{c["id"]}</td>'
            f'<td class="an">{"Measured" if c["scored"] else "No usable voice &mdash; not scored"}</td>'
            f'<td class="num">{jit}</td><td class="num">{shim}</td><td class="num">{hnr}</td>'
            f'<td class="flag {"hi" if standing else ""}">{standing or "—"}</td>'
            f'<td class="flag">{gone or "—"}</td></tr>'
        )
    st.markdown(
        '<table class="lab"><thead><tr><th>Date</th><th>Check-in</th><th>Result</th>'
        '<th class="num">Jitter %</th><th class="num">Shimmer %</th><th class="num">HNR dB</th>'
        '<th class="flag">Open</th><th class="flag">W/D</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cap">A check-in that captured no usable voice &mdash; silence, '
        'or the patient hanging up &mdash; is recorded as unscored rather than '
        'estimated. Producing a measurement from silence would place a number in '
        'this patient&rsquo;s record that nothing supports.</div>',
        unsafe_allow_html=True,
    )

# --- record ------------------------------------------------------------------
with tab_record:
    chosen = st.selectbox(
        "Check-in", [c["id"] for c in reversed(calls)],
        format_func=lambda cid: next(
            f"{c['timestamp'][:10]} · #{cid}" for c in calls if c["id"] == cid),
    )
    call = next(c for c in calls if c["id"] == chosen)
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("## Conversation")
        for turn in call["turns"]:
            is_pt = turn["speaker"] != "agent"
            st.markdown(
                f'<div class="tr {"pt" if is_pt else ""}">'
                f'<span class="w">{"Patient" if is_pt else "Agent"}</span>'
                f'<span class="s">{html.escape(turn["text"])}</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("## Findings")
        standing = [f for f in call["flags"] if not f["withdrawn"]]
        gone = [f for f in call["flags"] if f["withdrawn"]]
        if not standing and not gone:
            st.markdown('<div class="fd ok">Nothing flagged during this check-in.</div>',
                        unsafe_allow_html=True)
        for f in standing:
            st.markdown(
                f'<div class="fd"><span class="q">{html.escape(str(f["question_id"]))}</span>'
                f'{html.escape(f["reason"])}</div>', unsafe_allow_html=True)
        for f in gone:
            st.markdown(
                f'<div class="fd gone"><span class="q">{html.escape(str(f["question_id"]))}'
                f' &middot; withdrawn</span><s>{html.escape(f["reason"])}</s><br>'
                f'{html.escape(f["withdrawn_because"] or "")}</div>',
                unsafe_allow_html=True)
        if gone:
            st.markdown(
                '<div class="cap">A finding is withdrawn when something later in the '
                'same check-in disproves it &mdash; a mis-transcribed word, or words '
                'recalled after a cue. The original is kept, not deleted.</div>',
                unsafe_allow_html=True)

        st.markdown("## Tasks completed")
        if call["segments"]:
            seg_rows = "".join(
                f'<tr><td class="an">{TASK_LABELS.get(s["name"], s["name"])}</td>'
                f'<td class="num">{s["duration_s"]:.1f}<span class="u"> s</span></td></tr>'
                for s in call["segments"]
            )
            st.markdown(f'<table class="lab"><tbody>{seg_rows}</tbody></table>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="cap">No task audio captured.</div>',
                        unsafe_allow_html=True)

        if call["scored"]:
            st.markdown("## Measurements")
            st.markdown(results_table(call["features"]), unsafe_allow_html=True)

        if call.get("prediction"):
            st.markdown("## Research model output")
            st.markdown(
                f'<div class="note"><b>UPDRS {call["prediction"]["predicted_score"]}'
                '</b> &mdash; not a clinical measurement and not to be acted on. '
                'This model does not generalise to patients it was not trained on: '
                'held-out R&sup2; is &minus;0.25, where predicting the cohort average '
                'scores &minus;0.06. It is surfaced because it is part of the '
                'pipeline, not because it is reliable. The measurements above are '
                'real; this number is not.</div>',
                unsafe_allow_html=True,
            )

    if call.get("report"):
        with st.expander("Report generated after this check-in"):
            st.code(call["report"], language=None)
