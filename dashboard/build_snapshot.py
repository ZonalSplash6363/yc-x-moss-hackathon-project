"""
Build the dashboard's data snapshot — /dashboard zone.

The Streamlit dashboard reads a committed JSON snapshot rather than the live
database, because Streamlit Community Cloud gives an app ephemeral disk and
both data/app.db and calls/ are gitignored. One data path that always works
when deployed beats two paths where the deployed one quietly serves nothing.

Regenerate after new calls:

    python -m dashboard.build_snapshot
    python -m dashboard.build_snapshot --exclude 24 --redact

Nothing here writes to the database; it only reads.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "app.db"
CALLS_DIR = REPO_ROOT / "calls"
OUT_PATH = Path(__file__).resolve().parent / "demo_data.json"

# Test calls said things a public dashboard shouldn't repeat. --redact masks
# them rather than dropping the turn, so the conversation still reads in order.
_PROFANITY = re.compile(r"\b(fuck|shit|dick|cunt|bitch|bastard)\w*", re.I)

FEATURE_COLUMNS = [
    "jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq5",
    "hnr", "rpde", "dfa", "ppe",
    "speech_rate", "pause_freq", "pause_avg_duration",
]


def _redact(text: str) -> str:
    return _PROFANITY.sub("[redacted]", text)


def _patients(conn: sqlite3.Connection) -> list[dict]:
    """Patients with the enrolment details the dashboard header shows.

    The signup zone keeps timezone, schedule and caregiver in its own table,
    so they are joined in here rather than looked up at render time — the
    deployed dashboard has no database to look them up in.
    """
    out = []
    for row in conn.execute("SELECT id, name, created_at FROM patients ORDER BY id"):
        details = conn.execute(
            "SELECT timezone, call_frequency, preferred_call_times, caregiver_name "
            "FROM patient_signup_details WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        times = []
        if details and details["preferred_call_times"]:
            try:
                times = json.loads(details["preferred_call_times"])
            except (ValueError, TypeError):
                times = []
        out.append({
            "id": row["id"],
            "name": row["name"],
            "enrolled": row["created_at"],
            "timezone": details["timezone"] if details else None,
            "call_frequency": details["call_frequency"] if details else None,
            "preferred_call_times": times,
            "caregiver_name": details["caregiver_name"] if details else None,
        })
    return out


def build(exclude: set[int], redact: bool) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    calls = []
    for row in conn.execute("SELECT id, patient_id, timestamp FROM calls ORDER BY id"):
        call_id = row["id"]
        if call_id in exclude:
            continue
        transcript_path = CALLS_DIR / str(call_id) / "transcript.json"
        if not transcript_path.exists():
            continue  # nothing to show for a call with no recording
        transcript = json.loads(transcript_path.read_text())

        feat_row = conn.execute(
            "SELECT * FROM features WHERE call_id = ?", (call_id,)
        ).fetchone()
        features = {c: feat_row[c] for c in FEATURE_COLUMNS} if feat_row else None

        pred_row = conn.execute(
            "SELECT predicted_score, confidence_band, anomaly_flag "
            "FROM updrs_predictions WHERE call_id = ?", (call_id,)
        ).fetchone()
        prediction = dict(pred_row) if pred_row else None

        turns = []
        for turn in transcript.get("turns", []):
            text = turn.get("text", "")
            turns.append({
                "speaker": turn.get("speaker"),
                "text": _redact(text) if redact else text,
            })

        # Flags the call later disproved are kept but marked, so the dashboard
        # can show what was withdrawn instead of hiding the correction.
        flags = [
            {
                "question_id": f.get("question_id"),
                "reason": f.get("reason"),
                "withdrawn": bool(f.get("resolved")),
                "withdrawn_because": f.get("resolved") or None,
            }
            for f in transcript.get("flags", [])
        ]

        report_path = CALLS_DIR / str(call_id) / "report.txt"
        report = report_path.read_text() if report_path.exists() else None
        if report and redact:
            report = _redact(report)

        calls.append({
            "id": call_id,
            "patient_id": row["patient_id"],
            "timestamp": row["timestamp"],
            "scored": features is not None,
            "features": features,
            "prediction": prediction,
            "flags": flags,
            "segments": [
                {"name": s.get("name"), "duration_s": s.get("duration_s")}
                for s in transcript.get("segments", [])
            ],
            "patient_audio_seconds": transcript.get("patient_audio_seconds"),
            "turns": turns,
            "report": report,
        })

    patients = _patients(conn)
    conn.close()
    return {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "patients": patients,
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dashboard data snapshot.")
    parser.add_argument("--exclude", type=int, nargs="*", default=[],
                        help="call ids to leave out entirely")
    parser.add_argument("--redact", action="store_true",
                        help="mask profanity in transcripts and reports")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    snapshot = build(set(args.exclude), args.redact)
    Path(args.out).write_text(json.dumps(snapshot, indent=2))

    scored = sum(1 for c in snapshot["calls"] if c["scored"])
    print(f"wrote {args.out}")
    print(f"  patients: {len(snapshot['patients'])}")
    print(f"  calls:    {len(snapshot['calls'])} ({scored} scored, "
          f"{len(snapshot['calls']) - scored} declined)")
    if args.exclude:
        print(f"  excluded: {sorted(args.exclude)}")


if __name__ == "__main__":
    main()
