"""
Live Data Simulator — ATM Fraud Risk Prediction System
========================================================
Purpose
-------
Your predictive engine (predictive_engine.py) currently runs once on a
static complaints_log.csv and stops. This script simulates a "live" NCRP
complaint feed on top of that same file: it keeps generating new,
schema-correct SYNTHETIC complaint records (same distributions as the
existing 1000 rows — same fraud types, banks, amount ranges, etc.) and
appends them to complaints_log.csv at a fixed interval, then re-runs the
predictive engine so atm_risk_results.json and live_alerts.json update on
their own, just like a real ingestion pipeline would.

Nothing here is real crime data. It's a data-generation harness so you can
demo "real-time" behaviour (rising complaint counts, shifting risk scores,
new alerts appearing) without waiting for real NCRP feeds — useful for a
hackathon demo/judging round. Keep presenting it as simulated/demo data,
per the note already in predictive_engine.py.

How it works
------------
1. On first run, it profiles complaints_log.csv (value counts for
   categorical columns, ranges for numeric columns) so new rows look
   statistically like the real seed data instead of pure random noise.
2. It backs up the original file once (complaints_log_original_backup.csv)
   so you can always reset back to your 1000-row baseline.
3. Every `--interval` seconds it generates `--batch-size` new complaints
   with the current real timestamp, appends them to complaints_log.csv,
   and (unless --no-refresh) re-runs predictive_engine.py as a subprocess
   so the JSON outputs the dashboard reads are refreshed.
4. Ctrl+C stops it cleanly at any time. Nothing is lost — every append is
   flushed to disk immediately.

Usage
-----
    python live_data_simulator.py                       # 1 new complaint every 15s, forever
    python live_data_simulator.py --interval 5 --batch-size 3
    python live_data_simulator.py --cycles 20            # stop after 20 cycles
    python live_data_simulator.py --no-refresh           # just grow the CSV, don't re-run the model
    python live_data_simulator.py --reset                # restore the original 1000-row file and exit

Run it in its own terminal/process alongside (or instead of manually
re-running) predictive_engine.py while you demo the dashboard.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPLAINTS_PATH = os.path.join(SCRIPT_DIR, "complaints_log.csv")
BACKUP_PATH = os.path.join(SCRIPT_DIR, "complaints_log_original_backup.csv")
PREDICTIVE_ENGINE = os.path.join(SCRIPT_DIR, "predictive_engine.py")

CATEGORICAL_COLS = [
    "Fraud_Type", "ATM_ID", "Bank_Operator", "Withdrawal_Status",
    "Complaint_Source", "Case_Status",
]
# Latitude/Longitude are tied to ATM_ID in the seed data (one ATM = one
# coordinate pair), so they're sampled together rather than independently.


def profile_seed_data(df: pd.DataFrame) -> dict:
    """Learn distributions from the existing complaints so new rows are
    statistically consistent with the seed data rather than uniform noise."""
    profile = {}
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts(normalize=True)
        profile[col] = (counts.index.tolist(), counts.values.tolist())

    # ATM_ID -> (Bank_Operator, Latitude, Longitude) lookup so a generated
    # complaint's bank/location stays consistent with its ATM, exactly like
    # the real rows do.
    atm_lookup = (
        df.drop_duplicates("ATM_ID")
        .set_index("ATM_ID")[["Bank_Operator", "Latitude", "Longitude"]]
        .to_dict(orient="index")
    )
    profile["atm_lookup"] = atm_lookup

    profile["amount_min"] = int(df["Transaction_Amount"].min())
    profile["amount_max"] = int(df["Transaction_Amount"].max())
    profile["fund_recovery_choices"] = sorted(df["Fund_Recovery_Pct"].unique().tolist())
    profile["risk_score_min"] = float(df["Risk_Score"].min())
    profile["risk_score_max"] = float(df["Risk_Score"].max())

    ids = df["Complaint_ID"].str.extract(r"(\d+)$")[0].astype(int)
    profile["next_id"] = int(ids.max()) + 1
    profile["id_prefix"] = df["Complaint_ID"].iloc[0].rsplit("-", 1)[0]  # e.g. NCRP-CP-2026
    return profile


def generate_complaint(profile: dict, rng: np.random.Generator, next_id: int) -> dict:
    def pick(col):
        choices, weights = profile[col]
        return rng.choice(choices, p=weights)

    atm_id = pick("ATM_ID")
    atm_info = profile["atm_lookup"][atm_id]

    return {
        "Complaint_ID": f"{profile['id_prefix']}-{next_id:04d}",
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Fraud_Type": pick("Fraud_Type"),
        "Transaction_Amount": int(rng.integers(profile["amount_min"], profile["amount_max"] + 1)),
        "ATM_ID": atm_id,
        "Bank_Operator": atm_info["Bank_Operator"],
        "Latitude": atm_info["Latitude"],
        "Longitude": atm_info["Longitude"],
        "Withdrawal_Status": pick("Withdrawal_Status"),
        "Fund_Recovery_Pct": int(rng.choice(profile["fund_recovery_choices"])),
        "Complaint_Source": pick("Complaint_Source"),
        "Case_Status": pick("Case_Status"),
        "Risk_Score": round(float(rng.uniform(profile["risk_score_min"], profile["risk_score_max"])), 2),
    }


def run_predictive_engine():
    result = subprocess.run(
        [sys.executable, PREDICTIVE_ENGINE],
        cwd=SCRIPT_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  !! predictive_engine.py failed:")
        print(result.stderr[-1500:])
        return False
    # Last couple of lines are the most useful ("Wrote atm_risk_results...")
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    print(f"  engine refreshed → {tail}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Simulate a live complaint feed for the ATM fraud predictive engine.")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between batches (default: 5)")
    ap.add_argument("--batch-size", type=int, default=3, help="New complaints per batch (default: 3)")
    ap.add_argument("--cycles", type=int, default=0, help="Stop after N batches (default: 0 = run forever)")
    ap.add_argument("--no-refresh", action="store_true", help="Only grow the CSV; don't re-run predictive_engine.py each cycle")
    ap.add_argument("--reset", action="store_true", help="Restore the original complaints_log.csv from backup and exit")
    ap.add_argument("--fresh", action="store_true", help="Restore the baseline first, THEN start simulating (recommended before a demo)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: time-based, so each run differs)")
    args = ap.parse_args()

    if args.reset:
        if os.path.exists(BACKUP_PATH):
            pd.read_csv(BACKUP_PATH).to_csv(COMPLAINTS_PATH, index=False)
            print(f"Restored original {len(pd.read_csv(BACKUP_PATH))}-row complaints_log.csv from backup.")
        else:
            print("No backup found yet — nothing to reset (run the simulator at least once first).")
        return

    if not os.path.exists(BACKUP_PATH):
        pd.read_csv(COMPLAINTS_PATH).to_csv(BACKUP_PATH, index=False)
        print(f"Backed up original complaints_log.csv → {os.path.basename(BACKUP_PATH)}")

    if args.fresh:
        baseline = pd.read_csv(BACKUP_PATH)
        baseline.to_csv(COMPLAINTS_PATH, index=False)
        print(f"--fresh: restored the {len(baseline)}-row baseline before starting.\n")

    df = pd.read_csv(COMPLAINTS_PATH)

    # Every past run's rows stay in the file forever. Once the backlog is large,
    # each new batch is a vanishing fraction of the dataset, the aggregate risk
    # scores stop moving, and the dashboard looks like it's stuck on the same
    # data -- even though generation is working fine. Warn before that happens.
    baseline_rows = len(pd.read_csv(BACKUP_PATH))
    if len(df) > 2 * baseline_rows:
        print(
            f"WARNING: complaints_log.csv has grown to {len(df)} rows "
            f"(baseline is {baseline_rows}). Each new complaint is now only "
            f"{100 / len(df):.3f}% of the dataset, so risk scores will barely "
            f"move and the dashboard will look frozen.\n"
            f"         Run:  python live_data_simulator.py --fresh   "
            f"(or --reset) to start from the clean baseline.\n"
        )
    profile = profile_seed_data(df)
    rng = np.random.default_rng(args.seed)

    print(f"Starting live feed simulation: +{args.batch_size} complaint(s) every {args.interval}s "
          f"(currently {len(df)} rows). Press Ctrl+C to stop.\n")

    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            new_rows = []
            for _ in range(args.batch_size):
                row = generate_complaint(profile, rng, profile["next_id"])
                profile["next_id"] += 1
                new_rows.append(row)

            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(COMPLAINTS_PATH, mode="a", header=False, index=False)

            total = profile["next_id"] - 1
            ts = datetime.now().strftime("%H:%M:%S")
            ids = ", ".join(r["Complaint_ID"] for r in new_rows)
            print(f"[{ts}] cycle {cycle}: +{len(new_rows)} complaint(s) ({ids}) → total rows: {total}")

            if not args.no_refresh:
                run_predictive_engine()

            if args.cycles == 0 or cycle < args.cycles:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped. complaints_log.csv keeps every row generated so far "
              f"(run with --reset to restore the original {len(df)}-row baseline).")


if __name__ == "__main__":
    main()
