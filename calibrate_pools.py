#!/usr/bin/env python3
"""
calibrate_pools.py
------------------
Reads Freddie Mac Single-Family Loan-Level *sample* files (origination +
performance/servicing) for one or more vintages, classifies each loan's final
outcome from its Zero Balance Code, groups loans into synthetic "pools," and
reports cross-pool dispersion of default and loss rates.

The cross-pool dispersion is what calibrates the common-value distribution V
in the Econ 136 MSR auction simulation.

USAGE
-----
Put this script in the SAME folder as the unzipped files:
    sample_orig_2005.txt   sample_svcg_2005.txt
    sample_orig_2006.txt   sample_svcg_2006.txt
    sample_orig_2007.txt   sample_svcg_2007.txt
    sample_orig_2008.txt   sample_svcg_2008.txt

Then run:
    python calibrate_pools.py

Outputs:
    pool_stats.csv           one row per pool (default rate, loss rate, n loans)
    calibration_summary.txt  the dispersion numbers to drop into the paper
    (also prints the summary to screen)

No third-party libraries required (pure standard library). The big svcg files
are streamed line by line, so memory stays small even for 300 MB files.
"""

import csv
import glob
import os
import re
import statistics
from collections import defaultdict

# ----------------------------------------------------------------------------
# CONFIG -- edit these if you want different vintages or a different pool key.
# ----------------------------------------------------------------------------

VINTAGES = ["2005", "2006", "2007", "2008"]

# How to group loans into "pools." Each distinct value of the chosen key is one
# pool. Options implemented below: "state", "state_year", "credit_bucket",
# "state_credit". Coarser keys -> more loans per pool -> less noisy rates.
POOL_KEY = "state_year"

# Minimum loans for a pool to be reported (drops thin, noisy cells).
MIN_POOL_SIZE = 200

# Zero Balance Codes (svcg file) that indicate a DEFAULT / credit-event
# termination (NOT a voluntary prepayment).
DEFAULT_ZB_CODES = {"02", "03", "09", "15"}
# Prepaid / matured / repurchased-before-default (non-default exits).
PREPAY_ZB_CODES = {"01", "96"}

# ----------------------------------------------------------------------------
# FIELD POSITIONS (0-indexed) in the pipe-delimited files.
# Verified against the uploaded 2005 origination sample (32 fields) and the
# Freddie Mac Standard Dataset file layout.
# ----------------------------------------------------------------------------

# Origination file (sample_orig_YYYY.txt)
ORIG_CREDIT_SCORE = 0      # Credit Score (blank/9999 = unknown)
ORIG_STATE        = 16     # Property State (2-letter)
ORIG_UPB          = 10     # Original UPB (dollars)
ORIG_LOAN_ID      = 19     # Loan Sequence Number (e.g. F05Q10000006)
ORIG_OLTV         = 11     # Original LTV
ORIG_YEAR_FROM_ID = True   # vintage year is encoded in the loan id / filename

# Performance file (sample_svcg_YYYY.txt)
# Indices VERIFIED against the user's 2005 sample (a code-09 defaulted loan):
#   [8]  Zero Balance Code  (e.g. 09)
#   [21] Actual Loss Calculation  (e.g. -19995.67 ; losses stored as negatives)
SVCG_LOAN_ID      = 0      # Loan Sequence Number (join key)
SVCG_ZERO_BAL     = 8      # Zero Balance Code (blank until loan terminates)
SVCG_ACTUAL_LOSS  = 21     # Actual Loss Calculation (negative = loss)
AUTODETECT_LOSS   = False  # index 21 is confirmed; do not auto-detect


def credit_bucket(score_str):
    """Bucket a credit score string into a coarse band."""
    try:
        s = int(score_str)
    except (ValueError, TypeError):
        return "unknown"
    if s <= 0 or s >= 9999:
        return "unknown"
    if s < 620:
        return "sub620"
    if s < 680:
        return "620_679"
    if s < 740:
        return "680_739"
    return "740plus"


def pool_id(rec, vintage):
    """Build the pool key for one loan's origination record."""
    state = rec["state"] or "XX"
    cb = credit_bucket(rec["credit_score"])
    if POOL_KEY == "state":
        return state
    if POOL_KEY == "state_year":
        return f"{state}_{vintage}"
    if POOL_KEY == "credit_bucket":
        return cb
    if POOL_KEY == "state_credit":
        return f"{state}_{cb}"
    return state


def read_orig(path, vintage):
    """Read an origination file into {loan_id: {fields}}."""
    loans = {}
    with open(path, "r", newline="", encoding="latin-1") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) <= ORIG_LOAN_ID:
                continue
            lid = row[ORIG_LOAN_ID].strip()
            if not lid:
                continue
            try:
                upb = float(row[ORIG_UPB]) if row[ORIG_UPB].strip() else 0.0
            except ValueError:
                upb = 0.0
            loans[lid] = {
                "credit_score": row[ORIG_CREDIT_SCORE].strip(),
                "state": row[ORIG_STATE].strip(),
                "upb": upb,
                "vintage": vintage,
            }
    return loans


def detect_loss_col(sample_rows):
    """Heuristically find the Actual Loss column if the default index is off.
    Looks for a column whose nonblank values are mostly numeric and frequently
    negative (losses are reported as negatives in the Freddie layout)."""
    if not sample_rows:
        return SVCG_ACTUAL_LOSS
    ncols = max(len(r) for r in sample_rows)
    best_col, best_score = SVCG_ACTUAL_LOSS, -1
    for c in range(ncols):
        neg = 0
        num = 0
        for r in sample_rows:
            if c < len(r):
                v = r[c].strip()
                if v:
                    try:
                        x = float(v)
                        num += 1
                        if x < 0:
                            neg += 1
                    except ValueError:
                        pass
        # prefer columns that have some negative numeric entries
        score = neg
        if score > best_score and num > 0:
            best_score, best_col = score, c
    # only override if we actually found negatives somewhere
    return best_col if best_score > 0 else SVCG_ACTUAL_LOSS


def read_svcg(path):
    """Stream a performance file. Returns, per loan_id:
       (terminal_zero_balance_code, total_actual_loss)
    Keeps only the last seen ZB code (the terminating one) and the loan's
    Actual Loss Calculation. Streams line-by-line for low memory.

    Loss column is FIXED at SVCG_ACTUAL_LOSS (index 21), confirmed against the
    data. Losses are stored as negatives; we store them as positive magnitudes."""
    loss_col = SVCG_ACTUAL_LOSS

    zb = {}            # loan_id -> last nonblank ZB code
    loss = {}          # loan_id -> actual loss magnitude (last nonzero seen)
    with open(path, "r", newline="", encoding="latin-1") as f:
        for row in csv.reader(f, delimiter="|"):
            if not row:
                continue
            lid = row[SVCG_LOAN_ID].strip()
            if not lid:
                continue
            if len(row) > SVCG_ZERO_BAL:
                code = row[SVCG_ZERO_BAL].strip()
                if code:
                    zb[lid] = code
            if loss_col < len(row):
                v = row[loss_col].strip()
                if v:
                    try:
                        x = float(v)
                        if x != 0.0:
                            # Actual Loss Calculation is the official net figure,
                            # reported once at termination; take its magnitude.
                            loss[lid] = abs(x)
                    except ValueError:
                        pass
    return zb, loss, loss_col


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # loan_id -> origination record (across all vintages)
    orig_all = {}
    # loan_id -> terminal ZB code ; loan_id -> total loss
    zb_all = {}
    loss_all = {}

    for v in VINTAGES:
        op = f"sample_orig_{v}.txt"
        sp = f"sample_svcg_{v}.txt"
        if not os.path.exists(op):
            print(f"[skip] {op} not found")
            continue
        if not os.path.exists(sp):
            print(f"[warn] {sp} not found -- default/loss rates need it. Skipping vintage {v}.")
            continue
        print(f"[read] origination {op} ...")
        o = read_orig(op, v)
        orig_all.update(o)
        print(f"        {len(o):,} loans")
        print(f"[read] performance {sp} (streaming) ...")
        zb, loss, used_col = read_svcg(sp)
        zb_all.update(zb)
        loss_all.update(loss)
        print(f"        {len(zb):,} terminated loans seen; loss column used = index {used_col}")

    if not orig_all:
        print("No origination data read. Make sure the .txt files are in this folder.")
        return

    # Aggregate to pools
    pool_loans = defaultdict(int)
    pool_defaults = defaultdict(int)
    pool_upb = defaultdict(float)
    pool_loss = defaultdict(float)
    zb_tally = defaultdict(int)   # how many loans carry each terminal ZB code

    for lid, rec in orig_all.items():
        p = pool_id(rec, rec["vintage"])
        pool_loans[p] += 1
        pool_upb[p] += rec["upb"]
        code = zb_all.get(lid, "")
        zb_tally[code if code else "(none)"] += 1
        if code in DEFAULT_ZB_CODES:
            pool_defaults[p] += 1
        pool_loss[p] += loss_all.get(lid, 0.0)

    # Build per-pool stats, applying the size filter
    rows = []
    for p in sorted(pool_loans):
        n = pool_loans[p]
        if n < MIN_POOL_SIZE:
            continue
        dr = pool_defaults[p] / n
        lr = (pool_loss[p] / pool_upb[p]) if pool_upb[p] > 0 else 0.0
        rows.append((p, n, dr, lr))

    # Write per-pool CSV
    with open("pool_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "n_loans", "default_rate", "loss_rate"])
        for p, n, dr, lr in rows:
            w.writerow([p, n, f"{dr:.6f}", f"{lr:.6f}"])

    # Cross-pool dispersion summary
    default_rates = [dr for _, _, dr, _ in rows]
    loss_rates = [lr for _, _, _, lr in rows]

    def fmt(xs):
        if not xs:
            return "n/a"
        m = statistics.mean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return (f"mean={m:.4f}  sd={sd:.4f}  min={min(xs):.4f}  "
                f"max={max(xs):.4f}  n_pools={len(xs)}")

    lines = []
    lines.append("FREDDIE MAC SFLLD -- POOL-LEVEL CALIBRATION SUMMARY")
    lines.append(f"Vintages: {', '.join(VINTAGES)}")
    lines.append(f"Pool key: {POOL_KEY}   Min pool size: {MIN_POOL_SIZE}")
    lines.append(f"Pools reported: {len(rows)}")
    lines.append("")
    lines.append("DEFAULT RATE (share of loans with ZB in {02,03,09,15}):")
    lines.append("   " + fmt(default_rates))
    lines.append("")
    lines.append("LOSS RATE (sum actual loss / sum original UPB):")
    lines.append("   " + fmt(loss_rates))
    lines.append("")
    lines.append("USE IN SIMULATION:")
    if default_rates and len(default_rates) > 1:
        lines.append(f"   Common-value V ~ default rate across pools:")
        lines.append(f"     center (mean)      = {statistics.mean(default_rates):.4f}")
        lines.append(f"     dispersion (sd)    = {statistics.pstdev(default_rates):.4f}")
        lines.append(f"   -> set V's distribution to match this mean and sd.")
    lines.append("")
    lines.append("ZERO BALANCE CODE TALLY (terminal code, all loans):")
    for code in sorted(zb_tally, key=lambda c: -zb_tally[c]):
        mark = "  <-- counted as DEFAULT" if code in DEFAULT_ZB_CODES else ""
        lines.append(f"   {code:>7} : {zb_tally[code]:>7,}{mark}")
    summary = "\n".join(lines)

    with open("calibration_summary.txt", "w") as f:
        f.write(summary + "\n")

    print("\n" + summary)
    print("\nWrote pool_stats.csv and calibration_summary.txt")


if __name__ == "__main__":
    main()