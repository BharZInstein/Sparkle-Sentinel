"""Generate a synthetic transaction dataset in SAML-D schema with injected
AML patterns, used to evaluate detection quality and demo the agent without
shipping the full Kaggle dataset.

Injected patterns (labelled via Is_laundering / Laundering_type — labels are
for evaluation only, the detector never reads them):
  - Structuring:     repeated cash deposits just under the 10k threshold
  - Smurfing:        many small senders funnelling into one aggregator,
                     followed by a large cross-border transfer out
  - Rapid_Movement:  high-velocity in/out burst within hours

Usage: python scripts/generate_synthetic.py [--rows 20000] [--out data/SAML-D.csv]
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LOCATIONS = ["India", "India", "India", "India", "Uk", "Usa", "Singapore", "Uae", "Turkey"]
CURRENCIES = {"India": "Inr", "Uk": "Gbp", "Usa": "Usd", "Singapore": "Sgd",
              "Uae": "Dirham", "Turkey": "Lira"}
PAYMENT_TYPES = ["Credit Card", "Debit Card", "Cheque", "Ach", "Cash Deposit", "Cash Withdrawal"]

START = datetime(2026, 6, 1)
DAYS = 45


def _row(ts, sender, receiver, amount, ptype, s_loc, r_loc, laundering, ltype):
    return {
        "Time": ts.strftime("%H:%M:%S"),
        "Date": ts.strftime("%Y-%m-%d"),
        "Sender_account": sender,
        "Receiver_account": receiver,
        "Amount": round(amount, 2),
        "Payment_currency": CURRENCIES[s_loc],
        "Received_currency": CURRENCIES[r_loc],
        "Sender_bank_location": s_loc,
        "Receiver_bank_location": r_loc,
        "Payment_type": ptype,
        "Is_laundering": laundering,
        "Laundering_type": ltype,
    }


def _normal(rng, n_accounts, n_rows):
    accounts = rng.integers(10_000_000, 99_999_999, size=n_accounts)
    rows = []
    for _ in range(n_rows):
        s, r = rng.choice(accounts, 2, replace=False)
        loc = rng.choice(LOCATIONS)
        r_loc = rng.choice(LOCATIONS) if rng.random() < 0.15 else loc
        ts = START + timedelta(seconds=float(rng.uniform(0, DAYS * 86400)))
        rows.append(_row(ts, s, r, float(rng.lognormal(6.8, 1.0)),
                         rng.choice(PAYMENT_TYPES), loc, r_loc, 0, "Normal"))
    return rows, accounts


def _structuring(rng, ring_id, n_deposits=9):
    acct = 90_000_100 + ring_id
    cash_src = 90_000_900 + ring_id
    t0 = START + timedelta(days=float(rng.uniform(5, DAYS - 8)))
    rows = []
    for i in range(n_deposits):
        ts = t0 + timedelta(hours=float(rng.uniform(6, 18)) * (i + 1))
        rows.append(_row(ts, acct, cash_src, float(rng.uniform(9100, 9900)),
                         "Cash Deposit", "India", "India", 1, "Structuring"))
    return rows


def _smurfing(rng, cluster_id, n_smurfs=12):
    aggregator = 91_000_100 + cluster_id
    t0 = START + timedelta(days=float(rng.uniform(5, DAYS - 5)))
    rows, total = [], 0.0
    for s in range(n_smurfs):
        amt = float(rng.uniform(1500, 4500))
        total += amt
        ts = t0 + timedelta(hours=float(rng.uniform(0, 36)))
        rows.append(_row(ts, 91_000_500 + cluster_id * 100 + s, aggregator, amt,
                         "Ach", "India", "India", 1, "Smurfing"))
    rows.append(_row(t0 + timedelta(hours=40), aggregator, 91_000_999,
                     total * 0.97, "Ach", "India", "Uae", 1, "Smurfing"))
    return rows


def _rapid_movement(rng, burst_id, n_txns=15):
    acct = 92_000_100 + burst_id
    t0 = START + timedelta(days=float(rng.uniform(5, DAYS - 5)))
    rows = []
    for i in range(n_txns):
        ts = t0 + timedelta(minutes=float(rng.uniform(5, 18)) * (i + 1))
        r_loc = "Turkey" if i % 3 == 0 else "India"
        rows.append(_row(ts, acct, 92_000_500 + i, float(rng.uniform(3000, 22000)),
                         "Ach", "India", r_loc, 1, "Rapid_Movement"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--accounts", type=int, default=600)
    ap.add_argument("--out", default="data/SAML-D.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows, _ = _normal(rng, args.accounts, args.rows)
    for i in range(6):
        rows += _structuring(rng, i)
    for i in range(4):
        rows += _smurfing(rng, i)
    for i in range(4):
        rows += _rapid_movement(rng, i)

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows -> {args.out}")
    print(df["Laundering_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
