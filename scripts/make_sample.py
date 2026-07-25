"""Create an account-preserving sample of the full SAML-D dataset.

Reads the 9.5M-row CSV in chunks (never holds it all in memory), picks
random sender accounts, and keeps every transaction of each picked account
until the row budget is reached — preserving the per-account behavioural
patterns the detector relies on.

Usage: python scripts/make_sample.py [--rows 500000]
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SAML-D.csv")
    ap.add_argument("--out", default="data/SAML-D-sample.csv")
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = Counter()
    for chunk in pd.read_csv(args.src, usecols=["Sender_account"], chunksize=2_000_000):
        counts.update(chunk["Sender_account"].value_counts().to_dict())
    print(f"{sum(counts.values()):,} rows, {len(counts):,} sender accounts in source")

    accounts = np.array(list(counts.keys()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(accounts)
    keep, total = set(), 0
    for a in accounts:
        keep.add(a)
        total += counts[a]
        if total >= args.rows:
            break

    first = True
    for chunk in pd.read_csv(args.src, chunksize=2_000_000):
        part = chunk[chunk["Sender_account"].isin(keep)]
        part.to_csv(args.out, mode="w" if first else "a", header=first, index=False)
        first = False
    print(f"kept {len(keep):,} accounts, {total:,} rows -> {args.out}")


if __name__ == "__main__":
    main()
