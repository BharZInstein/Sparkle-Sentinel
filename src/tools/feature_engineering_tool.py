import pandas as pd
import numpy as np
from src.config import STRUCTURING_THRESHOLD, FEATURE_WINDOW_HOURS


def engineer_features(df: pd.DataFrame, pattern_type: str = "generic",
                       structuring_threshold: float = STRUCTURING_THRESHOLD,
                       window_hours: int = FEATURE_WINDOW_HOURS) -> pd.DataFrame:
    data = df.copy()
    data["Datetime"] = pd.to_datetime(data["Date"].astype(str) + " " + data["Time"].astype(str), errors="coerce")
    data = data.sort_values("Datetime").reset_index(drop=True)
    data["_row_id"] = data.index  # unique key, safe even with duplicate timestamps

    freq = data.groupby("Sender_account")["Sender_account"].transform("count")
    data["txn_frequency"] = freq

    # rolling sum per sender, computed on a temp indexed copy, then merged back by _row_id
    temp = data.set_index("Datetime")
    rolling_sum = (
        temp.groupby("Sender_account")["Amount"]
        .rolling(f"{window_hours}h")
        .sum()
    )
    rolling_sum = rolling_sum.reset_index()  # columns: Sender_account, Datetime, Amount
    rolling_sum["_row_id"] = temp["_row_id"].values  # align by position, not by index
    rolling_sum = rolling_sum[["_row_id", "Amount"]].rename(columns={"Amount": "rolling_amount_sum"})

    data = data.merge(rolling_sum, on="_row_id", how="left")

    sender_mean = data.groupby("Sender_account")["Amount"].transform("mean")
    sender_std = data.groupby("Sender_account")["Amount"].transform("std").replace(0, np.nan)
    data["amount_zscore"] = ((data["Amount"] - sender_mean) / sender_std).fillna(0)

    data["txn_velocity"] = data.groupby("Sender_account")["Datetime"].transform(
        lambda x: len(x) / max((x.max() - x.min()).total_seconds() / 3600, 1)
    )

    data["near_threshold_flag"] = (
        (data["Amount"] < structuring_threshold) &
        (data["Amount"] >= structuring_threshold * 0.9)
    ).astype(int)

    data["cross_border_flag"] = (
        data["Sender_bank_location"] != data["Receiver_bank_location"]
    ).astype(int)

    # structuring signature: one near-threshold txn is unremarkable, a sender
    # repeating them is the classic pattern — count them per sender
    data["near_threshold_count"] = data.groupby("Sender_account")["near_threshold_flag"].transform("sum")

    # cash-intensity: many cash deposits/withdrawals per sender (cash-out typologies
    # use lots of small cash movements that per-txn stats never notice)
    is_cash = data["Payment_type"].isin(["Cash Deposit", "Cash Withdrawal"]).astype(int)
    data["cash_txn_count"] = is_cash.groupby(data["Sender_account"]).transform("sum")

    # repeated-amount signature: launderers repeat near-identical amounts
    # (fan-in/fan-out layering); bucket amounts and count same-bucket txns per sender
    bucket = (data["Amount"] // 500).astype(int)
    data["repeated_amount_count"] = data.groupby(["Sender_account", bucket])["Amount"].transform("count")

    data = data.drop(columns=["_row_id"])
    return data