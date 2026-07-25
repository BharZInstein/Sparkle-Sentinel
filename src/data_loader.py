import pandas as pd
from src.config import DATA_PATH


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    before = len(data)
    data = data.drop_duplicates()
    dupes_removed = before - len(data)

    required_cols = ["Amount", "Date", "Time", "Sender_account", "Receiver_account"]
    missing_before = data[required_cols].isnull().sum().sum()
    data = data.dropna(subset=required_cols)

    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")
    data = data.dropna(subset=["Amount"])
    data = data[data["Amount"] > 0]

    for col in ["Sender_bank_location", "Receiver_bank_location", "Payment_currency", "Received_currency", "Payment_type"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.title()

    data = data.reset_index(drop=True)

    print(f"Cleaning report: removed {dupes_removed} duplicate rows, "
          f"{missing_before} missing values in required columns, "
          f"final shape {data.shape}")

    return data


def sample_dataset(df: pd.DataFrame, n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Sample by sender account, keeping every transaction of each sampled
    account. Random row-sampling would scatter per-account behaviour and
    destroy the patterns the detector looks for (a structuring ring's 9
    deposits would lose 8 of them)."""
    if n >= len(df):
        return df
    counts = df["Sender_account"].value_counts()
    shuffled = counts.sample(frac=1, random_state=seed)
    n_accounts = max(1, int((shuffled.cumsum() <= n).sum()) + 1)
    keep = shuffled.index[:n_accounts]
    return df[df["Sender_account"].isin(keep)]