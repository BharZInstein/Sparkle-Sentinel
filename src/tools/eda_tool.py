import pandas as pd


def run_eda(df: pd.DataFrame, mode: str = "full", entity_id: str = None,
            entity_col: str = "Sender_account", date_range: tuple = None) -> dict:
    data = df.copy()

    if date_range:
        start, end = date_range
        data = data[(data["Date"] >= start) & (data["Date"] <= end)]

    if mode == "quick" and entity_id is not None:
        data = data[
            (data[entity_col] == entity_id) |
            (data["Receiver_account"] == entity_id)
        ]

    result = {
        "mode": mode,
        "row_count": len(data),
        "date_range": [str(data["Date"].min()), str(data["Date"].max())] if len(data) else None,
        "amount_stats": {
            "mean": float(data["Amount"].mean()) if len(data) else None,
            "median": float(data["Amount"].median()) if len(data) else None,
            "std": float(data["Amount"].std()) if len(data) else None,
            "max": float(data["Amount"].max()) if len(data) else None,
            "min": float(data["Amount"].min()) if len(data) else None,
        },
        "class_balance": data["Is_laundering"].value_counts(normalize=True).to_dict() if "Is_laundering" in data else None,
        "payment_type_dist": data["Payment_type"].value_counts(normalize=True).to_dict() if "Payment_type" in data else None,
        "top_sender_countries": data["Sender_bank_location"].value_counts().head(5).to_dict() if "Sender_bank_location" in data else None,
        "top_receiver_countries": data["Receiver_bank_location"].value_counts().head(5).to_dict() if "Receiver_bank_location" in data else None,
        "currency_dist": data["Payment_currency"].value_counts(normalize=True).to_dict() if "Payment_currency" in data else None,
        "laundering_type_dist": data["Laundering_type"].value_counts().to_dict() if "Laundering_type" in data else None,
    }
    return result