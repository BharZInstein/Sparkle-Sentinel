import pandas as pd
from src.config import RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD, HIGH_RISK_COUNTRIES


def classify_risk(df: pd.DataFrame,
                   low_threshold: float = RISK_LOW_THRESHOLD,
                   high_threshold: float = RISK_HIGH_THRESHOLD) -> pd.DataFrame:
    data = df.copy()

    def _bucket(score):
        if score >= high_threshold:
            return "High"
        elif score >= low_threshold:
            return "Medium"
        return "Low"

    data["risk_level"] = data["anomaly_score"].apply(_bucket)

    def _apply_override(row):
        if row["risk_level"] == "Medium":
            sender_hr = row.get("Sender_bank_location") in HIGH_RISK_COUNTRIES
            receiver_hr = row.get("Receiver_bank_location") in HIGH_RISK_COUNTRIES
            if sender_hr or receiver_hr:
                return "High"
        return row["risk_level"]

    data["risk_level"] = data.apply(_apply_override, axis=1)

    action_map = {"Low": "monitor", "Medium": "flag for review", "High": "report"}
    data["recommended_action"] = data["risk_level"].map(action_map)

    return data