import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "txn_frequency", "rolling_amount_sum", "amount_zscore",
    "txn_velocity", "near_threshold_flag", "near_threshold_count",
    "cash_txn_count", "repeated_amount_count", "cross_border_flag"
]


def detect_anomalies(df: pd.DataFrame, method: str = "hybrid",
                      rule_thresholds: dict = None, contamination: float = 0.01) -> pd.DataFrame:
    data = df.copy()
    thresholds = rule_thresholds or {
        "txn_frequency": 10,
        "amount_zscore": 2.5,
        "txn_velocity": 3,
        "near_threshold_count": 3,   # repeated sub-threshold txns = structuring
        "cash_txn_count": 25,        # cash-heavy accounts (cash-out typologies)
        "repeated_amount_count": 4,  # near-identical amounts repeated = layering
    }

    cross_border_rate = data["cross_border_flag"].mean()
    cross_border_weight = 0.3 if cross_border_rate > 0.4 else 1.0  # down-weight if common

    rule_score = (
        (data["txn_frequency"] > thresholds["txn_frequency"]).astype(int) +
        (data["amount_zscore"].abs() > thresholds["amount_zscore"]).astype(int) +
        (data["txn_velocity"] > thresholds["txn_velocity"]).astype(int) +
        data["near_threshold_flag"] +
        (data["near_threshold_count"] >= thresholds["near_threshold_count"]).astype(int) +
        (data["cash_txn_count"] >= thresholds["cash_txn_count"]).astype(int) +
        (data["repeated_amount_count"] >= thresholds["repeated_amount_count"]).astype(int) +
        data["cross_border_flag"] * cross_border_weight
    )
    max_rule_score = 7 + cross_border_weight
    data["rule_score"] = rule_score
    data["anomaly_score"] = data["rule_score"] / max_rule_score

    if method == "rule":
        return data

    features = data[FEATURE_COLS].fillna(0)
    scaled = StandardScaler().fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(scaled)
    iso_score = -iso.score_samples(scaled)
    data["iso_score"] = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min())

    if method == "ml":
        data["anomaly_score"] = data["iso_score"]
        return data

    data["anomaly_score"] = 0.5 * (data["rule_score"] / max_rule_score) + 0.5 * data["iso_score"]
    return data