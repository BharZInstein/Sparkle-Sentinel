import pandas as pd

from src.tools.eda_tool import run_eda
from src.tools.feature_engineering_tool import engineer_features
from src.tools.anomaly_detection_tool import detect_anomalies
from src.tools.risk_classification_tool import classify_risk
from src.agent.intent_parser import parse_intent
from src.agent.planner import build_execution_plan
from src.utils.formatting import format_final_output


def apply_time_filter(df: pd.DataFrame, last_n_days: int = None, date_range: tuple = None) -> pd.DataFrame:
    data = df.copy()
    dates = pd.to_datetime(data["Date"], errors="coerce")
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        return data[(dates >= start) & (dates <= end)]
    if last_n_days:
        # relative to the newest transaction in the data, not the wall clock —
        # historical datasets would otherwise always filter to nothing
        cutoff = dates.max() - pd.Timedelta(days=last_n_days)
        return data[dates >= cutoff]
    return data


def apply_entity_filter(df: pd.DataFrame, entity_id) -> pd.DataFrame:
    eid = str(entity_id)
    return df[
        (df["Sender_account"].astype(str) == eid) |
        (df["Receiver_account"].astype(str) == eid)
    ]


def run_aggregation_query(df: pd.DataFrame, filters: dict) -> dict:
    data = df.copy()
    if filters.get("amount_max") is not None:
        data = data[data["Amount"] < filters["amount_max"]]
    if filters.get("amount_min") is not None:
        data = data[data["Amount"] > filters["amount_min"]]
    count_min = filters.get("count_min") or 10
    counts = (
        data.groupby("Sender_account")
        .agg(txn_count=("Amount", "size"), total_amount=("Amount", "sum"))
        .reset_index()
    )
    result = counts[counts["txn_count"] >= count_min].sort_values(
        "txn_count", ascending=False)
    return {"aggregation_result": result.to_dict(orient="records"),
            "count_min_applied": count_min}


def _offline_explanations(df: pd.DataFrame, top_n: int) -> list:
    """Template-based explanations used when Gemini is unavailable, built
    from the same feature evidence the LLM would see."""
    out = []
    subset = df.sort_values("anomaly_score", ascending=False).head(top_n)
    for _, row in subset.iterrows():
        reasons = []
        if row.get("near_threshold_flag"):
            reasons.append("amount just under the 10k reporting threshold")
        if abs(row.get("amount_zscore", 0)) > 2.5:
            reasons.append(f"amount {abs(row['amount_zscore']):.1f} std devs from this sender's norm")
        if row.get("txn_frequency", 0) > 10:
            reasons.append(f"{int(row['txn_frequency'])} txns by this sender")
        if row.get("txn_velocity", 0) > 3:
            reasons.append("burst-level transaction velocity")
        if row.get("cross_border_flag"):
            reasons.append(f"cross-border ({row.get('Sender_bank_location')} to {row.get('Receiver_bank_location')})")
        out.append({
            "Sender_account": row.get("Sender_account"),
            "Receiver_account": row.get("Receiver_account"),
            "Amount": row.get("Amount"),
            "risk_level": row.get("risk_level"),
            "recommended_action": row.get("recommended_action"),
            "explanation": "Flagged for: " + "; ".join(reasons) if reasons else "Unusual combination of behavioural features",
        })
    return out


def _explanations(df: pd.DataFrame, top_n: int) -> list:
    try:
        from src.tools.explanation_tool import explain_batch
        return explain_batch(df, top_n=top_n)
    except Exception as e:
        print(f"LLM explanations unavailable ({e}); using template explanations.")
        return _offline_explanations(df, top_n)


def run_agent(user_query: str, df: pd.DataFrame) -> dict:
    parsed_intent = parse_intent(user_query)
    plan = build_execution_plan(parsed_intent)

    context = {"df": df}
    tool_log = []

    for step in plan:
        tool_name = step["tool"]
        params = step.get("params", {})

        if tool_name == "time_filter":
            context["df"] = apply_time_filter(context["df"], **params)
            n = params.get("last_n_days")
            tool_log.append(f"time_filter[last {n} days]" if n else "time_filter")
        elif tool_name == "entity_filter":
            context["df"] = apply_entity_filter(context["df"], params["entity_id"])
            tool_log.append(f"entity_filter[{params['entity_id']}]")
            if context["df"].empty:
                context["error"] = f"No transactions found for account {params['entity_id']}"
                break
        elif tool_name == "eda":
            context["eda_result"] = run_eda(context["df"], **params)
            tool_log.append("eda")
        elif tool_name == "aggregation":
            context.update(run_aggregation_query(context["df"], params["filters"]))
            tool_log.append(f"aggregation[{params['filters'].get('count_min') or 10}+ txns]")
        elif tool_name == "feature_engineering":
            context["df"] = engineer_features(context["df"], **params)
            tool_log.append("feature_engineering")
        elif tool_name == "anomaly_detection":
            context["df"] = detect_anomalies(context["df"], **params)
            tool_log.append(f"anomaly_detection[{params.get('method', 'hybrid')}]")
        elif tool_name == "risk_classification":
            context["df"] = classify_risk(context["df"], **params)
            tool_log.append("risk_classification")
        elif tool_name == "explanation":
            context["explanations"] = _explanations(context["df"], params.get("top_n", 5))
            tool_log.append("explanation")

    output = format_final_output(user_query, parsed_intent, tool_log, context)
    output["parser_used"] = parsed_intent.get("parser_used", "unknown")
    if "error" in context:
        output["error"] = context["error"]
    return output
