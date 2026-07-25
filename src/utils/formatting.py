def format_final_output(user_query: str, parsed_intent: dict, tool_log: list, context: dict) -> dict:
    output = {
        "query": user_query,
        "detected_intent": parsed_intent.get("intent"),
        "detected_filters": parsed_intent.get("filters"),
        "detected_pattern": parsed_intent.get("pattern_type"),
        "tools_invoked": tool_log,
    }

    if "eda_result" in context:
        output["eda_summary"] = context["eda_result"]
    if "aggregation_result" in context:
        output["aggregation_result"] = context["aggregation_result"]
    if "explanations" in context:
        output["flags"] = context["explanations"]
        output["flag_count"] = len(context["explanations"])
        output["high_risk_count"] = sum(1 for f in context["explanations"] if f["risk_level"] == "High")

    return output