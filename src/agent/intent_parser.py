import json
import os
import re
import time

from src.config import GEMINI_MODEL_INTENT

INTENT_SCHEMA_PROMPT = """You are an intent parser for an AML transaction monitoring agent.
Parse the user's natural language query into structured JSON with this exact schema:

{
  "intent": "broad_exploration" | "aggregation_query" | "single_entity_lookup" | "pattern_search",
  "scope": "full_dataset" | "filtered" | "single_entity",
  "date_range": [start_date, end_date] or null,
  "last_n_days": integer or null,
  "entity_id": string or null,
  "pattern_type": "structuring" | "layering" | "smurfing" | "generic" | null,
  "filters": {
    "amount_min": number or null,
    "amount_max": number or null,
    "count_min": integer or null,
    "country": string or null,
    "payment_type": string or null
  },
  "requires_eda": boolean,
  "requires_feature_engineering": boolean,
  "requires_anomaly_detection": boolean,
  "requires_explanation": boolean
}

Rules:
- "Find structuring patterns in the last 30 days" -> intent=pattern_search, pattern_type=structuring, last_n_days=30, scope=filtered, requires_eda=false, requires_feature_engineering=true, requires_anomaly_detection=true, requires_explanation=true
- "Which customers made 10+ transactions under $10,000?" -> intent=aggregation_query, filters.count_min=10, filters.amount_max=10000, requires_eda=false, requires_feature_engineering=false, requires_anomaly_detection=false, requires_explanation=false
- "Is customer ID X suspicious?" -> intent=single_entity_lookup, entity_id="X", scope=single_entity, requires_eda=false, requires_feature_engineering=true, requires_anomaly_detection=true, requires_explanation=true
- Broad/general queries -> intent=broad_exploration, scope=full_dataset, all requires_* = true
- "last N days" phrasing -> set last_n_days, leave date_range null; explicit dates -> date_range

Return ONLY valid JSON, no markdown fences, no commentary."""


class DailyQuotaExhausted(RuntimeError):
    pass


def parse_intent(user_query: str, model: str = GEMINI_MODEL_INTENT, max_retries: int = 3) -> dict:
    """Parse a query with Gemini when available, falling back to the offline
    keyword parser so the agent keeps working with no API key or no quota."""
    if os.getenv("GEMINI_API_KEY"):
        try:
            parsed = _parse_with_gemini(user_query, model, max_retries)
            parsed["parser_used"] = "gemini"
            return parsed
        except Exception as e:
            print(f"Gemini intent parsing unavailable ({e}); using offline parser.")
    parsed = _parse_offline(user_query)
    parsed["parser_used"] = "offline"
    return parsed


def _parse_with_gemini(user_query: str, model: str, max_retries: int) -> dict:
    from google import genai
    from google.genai import types, errors

    client = genai.Client()
    last_error = None
    for _ in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=INTENT_SCHEMA_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except errors.ClientError as e:
            last_error = e
            if e.code == 429:
                detail = str(getattr(e, "details", e)).lower()
                if "perday" in detail:
                    raise DailyQuotaExhausted(f"Daily quota exhausted for '{model}'") from e
                time.sleep(20)
                continue
            raise
    raise RuntimeError(f"Intent parsing failed after retries: {last_error}")


_EMPTY_FILTERS = {"amount_min": None, "amount_max": None, "count_min": None,
                  "country": None, "payment_type": None}

_PATTERN_WORDS = {
    "structuring": ["structur", "under the threshold", "just under", "near threshold"],
    "smurfing": ["smurf", "many small", "fan-in", "fan in"],
    "layering": ["layer", "chain of transfer"],
}


def _parse_offline(query: str) -> dict:
    q = query.lower()
    parsed = {
        "intent": "broad_exploration", "scope": "full_dataset",
        "date_range": None, "last_n_days": None, "entity_id": None,
        "pattern_type": None, "filters": dict(_EMPTY_FILTERS),
        "requires_eda": True, "requires_feature_engineering": True,
        "requires_anomaly_detection": True, "requires_explanation": True,
    }

    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        parsed["last_n_days"] = int(m.group(1))
        parsed["scope"] = "filtered"

    m = re.search(r"(?:customer|account)\s*(?:id\s*)?[#]?(\w+)", q)
    if m and any(w in q for w in ["suspicious", "check", "lookup", "risky", "risk of"]):
        parsed.update({
            "intent": "single_entity_lookup", "scope": "single_entity",
            "entity_id": m.group(1).upper() if m.group(1).isalnum() else m.group(1),
            "requires_eda": False,
        })
        return parsed

    m_count = re.search(r"(\d+)\s*\+?\s*(?:or more\s+)?transactions?", q)
    m_max = re.search(r"under\s+\$?\s*([\d,]+)", q)
    m_min = re.search(r"(?:over|above)\s+\$?\s*([\d,]+)", q)
    if m_count and (m_max or m_min):
        parsed.update({
            "intent": "aggregation_query", "scope": "filtered",
            "requires_eda": False, "requires_feature_engineering": False,
            "requires_anomaly_detection": False, "requires_explanation": False,
        })
        parsed["filters"]["count_min"] = int(m_count.group(1))
        if m_max:
            parsed["filters"]["amount_max"] = float(m_max.group(1).replace(",", ""))
        if m_min:
            parsed["filters"]["amount_min"] = float(m_min.group(1).replace(",", ""))
        return parsed

    for pattern, words in _PATTERN_WORDS.items():
        if any(w in q for w in words):
            parsed.update({
                "intent": "pattern_search", "scope": "filtered",
                "pattern_type": pattern, "requires_eda": False,
            })
            return parsed
    if any(w in q for w in ["flag high-risk", "flag risky", "top suspicious"]):
        parsed.update({"intent": "pattern_search", "scope": "filtered",
                       "pattern_type": "generic", "requires_eda": False})

    return parsed
