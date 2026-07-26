import time
from google import genai
from google.genai import types
from google.genai import errors
from src.config import GEMINI_MODEL_EXPLANATION

client = genai.Client()


def _build_reason_prompt(row: dict) -> str:
    return f"""Given this flagged AML transaction, write ONE concise sentence (max 30 words) explaining why it was flagged, referencing the specific triggering factors. Be factual, no hedging.

Transaction data:
- Amount: {row.get('Amount')}
- Payment type: {row.get('Payment_type')}
- Sender country: {row.get('Sender_bank_location')}
- Receiver country: {row.get('Receiver_bank_location')}
- Transaction frequency (sender): {row.get('txn_frequency')}
- Rolling amount sum: {row.get('rolling_amount_sum')}
- Amount z-score: {row.get('amount_zscore')}
- Transaction velocity: {row.get('txn_velocity')}
- Near-threshold flag: {row.get('near_threshold_flag')}
- Near-threshold count (sender): {row.get('near_threshold_count')}
- Cash txn count (sender): {row.get('cash_txn_count')}
- Repeated similar-amount txns (sender): {row.get('repeated_amount_count')}
- Cross-border flag: {row.get('cross_border_flag')}
- Risk level: {row.get('risk_level')}

Return only the sentence."""


def explain_flag(row: dict, model: str = GEMINI_MODEL_EXPLANATION, max_retries: int = 3) -> str:
    prompt = _build_reason_prompt(row)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.3),
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return "[Explanation unavailable — empty response]"

        except errors.ClientError as e:
            if e.code == 429:
                time.sleep(15)  # free tier resets quickly; wait it out
                continue
            raise

    return "[Explanation unavailable — rate limited after retries]"


def explain_batch(df, top_n: int = 5) -> list:
    """Explain the top-N flags in ONE Gemini call (a per-flag call with rate-limit
    sleeps took 15-20s per query; batching brings it to ~2s)."""
    import json

    subset = df.sort_values("anomaly_score", ascending=False).head(top_n)
    rows = [r.to_dict() for _, r in subset.iterrows()]

    numbered = "\n\n".join(
        f"FLAG {i + 1}:\n" + _build_reason_prompt(r).split("Transaction data:")[1].rsplit("Return only", 1)[0]
        for i, r in enumerate(rows)
    )
    prompt = (
        f"For each of the {len(rows)} flagged AML transactions below, write ONE concise sentence "
        "(max 30 words) explaining why it was flagged, referencing its specific triggering factors. "
        "Be factual, no hedging. Never mention the flag number; start each sentence with "
        "'This transaction' or similar.\n\n" + numbered +
        '\n\nReturn ONLY a JSON array of strings, one per flag, in order. Example: ["reason 1", "reason 2"]'
    )

    texts = None
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_EXPLANATION,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=1200, temperature=0.3,
                                                   response_mime_type="application/json"),
            )
            texts = json.loads(response.text)
            break
        except errors.ClientError as e:
            if e.code == 429:
                time.sleep(10)
                continue
            raise
        except Exception:
            break

    explanations = []
    for i, r in enumerate(rows):
        explanations.append({
            "Sender_account": r.get("Sender_account"),
            "Receiver_account": r.get("Receiver_account"),
            "Amount": r.get("Amount"),
            "risk_level": r.get("risk_level"),
            "recommended_action": r.get("recommended_action"),
            "explanation": texts[i] if texts and i < len(texts) else explain_flag(r),
        })
    return explanations