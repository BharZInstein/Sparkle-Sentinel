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


def explain_batch(df, top_n: int = 5) -> list:  # reduced default from 20
    subset = df.sort_values("anomaly_score", ascending=False).head(top_n)
    explanations = []
    for _, row in subset.iterrows():
        explanations.append({
            "Sender_account": row.get("Sender_account"),
            "Receiver_account": row.get("Receiver_account"),
            "Amount": row.get("Amount"),
            "risk_level": row.get("risk_level"),
            "recommended_action": row.get("recommended_action"),
            "explanation": explain_flag(row.to_dict()),
        })
        time.sleep(2)  # stay under 5 req/min even with intent-parser calls mixed in
    return explanations