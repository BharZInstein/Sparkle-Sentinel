import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL_INTENT = "gemini-3.5-flash-lite"
GEMINI_MODEL_EXPLANATION = "gemini-3.5-flash-lite"

DATA_PATH = os.path.join("data", "SAML-D.csv")

STRUCTURING_THRESHOLD = 10000
FEATURE_WINDOW_HOURS = 48

RISK_LOW_THRESHOLD = 0.3
RISK_HIGH_THRESHOLD = 0.65

HIGH_RISK_COUNTRIES = {"Turkey", "Mexico", "Morocco", "Uae", "Iran", "North Korea"}