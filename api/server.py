"""FastAPI server: JSON API around the agent + static dashboard."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.chdir(ROOT)  # data paths in config are relative to repo root

from src.data_loader import load_dataset, clean_dataset, sample_dataset
from src.agent.orchestrator import run_agent

app = FastAPI(title="Sparkle Sentinel API")

_df = None
_account_ids = None


def get_df() -> pd.DataFrame:
    global _df, _account_ids
    if _df is None:
        _df = clean_dataset(load_dataset())
        _account_ids = set(_df["Sender_account"].astype(str)) | set(_df["Receiver_account"].astype(str))
    return _df


def jsonable(obj):
    """Recursively convert pandas/numpy values into JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


class QueryRequest(BaseModel):
    query: str
    sample_size: int = 20_000


@app.get("/api/stats")
def stats():
    df = get_df()
    dates = pd.to_datetime(df["Date"])
    daily = df.groupby(df["Date"]).size()
    return jsonable({
        "transactions": len(df),
        "accounts": int(df["Sender_account"].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "cross_border_pct": round(float((df["Sender_bank_location"] != df["Receiver_bank_location"]).mean() * 100), 1),
        "payment_types": int(df["Payment_type"].nunique()),
        "daily_volume": [{"date": d, "count": int(c)} for d, c in daily.items()],
    })


_samples = {}


@app.post("/api/query")
def query(req: QueryRequest):
    import re

    df = get_df()
    if req.sample_size not in _samples:
        _samples[req.sample_size] = sample_dataset(df, req.sample_size)
    working = _samples[req.sample_size]

    # entity lookups run against the full dataset — the queried account may
    # not be in the demo sample, and the entity filter narrows it instantly
    m = re.search(r"(\d{6,})", req.query)
    if m and m.group(1) in _account_ids:
        working = df
    result = run_agent(req.query, working)
    result["working_rows"] = len(working)
    if result.get("flags"):
        levels = [f.get("risk_level") for f in result["flags"]]
        result["risk_breakdown"] = {lv: levels.count(lv) for lv in ["High", "Medium", "Low"]}
    if result.get("aggregation_result"):
        result["aggregation_result"] = result["aggregation_result"][:100]
    return jsonable(result)


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8600)))
