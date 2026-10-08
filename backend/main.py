from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import InterventionIn
from src.score import compute_score
from src.utils import DATA_OUTPUTS, load_json, save_json

app = FastAPI(title="DualRisk AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INTERVENTIONS_PATH = DATA_OUTPUTS / "interventions.json"


def _load(name: str):
    path = DATA_OUTPUTS / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Output not available: {name}. Run pipeline first.")
    return load_json(path)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/summary")
def get_summary():
    return _load("summary.json")


@app.get("/students")
def get_students(
    department: str | None = None,
    semester: int | None = None,
    risk: str | None = Query(default=None, description="academic|placement"),
    tier: str | None = None,
):
    rows = _load("students.json")
    df = pd.DataFrame(rows)
    if department:
        df = df[df["department"] == department]
    if semester is not None:
        df = df[df["semester"] == semester]
    if risk == "academic":
        df = df[df["academic_risk_flag"] == 1]
    if risk == "placement":
        df = df[df["placement_risk_flag"] == 1]
    if tier:
        df = df[df["score_tier"] == tier]
    return _json_safe_records(df)


@app.get("/students/{student_id}")
def get_student(student_id: str):
    details = _load("student_details.json")
    interventions = _load("interventions.json") if INTERVENTIONS_PATH.exists() else []
    item = next((x for x in details if x["student_id"] == student_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Student not found")
    item["interventions"] = [x for x in interventions if x["student_id"] == student_id]
    return item


@app.get("/worklist")
def get_worklist(
    limit: int = 25,
    department: str | None = None,
    semester: int | None = None,
    risk: str | None = None,
    tier: str | None = None,
    sort_by: str = "priority_score",
    order: str = "desc",
):
    df = pd.DataFrame(_load("worklist.json"))
    if department:
        df = df[df["department"] == department]
    if semester is not None:
        df = df[df["semester"] == semester]
    if risk == "academic":
        df = df[df["academic_risk_flag"] == 1]
    if risk == "placement":
        df = df[df["placement_risk_flag"] == 1]
    if tier:
        df = df[df["score_tier"] == tier]
    asc = order.lower() == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=asc)
    return _json_safe_records(df.head(limit))


@app.get("/segments")
def get_segments():
    return _load("segments_output.json")


@app.get("/metrics")
def get_metrics():
    return {
        "models": _load("risk_metrics.json"),
        "fairness": _load("fairness_output.json"),
    }


@app.get("/data-quality")
def get_data_quality():
    return _load("cleaning_log_output.json")


@app.get("/weights")
def get_weights():
    return _load("weights.json")


@app.post("/weights/recompute")
def recompute_weights(weights: dict):
    monthly = pd.read_csv(DATA_OUTPUTS / "monthly_features.csv")
    rescored = compute_score(monthly, weights)
    latest = rescored.loc[rescored.groupby("student_id")["month_idx"].idxmax()].reset_index(drop=True)

    risk = pd.read_csv(DATA_OUTPUTS / "current_risk_predictions_latest.csv")
    worklist = pd.read_csv(DATA_OUTPUTS / "priority_worklist.csv")

    merged = latest.merge(
        risk[["student_id", "academic_risk_prob", "academic_risk_flag", "placement_risk_prob", "placement_risk_flag"]],
        on="student_id",
        how="left",
    )
    merged = merged.merge(worklist[["student_id", "priority_score", "trend", "recommended_actions"]], on="student_id", how="left")

    save_json(DATA_OUTPUTS / "weights.json", weights)
    save_json(DATA_OUTPUTS / "students.json", merged.to_dict(orient="records"))

    summary = {
        "total_students": int(len(merged)),
        "avg_success_score": round(float(merged["success_score"].mean()), 2),
        "academic_risk_count": int(merged["academic_risk_flag"].fillna(0).sum()),
        "placement_risk_count": int(merged["placement_risk_flag"].fillna(0).sum()),
        "tier_counts": merged["score_tier"].value_counts(dropna=False).to_dict(),
        "department_comparison": merged.groupby("department")["success_score"].mean().round(2).to_dict(),
    }
    save_json(DATA_OUTPUTS / "summary.json", summary)
    return {"message": "Scores recomputed without model retraining", "summary": summary}


@app.post("/interventions")
def start_intervention(intervention: InterventionIn):
    students = {x["student_id"] for x in _load("students.json")}
    if intervention.student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")

    records = _load("interventions.json") if INTERVENTIONS_PATH.exists() else []
    payload = {
        **intervention.model_dump(),
        "start_date": str(date.today()),
        "simulated_30_day_followup": "SIMULATED: Score change after intervention (30 days).",
        "simulated_60_day_followup": "SIMULATED: Score change after intervention (60 days).",
    }
    records.append(payload)
    save_json(INTERVENTIONS_PATH, records)
    return payload
