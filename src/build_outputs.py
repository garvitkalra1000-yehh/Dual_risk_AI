from __future__ import annotations

import json
import ast
from datetime import date

import pandas as pd

from src.utils import DATA_CLEAN, DATA_OUTPUTS, load_json, save_json


def _to_builtin(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError


def main() -> None:
    latest = pd.read_csv(DATA_OUTPUTS / "success_scores_latest.csv")
    monthly_scores = pd.read_csv(DATA_OUTPUTS / "success_scores_monthly.csv")
    risk_latest = pd.read_csv(DATA_OUTPUTS / "current_risk_predictions_latest.csv")
    risk_monthly = pd.read_csv(DATA_OUTPUTS / "current_risk_predictions_monthly.csv")
    worklist = pd.read_csv(DATA_OUTPUTS / "priority_worklist.csv")
    segments = pd.read_csv(DATA_OUTPUTS / "student_segments.csv")
    explanations = load_json(DATA_OUTPUTS / "explanations.json", default=[])
    clean_log = load_json(DATA_CLEAN / "cleaning_log.json", default=[])
    fair = load_json(DATA_OUTPUTS / "fairness.json", default={})

    merged_latest = (
        latest.merge(risk_latest[["student_id", "academic_risk_prob", "academic_risk_flag", "placement_risk_prob", "placement_risk_flag"]], on="student_id", how="left")
        .merge(segments[["student_id", "segment"]], on="student_id", how="left")
    )

    total_students = int(len(merged_latest))
    summary = {
        "total_students": total_students,
        "avg_success_score": round(float(merged_latest["success_score"].mean()), 2),
        "academic_risk_count": int(merged_latest["academic_risk_flag"].fillna(0).sum()),
        "placement_risk_count": int(merged_latest["placement_risk_flag"].fillna(0).sum()),
        "tier_counts": merged_latest["score_tier"].value_counts(dropna=False).to_dict(),
        "department_comparison": merged_latest.groupby("department")["success_score"].mean().round(2).to_dict(),
    }

    students_out = merged_latest.to_dict(orient="records")

    explanations_map = {x["student_id"]: x for x in explanations}
    actions_map = worklist.set_index("student_id")["recommended_actions"].to_dict()

    student_details = []
    for rec in students_out:
        sid = rec["student_id"]
        trends = risk_monthly[risk_monthly["student_id"] == sid].sort_values("month_idx")
        score_tr = monthly_scores[monthly_scores["student_id"] == sid].sort_values("month_idx")
        detail = {
            **rec,
            "trends": {
                "months": trends["month"].tolist(),
                "academic_risk": trends.get("academic_risk_prob", pd.Series(dtype=float)).round(4).tolist(),
                "placement_risk": trends.get("placement_risk_prob", pd.Series(dtype=float)).round(4).tolist(),
                "success_score": score_tr["success_score"].round(2).tolist(),
                "attendance": score_tr.get("attendance_pct", pd.Series(dtype=float)).round(2).tolist(),
                "lms": score_tr.get("lms_logins", pd.Series(dtype=float)).round(2).tolist(),
                "coding": score_tr.get("coding", pd.Series(dtype=float)).round(2).tolist(),
            },
            "explanations": explanations_map.get(sid, {}),
            "recommended_actions": (
                ast.literal_eval(actions_map.get(sid, "[]"))
                if isinstance(actions_map.get(sid, "[]"), str)
                else actions_map.get(sid, [])
            ),
            "interventions": [],
        }
        student_details.append(detail)

    segments_summary = load_json(DATA_OUTPUTS / "segments.json", default={})
    model_info = load_json(DATA_OUTPUTS / "model_info.json", default={})

    save_json(DATA_OUTPUTS / "summary.json", summary)
    save_json(DATA_OUTPUTS / "students.json", students_out)
    save_json(DATA_OUTPUTS / "student_details.json", student_details)
    save_json(DATA_OUTPUTS / "worklist.json", worklist.to_dict(orient="records"))
    save_json(DATA_OUTPUTS / "segments_output.json", segments_summary)
    save_json(DATA_OUTPUTS / "risk_metrics.json", model_info)
    save_json(DATA_OUTPUTS / "fairness_output.json", fair)
    save_json(DATA_OUTPUTS / "cleaning_log_output.json", clean_log)

    weights = pd.read_csv(DATA_OUTPUTS / "weights.csv").to_dict(orient="records")[0]
    save_json(DATA_OUTPUTS / "weights.json", weights)
    save_json(
        DATA_OUTPUTS / "meta.json",
        {
            "generated_on": str(date.today()),
            "disclaimer": "Synthetic dataset. Model performance is illustrative and requires real institutional validation before deployment.",
        },
    )

    print("Frontend/API outputs generated")


if __name__ == "__main__":
    main()
