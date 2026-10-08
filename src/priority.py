from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import DATA_OUTPUTS


def severity(row: pd.Series) -> float:
    if row.get("attendance_pct", 100) < 75 or row.get("backlogs", 0) >= 2:
        return 1.0
    if row.get("attendance_pct", 100) < 82 or row.get("backlogs", 0) == 1:
        return 0.8
    return 0.6


def trend_multiplier(row: pd.Series) -> float:
    slope = row.get("success_score_slope", 0)
    if slope < -0.4:
        return 1.2
    if slope > 0.2:
        return 0.8
    return 1.0


def action_rules(row: pd.Series) -> list[dict]:
    actions = []
    if row.get("attendance_pct", 100) < 65:
        actions.append({"intervention_type": "Attendance Recovery Plan", "owner": "Faculty", "deadline_days": 7})
    if row.get("cgpa", 10) < 6.5:
        actions.append({"intervention_type": "Academic Mentor", "owner": "Faculty", "deadline_days": 7})
    if row.get("lms_logins", 999) < 4:
        actions.append({"intervention_type": "LMS Engagement Plan", "owner": "Mentor", "deadline_days": 7})
    if row.get("coding", 100) < 50:
        actions.append({"intervention_type": "Coding Improvement Program", "owner": "Placement Cell", "deadline_days": 14})
    if row.get("mock_interview", 100) < 50:
        actions.append({"intervention_type": "Mock Interview Program", "owner": "Placement Cell", "deadline_days": 14})
    return actions


def main() -> None:
    latest = pd.read_csv(DATA_OUTPUTS / "success_scores_latest.csv")
    risk = pd.read_csv(DATA_OUTPUTS / "current_risk_predictions_latest.csv")

    df = latest.merge(
        risk[["student_id", "academic_risk_prob", "academic_risk_flag", "placement_risk_prob", "placement_risk_flag"]],
        on="student_id",
        how="left",
    )

    monthly = pd.read_csv(DATA_OUTPUTS / "success_scores_monthly.csv")
    slope = monthly.sort_values(["student_id", "month_idx"]).groupby("student_id")["success_score"].apply(
        lambda s: np.polyfit(np.arange(len(s)), s.fillna(s.median()), 1)[0] if len(s) > 1 else 0
    )
    df["success_score_slope"] = df["student_id"].map(slope).fillna(0)

    df["combined_risk"] = (df["academic_risk_prob"].fillna(0) + df["placement_risk_prob"].fillna(0)) / 2
    df["severity"] = df.apply(severity, axis=1)
    df["trend_multiplier"] = df.apply(trend_multiplier, axis=1)

    df["priority_score"] = (df["combined_risk"] * 100 * df["severity"] * df["trend_multiplier"]).clip(0, 100)
    df["priority_score"] = df["priority_score"].round(2)
    df["trend"] = np.where(df["success_score_slope"] < -0.4, "Declining", np.where(df["success_score_slope"] > 0.2, "Improving", "Stable"))

    df["recommended_actions"] = df.apply(action_rules, axis=1)
    df = df.sort_values("priority_score", ascending=False)

    df.to_csv(DATA_OUTPUTS / "priority_worklist.csv", index=False)
    print("Priority worklist generated")


if __name__ == "__main__":
    main()
