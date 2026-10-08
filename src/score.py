from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import DATA_OUTPUTS, ensure_dirs

DEFAULT_WEIGHTS = {
    "academic": 0.35,
    "attendance": 0.20,
    "lms": 0.15,
    "placement": 0.15,
    "engagement_skills": 0.10,
    "feedback": 0.05,
}


def _minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if pd.isna(low) or pd.isna(high) or low == high:
        return pd.Series(50.0, index=series.index)
    return ((series - low) / (high - low) * 100).clip(0, 100)


def _score_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cgpa_norm = (out["cgpa"] / 10) * 100
    backlog_component = ((12 - out["backlogs"].clip(0, 12)) / 12) * 100
    out["academic_component"] = 0.7 * cgpa_norm + 0.3 * backlog_component

    out["attendance_component"] = out["attendance_pct"].clip(0, 100)
    out["lms_component"] = _minmax(out["lms_logins"]) * 0.5 + out["assignment_completion"].clip(0, 100) * 0.5
    out["placement_component"] = out[["aptitude", "coding", "mock_interview"]].mean(axis=1).clip(0, 100)

    engagement_raw = out[["events", "clubs", "hackathons", "certifications"]].fillna(0).sum(axis=1)
    engagement_norm = _minmax(engagement_raw)
    skills_norm = out[["technical_skills", "soft_skills"]].mean(axis=1).clip(0, 100)
    out["engagement_skills_component"] = 0.5 * engagement_norm + 0.5 * skills_norm

    out["feedback_component"] = (out[["satisfaction", "faculty_rating"]].mean(axis=1) / 5) * 100

    return out


def compute_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    out = _score_components(df)

    comp_cols = {
        "academic": "academic_component",
        "attendance": "attendance_component",
        "lms": "lms_component",
        "placement": "placement_component",
        "engagement_skills": "engagement_skills_component",
        "feedback": "feedback_component",
    }

    values = []
    confidence = []
    for _, row in out.iterrows():
        available_w = 0.0
        weighted_sum = 0.0
        for key, col in comp_cols.items():
            val = row[col]
            if pd.notna(val):
                weighted_sum += w[key] * val
                available_w += w[key]
        conf = available_w * 100
        confidence.append(conf)
        if available_w == 0:
            values.append(np.nan)
        else:
            values.append(weighted_sum / available_w)

    out["success_score"] = np.round(values, 2)
    out["score_confidence"] = np.round(confidence, 2)
    out["score_status"] = np.where(out["score_confidence"] < 50, "Insufficient data", "Available")

    bins = [-np.inf, 50, 70, np.inf]
    labels = ["Red", "Amber", "Green"]
    out["score_tier"] = pd.cut(out["success_score"], bins=bins, labels=labels, right=False).astype(str)
    out.loc[out["score_status"] == "Insufficient data", "score_tier"] = "N/A"

    return out


def main() -> None:
    ensure_dirs()
    monthly = pd.read_csv(DATA_OUTPUTS / "monthly_features.csv")
    scored_monthly = compute_score(monthly)

    latest_idx = scored_monthly.groupby("student_id")["month_idx"].idxmax()
    latest = scored_monthly.loc[latest_idx].reset_index(drop=True)

    scored_monthly.to_csv(DATA_OUTPUTS / "success_scores_monthly.csv", index=False)
    latest.to_csv(DATA_OUTPUTS / "success_scores_latest.csv", index=False)
    pd.DataFrame([DEFAULT_WEIGHTS]).to_csv(DATA_OUTPUTS / "weights.csv", index=False)

    print("Success scores generated")


if __name__ == "__main__":
    main()
