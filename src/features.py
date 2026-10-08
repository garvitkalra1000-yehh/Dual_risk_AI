from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import DATA_CLEAN, DATA_OUTPUTS, ensure_dirs

TREND_COLS = [
    "cgpa",
    "attendance_pct",
    "lms_logins",
    "coding",
    "assignment_completion",
    "backlogs",
]


def _slope(values: pd.Series) -> float:
    arr = values.dropna().to_numpy()
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    m = np.polyfit(x, arr, 1)[0]
    return float(m)


def build_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in TREND_COLS:
        out[f"{col}_slope"] = out.groupby("student_id")[col].transform(lambda s: s.expanding().apply(lambda x: _slope(pd.Series(x)), raw=False))
        out[f"{col}_recent_change"] = out.groupby("student_id")[col].transform(lambda s: s - s.shift(1)).fillna(0)
        out[f"{col}_volatility"] = out.groupby("student_id")[col].transform(lambda s: s.expanding().std()).fillna(0)
        out[f"{col}_rolling3"] = out.groupby("student_id")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out[f"{col}_trend_dir"] = np.sign(out[f"{col}_slope"].fillna(0)).astype(int)

    out["attendance_overall"] = out["attendance_pct"]
    out["subject_attendance_overall"] = out["subject_attendance"]
    out["logins_per_week"] = out["lms_logins"] / 4.0

    out["engagement_score"] = (
        out[["events", "clubs", "hackathons", "certifications"]].fillna(0).sum(axis=1)
    )
    out["placement_readiness"] = out[["aptitude", "coding", "mock_interview"]].mean(axis=1)
    out["skills_score"] = out[["technical_skills", "soft_skills"]].mean(axis=1)
    out["feedback_score"] = (out[["satisfaction", "faculty_rating"]].mean(axis=1) / 5) * 100

    for col in [
        "cgpa_missing",
        "attendance_pct_missing",
        "lms_logins_missing",
        "coding_missing",
        "technical_skills_missing",
    ]:
        if col not in out.columns:
            out[col] = 0

    return out


def latest_snapshot(monthly: pd.DataFrame) -> pd.DataFrame:
    idx = monthly.groupby("student_id")["month_idx"].idxmax()
    return monthly.loc[idx].reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    clean_path = DATA_CLEAN / "students_clean_monthly.csv"
    if not clean_path.exists():
        raise FileNotFoundError("Run clean.py first")

    df = pd.read_csv(clean_path)
    monthly = build_monthly_features(df)
    latest = latest_snapshot(monthly)

    monthly.to_csv(DATA_OUTPUTS / "monthly_features.csv", index=False)
    latest.to_csv(DATA_OUTPUTS / "student_features.csv", index=False)
    print(f"Saved feature outputs to {DATA_OUTPUTS}")


if __name__ == "__main__":
    main()
