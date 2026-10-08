from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import DATA_RAW, ensure_dirs


def _inject_issues(df: pd.DataFrame, numeric_cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        mask = rng.random(len(out)) < 0.04
        out.loc[mask, col] = np.nan
    dup_n = max(5, int(len(out) * 0.02))
    dup_rows = out.sample(n=dup_n, random_state=42)
    out = pd.concat([out, dup_rows], ignore_index=True)
    return out


def main(seed: int = 42) -> None:
    ensure_dirs()
    rng = np.random.default_rng(seed)

    n_hist = 600
    n_current = 800
    months = pd.period_range("2025-01", periods=6, freq="M").astype(str)

    hist_ids = [f"H{i:04d}" for i in range(1, n_hist + 1)]
    current_ids = [f"C{i:04d}" for i in range(1, n_current + 1)]
    student_ids = hist_ids + current_ids

    dept_std = ["CSE", "ECE", "ME", "Civil", "EEE"]
    dept_variants = {
        "CSE": ["CSE", "Computer Science", "cs", "Comp Sci"],
        "ECE": ["ECE", "Electronics", "e.c.e"],
        "ME": ["ME", "Mechanical", "Mech"],
        "Civil": ["Civil", "CIVIL", "Civil Engg"],
        "EEE": ["EEE", "Electrical", "E.E.E"],
    }

    base = pd.DataFrame(
        {
            "student_id": student_ids,
            "cohort": ["historical"] * n_hist + ["current"] * n_current,
            "department_std": rng.choice(dept_std, len(student_ids), p=[0.35, 0.2, 0.2, 0.15, 0.1]),
            "semester": rng.integers(3, 9, len(student_ids)),
        }
    )
    base["department"] = [rng.choice(dept_variants[d]) for d in base["department_std"]]

    records = []
    for month_idx, month in enumerate(months, start=1):
        noise = rng.normal(0, 0.35, len(base))
        cgpa_base = np.clip(6.8 + 0.6 * (base["department_std"] == "CSE") + noise, 4.0, 9.8)
        trend = rng.normal(-0.03, 0.12, len(base)) * month_idx
        backlogs = np.clip(rng.poisson(0.6 + 0.2 * (cgpa_base < 6.5) + 0.15 * month_idx), 0, 6)

        month_df = pd.DataFrame(
            {
                "student_id": base["student_id"],
                "month": month,
                "department": base["department"],
                "semester": base["semester"],
                "cgpa": np.clip(cgpa_base + trend, 3.5, 10),
                "internal_marks": np.clip(55 + cgpa_base * 4.5 + rng.normal(0, 8, len(base)), 20, 100),
                "subject_avg": np.clip(52 + cgpa_base * 4.8 + rng.normal(0, 7, len(base)), 15, 100),
                "backlogs": backlogs,
                "attendance_pct": np.clip(68 + cgpa_base * 2 + rng.normal(0, 12, len(base)), 25, 100),
                "subject_attendance": np.clip(65 + cgpa_base * 2.2 + rng.normal(0, 10, len(base)), 20, 100),
                "lms_logins": np.clip(4 + cgpa_base * 1.5 + rng.normal(0, 3, len(base)), 0, None),
                "assignment_completion": np.clip(45 + cgpa_base * 5 + rng.normal(0, 14, len(base)), 5, 100),
                "events": np.clip(rng.poisson(1.5 + month_idx * 0.1, len(base)), 0, 8),
                "clubs": np.clip(rng.poisson(1.2, len(base)), 0, 4),
                "hackathons": np.clip(rng.poisson(0.6, len(base)), 0, 3),
                "certifications": np.clip(rng.poisson(0.7 + 0.08 * month_idx, len(base)), 0, 6),
                "aptitude": np.clip(40 + cgpa_base * 5 + rng.normal(0, 14, len(base)), 10, 100),
                "coding": np.clip(35 + cgpa_base * 5.3 + rng.normal(0, 16, len(base)), 5, 100),
                "mock_interview": np.clip(38 + cgpa_base * 5 + rng.normal(0, 15, len(base)), 5, 100),
                "technical_skills": np.clip(42 + cgpa_base * 4.8 + rng.normal(0, 12, len(base)), 10, 100),
                "soft_skills": np.clip(45 + cgpa_base * 4.5 + rng.normal(0, 11, len(base)), 10, 100),
                "satisfaction": np.clip(2.2 + (cgpa_base - 5) * 0.45 + rng.normal(0, 0.45, len(base)), 1, 5),
                "faculty_rating": np.clip(2.5 + (cgpa_base - 5) * 0.4 + rng.normal(0, 0.4, len(base)), 1, 5),
            }
        )
        records.append(month_df)

    full = pd.concat(records, ignore_index=True)

    academic = full[["student_id", "month", "department", "semester", "cgpa", "internal_marks", "subject_avg", "backlogs"]].copy()
    attendance = full[["student_id", "month", "attendance_pct", "subject_attendance"]].copy()
    lms = full[["student_id", "month", "lms_logins", "assignment_completion"]].copy()
    engagement = full[["student_id", "month", "events", "clubs", "hackathons", "certifications"]].copy()
    placement = full[["student_id", "month", "aptitude", "coding", "mock_interview"]].copy()
    skills = full[["student_id", "month", "technical_skills", "soft_skills"]].copy()
    feedback = full[["student_id", "month", "satisfaction", "faculty_rating"]].copy()

    # quality issues
    attendance["attendance_pct"] = attendance["attendance_pct"].round(0).astype(int).astype(str) + "%"
    invalid_idx = attendance.sample(frac=0.01, random_state=1).index
    attendance.loc[invalid_idx, "attendance_pct"] = "112/100"

    academic = _inject_issues(academic, ["cgpa", "internal_marks", "subject_avg", "backlogs"], rng)
    lms = _inject_issues(lms, ["lms_logins", "assignment_completion"], rng)
    engagement = _inject_issues(engagement, ["events", "clubs", "hackathons", "certifications"], rng)
    placement = _inject_issues(placement, ["aptitude", "coding", "mock_interview"], rng)
    skills = _inject_issues(skills, ["technical_skills", "soft_skills"], rng)
    feedback = _inject_issues(feedback, ["satisfaction", "faculty_rating"], rng)
    attendance = _inject_issues(attendance, ["subject_attendance"], rng)

    # remove some students from source tables to simulate sparse systems
    attendance = attendance[~attendance["student_id"].isin(rng.choice(student_ids, 45, replace=False))]
    lms = lms[~lms["student_id"].isin(rng.choice(student_ids, 52, replace=False))]

    latest = full[full["month"] == months[-1]][["student_id", "cgpa", "backlogs", "coding", "aptitude", "mock_interview", "technical_skills"]]
    placed_prob = 1 / (1 + np.exp(-(-7 + 0.7 * latest["cgpa"] + 0.02 * latest["coding"] + 0.015 * latest["aptitude"] + 0.01 * latest["technical_skills"])))
    placed = (rng.random(len(latest)) < placed_prob).astype(int)

    outcomes = base[["student_id", "cohort", "department_std", "semester"]].copy()
    outcomes = outcomes.merge(latest[["student_id", "cgpa", "backlogs"]], on="student_id", how="left")
    placed_map = dict(zip(latest["student_id"], placed))
    outcomes["placed"] = outcomes["student_id"].map(placed_map).astype(float)
    outcomes.loc[outcomes["cohort"] != "historical", "placed"] = np.nan

    academic.to_csv(DATA_RAW / "academic.csv", index=False)
    attendance.to_csv(DATA_RAW / "attendance.csv", index=False)
    lms.to_csv(DATA_RAW / "lms.csv", index=False)
    engagement.to_csv(DATA_RAW / "engagement.csv", index=False)
    placement.to_csv(DATA_RAW / "placement.csv", index=False)
    skills.to_csv(DATA_RAW / "skills.csv", index=False)
    feedback.to_csv(DATA_RAW / "feedback.csv", index=False)
    outcomes.to_csv(DATA_RAW / "outcomes.csv", index=False)

    print(f"Generated raw data in {DATA_RAW}")


if __name__ == "__main__":
    main()
