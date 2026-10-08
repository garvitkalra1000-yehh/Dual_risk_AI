from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import DATA_CLEAN, DATA_RAW, ensure_dirs, save_json

SOURCE_FILES = {
    "academic": "academic.csv",
    "attendance": "attendance.csv",
    "lms": "lms.csv",
    "engagement": "engagement.csv",
    "placement": "placement.csv",
    "skills": "skills.csv",
    "feedback": "feedback.csv",
    "outcomes": "outcomes.csv",
}

DEPT_MAP = {
    "cse": "CSE",
    "computer science": "CSE",
    "comp sci": "CSE",
    "cs": "CSE",
    "ece": "ECE",
    "electronics": "ECE",
    "e.c.e": "ECE",
    "me": "ME",
    "mechanical": "ME",
    "mech": "ME",
    "civil": "Civil",
    "civil engg": "Civil",
    "eee": "EEE",
    "electrical": "EEE",
    "e.e.e": "EEE",
}

RANGES = {
    "cgpa": (0, 10),
    "internal_marks": (0, 100),
    "subject_avg": (0, 100),
    "backlogs": (0, 12),
    "attendance_pct": (0, 100),
    "subject_attendance": (0, 100),
    "lms_logins": (0, 60),
    "assignment_completion": (0, 100),
    "events": (0, 30),
    "clubs": (0, 10),
    "hackathons": (0, 20),
    "certifications": (0, 20),
    "aptitude": (0, 100),
    "coding": (0, 100),
    "mock_interview": (0, 100),
    "technical_skills": (0, 100),
    "soft_skills": (0, 100),
    "satisfaction": (1, 5),
    "faculty_rating": (1, 5),
}


def parse_percentage(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?%", text):
        return float(text.rstrip("%"))
    if re.fullmatch(r"\d+(\.\d+)?/100", text):
        return float(text.split("/")[0])
    try:
        return float(text)
    except ValueError:
        return np.nan


def standardize_department(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(DEPT_MAP).fillna(series)


def clean_source(name: str, file_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(file_path)
    original_rows = len(df)
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    transformations = []
    if "department" in df.columns:
        df["department"] = standardize_department(df["department"])
        transformations.append("standardized department names")

    if "attendance_pct" in df.columns:
        df["attendance_pct"] = df["attendance_pct"].apply(parse_percentage)
        transformations.append("converted attendance percentage strings")

    missing_before = int(df.isna().sum().sum())
    invalid_values = 0

    for col, (low, high) in RANGES.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = (series < low) | (series > high)
            invalid_values += int(invalid_mask.sum())
            series[invalid_mask] = np.nan
            df[col] = series

    for col in df.columns:
        if col in ["student_id", "month", "cohort"]:
            continue
        if df[col].dtype.kind in "biufc":
            miss_flag_col = f"{col}_missing"
            df[miss_flag_col] = df[col].isna().astype(int)
            if df[col].isna().all():
                continue
            med = float(df[col].median())
            df[col] = df[col].fillna(med)

    missing_after = int(df.isna().sum().sum())
    if "month" in df.columns:
        df["month"] = df["month"].astype(str)

    log_entry = {
        "source": name,
        "original_rows": int(original_rows),
        "duplicate_rows": int(dup_rows),
        "missing_values_before": int(missing_before),
        "missing_values_after": int(missing_after),
        "invalid_values": int(invalid_values),
        "rows_removed": int(original_rows - len(df)),
        "transformations": transformations,
    }
    return df, log_entry


def merge_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = sources["academic"].copy()
    for name in ["attendance", "lms", "engagement", "placement", "skills", "feedback"]:
        merged = merged.merge(sources[name], on=["student_id", "month"], how="left", suffixes=("", f"_{name}"))

    outcomes = sources["outcomes"].drop_duplicates(subset=["student_id"])
    keep_cols = [c for c in ["student_id", "cohort", "placed"] if c in outcomes.columns]
    merged = merged.merge(outcomes[keep_cols], on="student_id", how="left")

    source_presence = {}
    for name in ["attendance", "lms", "engagement", "placement", "skills", "feedback"]:
        key_cols = [c for c in sources[name].columns if c not in {"student_id", "month"} and not c.endswith("_missing")]
        if key_cols:
            source_presence[f"missing_from_{name}"] = merged[key_cols].isna().all(axis=1).astype(int)

    for col, values in source_presence.items():
        merged[col] = values

    merged["month_idx"] = pd.PeriodIndex(merged["month"], freq="M").astype(int)
    merged = merged.sort_values(["student_id", "month_idx"]).reset_index(drop=True)
    return merged


def main() -> None:
    ensure_dirs()
    logs: list[dict[str, Any]] = []
    cleaned_sources: dict[str, pd.DataFrame] = {}

    for name, filename in SOURCE_FILES.items():
        path = DATA_RAW / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw source: {path}")
        cleaned_df, log = clean_source(name, path)
        cleaned_sources[name] = cleaned_df
        logs.append(log)
        cleaned_df.to_csv(DATA_CLEAN / f"{name}_clean.csv", index=False)

    merged = merge_sources(cleaned_sources)
    merged.to_csv(DATA_CLEAN / "students_clean_monthly.csv", index=False)

    save_json(DATA_CLEAN / "cleaning_log.json", logs)
    print(f"Saved cleaned data to {DATA_CLEAN}")


if __name__ == "__main__":
    main()
