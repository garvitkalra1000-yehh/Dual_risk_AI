from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import DATA_CLEAN, DATA_OUTPUTS, MODELS_DIR, ensure_dirs, save_json

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


BASE_FEATURES = [
    "cgpa",
    "internal_marks",
    "subject_avg",
    "backlogs",
    "attendance_pct",
    "subject_attendance",
    "lms_logins",
    "assignment_completion",
    "events",
    "clubs",
    "hackathons",
    "certifications",
    "aptitude",
    "coding",
    "mock_interview",
    "technical_skills",
    "soft_skills",
    "satisfaction",
    "faculty_rating",
    "cgpa_slope",
    "attendance_pct_slope",
    "lms_logins_slope",
    "coding_slope",
    "assignment_completion_slope",
    "cgpa_volatility",
    "attendance_pct_volatility",
    "lms_logins_volatility",
    "coding_volatility",
    "cgpa_missing",
    "attendance_pct_missing",
    "lms_logins_missing",
    "coding_missing",
    "technical_skills_missing",
    "department",
    "semester",
]


def pick_threshold(y_true: np.ndarray, probs: np.ndarray, min_recall: float = 0.75) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    best_t, best_p = 0.5, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall and p > best_p:
            best_t, best_p = float(t), float(p)
    return best_t


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "threshold": float(threshold),
    }


def build_targets(clean_monthly: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    df = clean_monthly.copy()
    df = df[df["cohort"] == "historical"].sort_values(["student_id", "month_idx"])

    df["next_cgpa"] = df.groupby("student_id")["cgpa"].shift(-1)
    df["next_backlogs"] = df.groupby("student_id")["backlogs"].shift(-1)
    academic_event = ((df["cgpa"] - df["next_cgpa"]) >= 0.5) | ((df["next_backlogs"] - df["backlogs"]) >= 1)
    df["academic_risk_target"] = academic_event.astype(int)
    df = df[df["next_cgpa"].notna()].copy()

    placement_map = outcomes.set_index("student_id")["placed"].dropna()
    df["placement_risk_target"] = (1 - df["student_id"].map(placement_map)).fillna(0).astype(int)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in BASE_FEATURES if c in df.columns and c not in {"department"}]
    cat_cols = [c for c in ["department"] if c in df.columns]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )


def make_models() -> dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest_fallback": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    return models


def train_task(df: pd.DataFrame, target_col: str, model_path: Path, metrics_path: Path) -> dict:
    X = df[[c for c in BASE_FEATURES if c in df.columns]].copy()
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocessor = build_preprocessor(X)
    candidate_models = make_models()

    model_metrics = {}
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            threshold = pick_threshold(y_test.to_numpy(), probs, min_recall=0.75)
            metrics = evaluate(y_test.to_numpy(), probs, threshold)
            model_metrics[name] = metrics
            fitted_pipelines[name] = pipe
        except Exception:
            continue

    if not model_metrics:
        raise RuntimeError(f"No valid models trained for {target_col}")
    chosen_name = max(model_metrics.keys(), key=lambda k: (model_metrics[k]["recall"], model_metrics[k]["pr_auc"]))
    best_pipeline = fitted_pipelines[chosen_name]

    joblib.dump(best_pipeline, model_path)
    payload = {
        "target": target_col,
        "chosen_model": chosen_name,
        "feature_columns": [c for c in BASE_FEATURES if c in df.columns],
        "metrics_by_model": model_metrics,
    }
    save_json(metrics_path, payload)
    return payload


def predict_current(models_info: dict[str, dict], monthly_features: pd.DataFrame) -> None:
    current = monthly_features[monthly_features["cohort"] == "current"].copy()

    for risk_name, info in models_info.items():
        model = joblib.load(MODELS_DIR / f"{risk_name}_model.joblib")
        feat_cols = info["feature_columns"]
        current[f"{risk_name}_risk_prob"] = model.predict_proba(current[feat_cols])[:, 1]
        threshold = info["metrics_by_model"][info["chosen_model"]]["threshold"]
        current[f"{risk_name}_risk_flag"] = (current[f"{risk_name}_risk_prob"] >= threshold).astype(int)

    current.to_csv(DATA_OUTPUTS / "current_risk_predictions_monthly.csv", index=False)
    idx = current.groupby("student_id")["month_idx"].idxmax()
    current.loc[idx].to_csv(DATA_OUTPUTS / "current_risk_predictions_latest.csv", index=False)


def fairness_report(df: pd.DataFrame, pred_col: str, target_col: str, threshold: float) -> list[dict]:
    rows = []
    for dept, grp in df.groupby("department"):
        y_true = grp[target_col].astype(int).to_numpy()
        y_pred = (grp[pred_col].to_numpy() >= threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        rows.append({"department": dept, "recall": recall, "false_positive_rate": fpr, "students": int(len(grp))})
    return rows


def main() -> None:
    ensure_dirs()
    monthly_features = pd.read_csv(DATA_OUTPUTS / "monthly_features.csv")
    clean_monthly = pd.read_csv(DATA_CLEAN / "students_clean_monthly.csv")
    outcomes = pd.read_csv(DATA_CLEAN / "outcomes_clean.csv")

    training_df = monthly_features.merge(
        build_targets(clean_monthly, outcomes)[["student_id", "month", "academic_risk_target", "placement_risk_target"]],
        on=["student_id", "month"],
        how="inner",
    )

    academic_info = train_task(
        training_df,
        "academic_risk_target",
        MODELS_DIR / "academic_model.joblib",
        DATA_OUTPUTS / "academic_metrics.json",
    )
    placement_info = train_task(
        training_df,
        "placement_risk_target",
        MODELS_DIR / "placement_model.joblib",
        DATA_OUTPUTS / "placement_metrics.json",
    )

    models_info = {"academic": academic_info, "placement": placement_info}
    save_json(DATA_OUTPUTS / "model_info.json", models_info)

    predict_current(models_info, monthly_features)

    eval_hist = training_df.copy()
    for risk_name, info in models_info.items():
        model = joblib.load(MODELS_DIR / f"{risk_name}_model.joblib")
        feat_cols = info["feature_columns"]
        eval_hist[f"{risk_name}_prob"] = model.predict_proba(eval_hist[feat_cols])[:, 1]

    fairness = fairness_report(
        eval_hist,
        pred_col="academic_prob",
        target_col="academic_risk_target",
        threshold=models_info["academic"]["metrics_by_model"][models_info["academic"]["chosen_model"]]["threshold"],
    )
    save_json(DATA_OUTPUTS / "fairness.json", {
        "note": "Capability demonstration on synthetic data; not proof of production fairness.",
        "department_metrics": fairness,
    })

    print("Model training and risk predictions complete")


if __name__ == "__main__":
    main()
