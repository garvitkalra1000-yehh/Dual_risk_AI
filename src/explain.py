from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from src.utils import DATA_OUTPUTS, MODELS_DIR, load_json, save_json


def _direction(val: float) -> str:
    return "increases risk" if val > 0 else "decreases risk"


def _top_from_vector(feature_names: list[str], vals: np.ndarray, top_n: int = 3) -> list[dict]:
    idx = np.argsort(np.abs(vals))[::-1][:top_n]
    return [
        {"feature": feature_names[i], "impact": float(vals[i]), "direction": _direction(float(vals[i]))}
        for i in idx
    ]


def _fallback_explanations(model, x_df: pd.DataFrame) -> dict[str, list[dict]]:
    model_step = model.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        fi = model_step.feature_importances_
    else:
        fi = np.abs(model_step.coef_).flatten()

    transformed = model.named_steps["prep"].transform(x_df)
    feat_names = model.named_steps["prep"].get_feature_names_out().tolist()

    results = {}
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    for i, sid in enumerate(x_df["student_id"].tolist()):
        impacts = arr[i] * fi
        results[sid] = _top_from_vector(feat_names, impacts)
    return results


def explain_risk(risk_name: str, latest_df: pd.DataFrame) -> tuple[dict, str]:
    import shap

    info = load_json(DATA_OUTPUTS / f"{risk_name}_metrics.json")
    model = joblib.load(MODELS_DIR / f"{risk_name}_model.joblib")
    feature_cols = info["feature_columns"]
    x = latest_df[feature_cols].copy()

    try:
        transformed = model.named_steps["prep"].transform(x)
        feat_names = model.named_steps["prep"].get_feature_names_out().tolist()
        model_step = model.named_steps["model"]

        if hasattr(model_step, "feature_importances_"):
            explainer = shap.TreeExplainer(model_step)
            shap_values = explainer.shap_values(transformed)
            vals = shap_values if isinstance(shap_values, np.ndarray) else shap_values[1]
        else:
            explainer = shap.LinearExplainer(model_step, transformed)
            vals = explainer.shap_values(transformed)

        arr = vals.toarray() if hasattr(vals, "toarray") else vals
        explanations = {
            sid: _top_from_vector(feat_names, arr[idx])
            for idx, sid in enumerate(latest_df["student_id"].tolist())
        }
        return explanations, "shap"
    except Exception:
        fallback = _fallback_explanations(model, pd.concat([latest_df[["student_id"]], x], axis=1))
        return fallback, "fallback_feature_importance"


def main() -> None:
    current = pd.read_csv(DATA_OUTPUTS / "current_risk_predictions_latest.csv")

    academic_explanations, academic_method = explain_risk("academic", current)
    placement_explanations, placement_method = explain_risk("placement", current)

    payload = []
    for sid in current["student_id"]:
        payload.append(
            {
                "student_id": sid,
                "academic_method": academic_method,
                "placement_method": placement_method,
                "academic_top_drivers": academic_explanations.get(sid, []),
                "placement_top_drivers": placement_explanations.get(sid, []),
            }
        )

    save_json(DATA_OUTPUTS / "explanations.json", payload)
    print("Explanations generated")


if __name__ == "__main__":
    main()
