from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import DATA_OUTPUTS, save_json


SEGMENT_FEATURES = [
    "success_score",
    "cgpa",
    "attendance_pct",
    "coding",
    "placement_component",
    "engagement_score",
    "lms_logins",
]


def pick_k(x: np.ndarray) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in range(3, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x)
        scores[k] = float(silhouette_score(x, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def segment_name(profile: pd.Series, medians: pd.Series) -> str:
    if profile["success_score"] >= medians["success_score"] and profile["cgpa"] >= medians["cgpa"]:
        return "Strong Performers"
    if profile["cgpa"] < medians["cgpa"] and profile["placement_component"] >= medians["placement_component"]:
        return "Placement-Ready Academic Strugglers"
    if profile["lms_logins"] < medians["lms_logins"] and profile["attendance_pct"] < medians["attendance_pct"]:
        return "Silent Strugglers"
    if profile["coding"] < medians["coding"]:
        return "High Engagement / Low Placement Readiness"
    return "Balanced Improvers"


def main() -> None:
    latest = pd.read_csv(DATA_OUTPUTS / "success_scores_latest.csv")

    df = latest[["student_id", "department", "semester"] + [c for c in SEGMENT_FEATURES if c in latest.columns]].copy()
    feature_cols = [c for c in SEGMENT_FEATURES if c in df.columns]

    scaler = StandardScaler()
    x = scaler.fit_transform(df[feature_cols].fillna(df[feature_cols].median()))

    best_k, silhouette = pick_k(x)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    df["cluster"] = model.fit_predict(x)

    medians = df[feature_cols].median()
    cluster_profiles = df.groupby("cluster")[feature_cols].mean()
    names = {cluster: segment_name(profile, medians) for cluster, profile in cluster_profiles.iterrows()}

    df["segment"] = df["cluster"].map(names)
    df.to_csv(DATA_OUTPUTS / "student_segments.csv", index=False)

    summary = []
    for cluster, grp in df.groupby("cluster"):
        profile = cluster_profiles.loc[cluster].to_dict()
        summary.append(
            {
                "cluster": int(cluster),
                "segment": names[cluster],
                "size": int(len(grp)),
                "percentage": float(len(grp) / len(df) * 100),
                "key_characteristics": {k: round(float(v), 2) for k, v in profile.items()},
                "recommended_playbook": "Target attendance, academics, placement, and mentor actions based on weakest drivers.",
            }
        )

    save_json(
        DATA_OUTPUTS / "segments.json",
        {
            "selected_k": best_k,
            "silhouette_by_k": silhouette,
            "segments": summary,
        },
    )

    print("Segmentation complete")


if __name__ == "__main__":
    main()
