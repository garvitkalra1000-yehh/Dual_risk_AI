import pandas as pd

from src.priority import severity, trend_multiplier
from src.score import compute_score


def test_success_score_confidence_and_tier():
    df = pd.DataFrame(
        [
            {
                "cgpa": 8.0,
                "backlogs": 0,
                "attendance_pct": 90,
                "lms_logins": 20,
                "assignment_completion": 90,
                "aptitude": 80,
                "coding": 75,
                "mock_interview": 78,
                "events": 2,
                "clubs": 1,
                "hackathons": 1,
                "certifications": 2,
                "technical_skills": 82,
                "soft_skills": 79,
                "satisfaction": 4.1,
                "faculty_rating": 4.0,
            }
        ]
    )
    out = compute_score(df)
    assert 0 <= float(out.loc[0, "success_score"]) <= 100
    assert float(out.loc[0, "score_confidence"]) == 100.0
    assert out.loc[0, "score_tier"] in {"Green", "Amber", "Red"}


def test_priority_rule_helpers():
    row = pd.Series({"attendance_pct": 70, "backlogs": 0, "success_score_slope": -0.8})
    assert severity(row) == 1.0
    assert trend_multiplier(row) == 1.2
