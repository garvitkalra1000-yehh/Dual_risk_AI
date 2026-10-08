# DualRisk AI – Smart Campus Student Success Platform

DualRisk AI is a full-stack synthetic-data demo for identifying student success, academic risk, placement risk, intervention priority, and action playbooks.

> **Synthetic data disclaimer:** The current dataset is synthetic and model performance is illustrative. Real institutional validation is required before deployment.

## Problem Statement
Campus stakeholders need early-warning visibility, clear risk reasoning, and actionable intervention prioritization.

## USP
Most dashboards only show who is at risk. DualRisk AI shows who to help first, why they are at risk, recommended interventions, and simulated follow-up tracking.

## Architecture
- **Pipeline scripts:** `src/`
- **Backend API:** FastAPI in `backend/`
- **Frontend UI:** React + Vite in `frontend/`
- **Data:** `data/raw`, `data/clean`, `data/outputs`
- **Models:** `models/`

## Technology Stack
- Python, Pandas, NumPy, scikit-learn, XGBoost (or RandomForest fallback), SHAP, FastAPI
- React, Vite, Recharts, Axios, React Router

## Folder Structure
```
backend/
frontend/
data/raw data/clean data/outputs
models/
src/
tests/
requirements.txt
README.md
```

## Dataset Description
Pipeline generates 1400 students (600 historical + 800 current), 6 monthly snapshots, 8 source tables, and intentional quality issues:
- duplicates
- missing values
- invalid values
- department naming variations
- percentage strings and malformed values
- missing students in source systems

## Data Cleaning (`src/clean.py`)
- Loads all raw sources
- Removes duplicates
- Standardizes department names
- Converts percentage strings to numeric
- Flags invalid/out-of-range values and imputes with medians
- Preserves missing indicators
- Merges multi-source monthly student records
- Saves reproducible cleaning outputs and logs

## Feature Engineering (`src/features.py`)
Builds academic, attendance, LMS, engagement, placement, skills, feedback, and temporal features:
- slopes, recent change, volatility, rolling averages, trend direction
- missing-data flags

## Success Score (`src/score.py`)
Transparent weighted score (0–100):
- Academic 35%
- Attendance 20%
- LMS 15%
- Placement 15%
- Engagement+Skills 10%
- Feedback 5%

Missing components are reweighted and confidence is calculated. If confidence < 50%, status is **Insufficient data**.

## Academic Risk Model (`src/train.py`)
Target: CGPA drop >= 0.5 or new backlog next month (historical cohort only).
- Baseline: Logistic Regression
- Primary: XGBoost (RandomForest fallback)
- Metrics: ROC-AUC, PR-AUC, precision, recall, confusion matrix
- Threshold selected with recall-prioritized logic

## Placement Risk Model (`src/train.py`)
Separate model using placement outcome for historical students with relevant academic/placement/skills/engagement features.

## Explainable AI (`src/explain.py`)
Per-student top 3 academic and placement risk drivers via SHAP where possible; fallback to model-importance contribution with explicit method label.

## Priority Score (`src/priority.py`)
`Priority = Risk × Severity × TrendMultiplier` (capped 0–100)
- Severity based on attendance/backlogs
- Trend multiplier from success trend
- Deterministic intervention playbook rules

## Segmentation (`src/segment.py`)
- KMeans (K=3..6) with silhouette-based K selection
- Segment profiles and action playbooks

## Fairness Analysis
Department-level recall and false-positive rate generated from synthetic evaluation output and labeled as demonstration-only.

## Output Builder (`src/build_outputs.py`)
Builds precomputed frontend/backend payloads:
- `summary.json`, `students.json`, `student_details.json`, `worklist.json`
- `segments_output.json`, `risk_metrics.json`, `fairness_output.json`
- `cleaning_log_output.json`, `weights.json`

Frontend consumes outputs via API and does not retrain models.

## API Endpoints
- `GET /summary`
- `GET /students`
- `GET /students/{student_id}`
- `GET /worklist`
- `GET /segments`
- `GET /metrics`
- `GET /data-quality`
- `GET /weights`
- `POST /weights/recompute` (recalculate success score only)
- `POST /interventions` (simulated intervention tracking)

## Frontend Pages
- Overview
- Worklist
- Student Details
- Segments
- Admin / Model & Data Quality

## Leakage Controls
- Historical outcomes are used only for target creation/training.
- Current-student future outcomes are not used as model features.
- Scoring and prediction for current students are done from current-feature snapshots.

## Local Setup
### 1) Python setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Frontend setup
```bash
cd frontend
npm install
cd ..
```

## Run Data Pipeline
```bash
python -m src.run_pipeline
```

## Run Backend
```bash
uvicorn backend.main:app --reload
```
Backend URL: `http://127.0.0.1:8000`

## Run Frontend
```bash
cd frontend
npm run dev
```
Frontend URL: `http://127.0.0.1:5173`

## Testing
```bash
pytest -q
```

## Limitations
- Synthetic data only
- Intervention outcomes are simulated and do not imply causality
- Fairness outputs are demonstration metrics, not production fairness assurance

## Future Improvements
- Institutional data integrations
- stronger calibration and drift tracking
- richer intervention outcome capture and experimental evaluation
