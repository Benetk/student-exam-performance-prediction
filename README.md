# Student Exam Performance Prediction

A complete end-to-end machine-learning project: dataset preparation, four-model comparison, saved model and metadata, Flask API, web dashboard, automated tests, Docker, Docker Compose, GitHub Actions CI/CD workflow, and documentation.

## Quick start — only commands you need

**Requirements:** Python 3.10/3.11 and Docker Desktop (or Docker Engine with Compose). The dataset preparation step needs internet access once.

### Windows PowerShell

```powershell
cd student_exam_performance_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_dataset.py
python train_model.py
python -m unittest -v
docker compose up --build
```

### macOS / Linux

```bash
cd student_exam_performance_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_dataset.py
python train_model.py
python -m unittest -v
docker compose up --build
```

Then open **http://localhost:5000** in your browser. To stop Docker, press `Ctrl+C`; to run in the background use `docker compose up --build -d`, and stop it with `docker compose down`.

`prepare_dataset.py` downloads the official UCI ZIP and creates `data/student-mat.csv` plus `data/cleaning_report.txt`. `train_model.py` creates `artifacts/model.joblib` and `artifacts/metadata.json`. Those generated files are intentionally created by the commands above and are not required to be pre-created in the ZIP.

## What is included

- `prepare_dataset.py` — downloads and cleans the official Mathematics CSV.
- `train_model.py` — compares mean baseline, Linear Regression, Ridge Regression, and Random Forest; selects by validation MAE; reports held-out test metrics.
- `app.py` and `index.html` — Flask API and interactive dashboard.
- `test_app.py` — API unit tests.
- `Dockerfile`, `docker-compose.yml` — container setup.
- `.github/workflows/ci-cd.yml` — GitHub Actions pipeline that prepares data, trains, tests, builds, and smoke-tests the container.

## Prediction design

Predicts final Mathematics grade `G3` (0–20). `G1` and `G2` are intentionally excluded to avoid grade leakage in an early-prediction scenario. The dashboard reads its input fields and model metrics from generated metadata.

## API endpoints

- `GET /health` — API process status.
- `GET /ready` — verifies model and metadata exist.
- `GET /model-info` — selected model, metrics, data counts, and input schema.
- `POST /predict` — returns a predicted grade for valid input JSON.

## Dataset attribution

UCI Machine Learning Repository, Student Performance dataset. Cortez, P. (2008), DOI: 10.24432/C5TG7T. License: CC BY 4.0. The dataset is downloaded when you run `python prepare_dataset.py`; it is not embedded in this project ZIP.

## Important limitations

This historical dataset comes from two Portuguese secondary schools and is not representative of every school or country. Predictions are estimates for an educational demonstration and should not be used as the sole basis for decisions about students.
