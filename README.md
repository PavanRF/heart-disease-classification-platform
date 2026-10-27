# Heart Disease Dataset Classification Platform

A Python machine-learning platform for structured heart-disease dataset classification. Includes reproducible training, cross-validation-based model comparison, held-out evaluation, persisted inference pipeline, Streamlit UI, batch CSV scoring, dataset exploration, Docker packaging, CI checks, model card and a research manuscript draft.

> **Important limitation:** This is not clinically validated software. It predicts labels in the supplied dataset and must not be used for diagnosis, triage, treatment, or other health-impacting decisions.

## Capabilities
- Dataset schema and target validation
- Exact duplicate audit and removal before splitting
- Preprocessing inside scikit-learn pipelines to reduce cross-validation leakage
- Comparison of Logistic Regression, Random Forest, Extra Trees and Gradient Boosting
- Stratified 5-fold CV on the training partition and a held-out test evaluation
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curve
- Saved model pipeline and evaluation artifacts
- Individual-record prediction and batch CSV scoring
- Dataset explorer and downloadable output files
- Dockerfile and GitHub Actions CI workflow
- Model card, deployment instructions and a research manuscript draft

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python test_smoke.py
python -m streamlit run app.py
```

## Docker
```bash
docker build -t heart-disease-classifier:0.1.0 .
docker run --rm -p 8501:8501 heart-disease-classifier:0.1.0
```
Then open `http://localhost:8501`.

## Project files
- `train_model.py` — training, model comparison and evaluation
- `app.py` — individual inference, batch CSV scoring, dataset explorer, evaluation UI
- `models/` — saved pipeline and evaluation artifacts
- `tests/` — model/data contract tests
- `docs/RESEARCH_PAPER_DRAFT.md` — working manuscript, not yet publication-ready
- `MODEL_CARD.md` — intended use, limits and evaluation
- `DEPLOYMENT.md` — local, Docker and publishing instructions

## Reproducibility
Use the exact dataset and code version, record dependency versions and retain generated evaluation artifacts. Current results are from one fixed stratified 80:20 split, with five-fold cross-validation on the training set. The effective dataset contains 302 rows after removal of 723 exact duplicates from 1,025 raw rows.

## Research and deployment caveats
The original dataset provenance, licence and feature-code definitions must be verified. The small effective dataset and lack of independent external validation, calibration, subgroup evaluation and clinical review prevent claims of clinical reliability. Read `MODEL_CARD.md` and `DEPLOYMENT.md` before publishing or hosting.
