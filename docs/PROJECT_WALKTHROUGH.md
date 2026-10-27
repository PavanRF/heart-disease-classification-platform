# Second Review Preparation Guide

## 90-second project explanation
“My project is an Enhanced Heart Disease Prediction System using supervised machine learning. It takes structured features from the supplied heart dataset, validates and preprocesses the data, compares four classification algorithms, evaluates the selected pipeline on a held-out test split, and exposes the model through a Streamlit demo. I use a scikit-learn pipeline so preprocessing is fitted within cross-validation folds, helping reduce leakage. The current result is an academic prototype, not a clinically validated diagnostic system.”

## Rubric mapping (5 × 10 marks)
1. **Technical Architecture & Design:** CSV → validation/deduplication → stratified split → preprocessing pipeline → cross-validation/model selection → held-out metrics → saved model → Streamlit UI.
2. **Implementation & Prototype Development:** show `train_model.py`, artifacts in `models/`, smoke test, and live app.
3. **Tools Integration, Testing, Performance & Validation:** Python, pandas, scikit-learn, joblib, matplotlib, Streamlit; show model comparison, confusion matrix, ROC curve, and smoke-test result.
4. **Industry Readiness & Refinement:** reproducible random seeds, pipeline packaging, input schema checks, downloadable demo output; discuss missing external validation, calibration, subgroup checks and monitoring.
5. **Project Outcomes:** functioning academic prototype and report/presentation artifacts. Do not claim a paper, patent, commercial product or clinical approval unless actually achieved.

## Demo sequence
1. Open `README.md` and explain the folder structure.
2. Run `python train_model.py`.
3. Open `models/model_comparison.csv` and `models/evaluation.json`.
4. Show `models/confusion_matrix.png` and `models/roc_curve.png`.
5. Run `python test_smoke.py`.
6. Run `python -m streamlit run app.py`; demonstrate the four tabs: individual prediction, batch CSV predictions using actual feature rows, dataset explorer, and model performance. Download the CSV template, score records, and download the result.
7. Close with limitations and next steps.

## Likely questions and concise answers

**Why remove duplicate rows?**  
Exact duplicates can cause repeated observations to be split between train and test, making evaluation overly optimistic. I remove them before splitting and report the count transparently.

**Why use a pipeline?**  
The imputer, scaler and encoder are fitted inside each training fold. This helps prevent information from validation/test rows leaking into preprocessing.

**Why compare multiple models?**  
Different algorithms capture different relationships. Comparing cross-validation metrics gives a more robust selection process than choosing a model from one test score.

**Why ROC-AUC as the main selection metric?**  
ROC-AUC evaluates ranking across thresholds. F1 is used as a tie-breaker. In a clinical context, threshold selection would also require carefully evaluating false negatives, false positives, calibration and clinical costs.

**Does 86.9% accuracy mean the model is medically reliable?**  
No. It is performance on a small held-out portion of this dataset. External validation, representative data, clinical review and prospective evaluation would be needed before any medical use.

**What does target=1 mean?**  
The model uses the target labels exactly as supplied in the CSV. The label semantics and categorical feature codes should be confirmed against the dataset's original documentation.

**What is the current project outcome?**  
A reproducible academic prototype with a saved model pipeline, evaluation artifacts and a basic interactive demo. No paper, patent, product release or clinical approval is claimed.

## Final checklist
- [ ] Update presenter and guide names in the slides.
- [ ] Install requirements in a clean environment.
- [ ] Run training and ensure it finishes without errors.
- [ ] Run smoke test and confirm it passes.
- [ ] Launch the Streamlit app and try a sample input.
- [ ] Keep terminal and app ready before the review.
- [ ] Explain limitations honestly; do not call output a diagnosis.
