# Model Card — Heart Disease Dataset Classifier

## Intended use
Research and software-engineering demonstration of a tabular binary-classification workflow. The model predicts the label defined by the supplied dataset.

## Out-of-scope use
Not for diagnosis, screening, triage, treatment, insurance, employment, or any decision affecting an individual's health or access to services.

## Data
The bundled `heart.csv` contains 1,025 rows before exact duplicate removal; 723 exact duplicate rows were removed by the current training script, leaving 302 rows. The source provenance and codebook must be verified before stronger claims are made.

## Inputs and output
13 columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal. Output: target label 0 or 1 and model-estimated probability for class 1. The probability is not necessarily calibrated.

## Evaluation
One stratified 80:20 split with random_state=42 and 5-fold stratified cross-validation on the training partition. Current run selected Logistic Regression by mean CV ROC-AUC with CV F1 tie-break. Reported test results are split-specific and uncertain due to the small sample.

## Known limitations
- Small effective sample after duplicate removal.
- Dataset provenance and feature-code semantics need verification.
- No independent external validation, prospective study, calibration evaluation, subgroup fairness assessment, or clinical review.
- No evidence of clinical utility or patient benefit.

## Monitoring and changes
Before any expanded deployment: version the data and model; monitor schema failures, missingness, prediction distributions and performance drift; define retraining approvals; maintain audit logs without storing unnecessary personal data; require independent validation and qualified clinical governance.
