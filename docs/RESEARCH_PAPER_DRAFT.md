# A Reproducible Machine-Learning Pipeline for Heart-Disease Dataset Classification: Model Comparison and Evaluation

**Author(s):** [Insert author names and affiliations]  
**Corresponding author:** [Insert contact]  
**Version:** Working manuscript draft, 27 October 2026

> **Submission status:** This is a manuscript draft, not a publication-ready paper. Verify the dataset source, licence, feature definitions and references; run independent validation; and replace placeholders before submission. Do not claim clinical validation.

## Abstract

Heart-disease classification is a common tabular machine-learning task, but reported performance can be misleading when duplicate records, preprocessing leakage, small test sets, and incomplete reporting are not addressed. This work describes a reproducible pipeline for comparing four supervised classifiers on a supplied heart-disease dataset. The workflow validates the schema, removes exact duplicate rows before splitting, applies preprocessing within scikit-learn pipelines, performs stratified five-fold cross-validation on the training partition, selects a candidate using mean cross-validation ROC-AUC with F1-score as a tie-breaker, and evaluates the selected pipeline on a held-out test partition. The supplied CSV contained 1,025 rows; 723 exact duplicate rows were removed, leaving 302 unique records. A stratified 80:20 split yielded 241 training and 61 test records. In the current fixed split, Logistic Regression was selected, with test accuracy 0.869, precision 0.903, recall 0.848, F1-score 0.875, and ROC-AUC 0.905. These estimates are uncertain because the effective sample is small and the test set contains only 61 observations. Dataset provenance, external validation, calibration, subgroup performance and clinical utility have not been established. The contribution is therefore a reproducible engineering workflow and transparent evaluation, not evidence of clinical readiness.

**Keywords:** heart disease classification; machine learning; Logistic Regression; data leakage; cross-validation; reproducibility; model evaluation

## 1. Introduction

Tabular clinical datasets are frequently used to explore supervised classification. However, model scores alone are insufficient to demonstrate reliability. Duplicate observations can contaminate evaluation splits, preprocessing can leak information across folds, and small test sets can produce unstable estimates. A useful engineering contribution is to make the full pipeline reproducible and to report both its results and limitations.

This study implements a reproducible classification workflow for a supplied heart-disease CSV. Four classifiers are compared using a training-only cross-validation procedure. Preprocessing is encapsulated with the estimator to reduce leakage risk, and the final fitted pipeline is packaged for inference through a web application.

The research question is: **How do four commonly used supervised classifiers compare under a consistent, leakage-conscious pipeline on this supplied dataset?** This is an exploratory computational question, not a claim that the system can diagnose disease.

## 2. Materials and Methods

### 2.1 Dataset and target

The supplied file `heart.csv` contains 1,025 rows, 13 predictors and a binary `target` column. Predictors used by the pipeline are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. The target values are used as supplied, with class 1 treated as the positive dataset label.

**Required before submission:** cite the original dataset source and licence, verify the meanings and coding of all categorical variables, and document any inclusion/exclusion criteria. If the dataset source cannot be verified, that limitation must remain explicit.

### 2.2 Data quality and duplicate handling

The implementation checks that required columns exist and that observed target labels are binary. It removes exact duplicate rows before the train/test split. In the current run, 723 rows were identified as exact duplicates, leaving 302 unique rows. The target is excluded from feature inputs during prediction. Missing target labels are removed; feature-level missing values are handled in the preprocessing pipeline.

Duplicate removal is not equivalent to proving that all observations are independent: near-duplicates, repeated individuals, or source-level grouping may still exist. The dataset provenance should be checked for such relationships.

### 2.3 Split and preprocessing

A stratified 80:20 train/test split was performed with `random_state=42`. This produced 241 training records and 61 held-out test records. Stratification preserves the class proportions approximately in each partition.

Numerical features were processed with median imputation and standard scaling. Categorical codes were processed with most-frequent imputation and one-hot encoding. Both transformations were implemented inside a scikit-learn `Pipeline` and `ColumnTransformer`, so the transformations are fitted on each training fold during cross-validation.

### 2.4 Candidate algorithms

The four candidates were:
1. Logistic Regression (`max_iter=2000`, class-weight balancing enabled);
2. Random Forest (300 trees, minimum leaf size 2, class-weight balancing enabled);
3. Extra Trees (300 trees, minimum leaf size 2, class-weight balancing enabled);
4. Gradient Boosting (120 estimators, learning rate 0.05, maximum depth 2).

The fixed random seeds improve reproducibility. These settings represent a defined comparison, not an exhaustive hyperparameter search.

### 2.5 Model selection and evaluation

A shuffled `StratifiedKFold` with five folds and fixed random seed was used on the training partition. Cross-validation metrics included accuracy, precision, recall, F1-score and ROC-AUC. The candidate with the highest mean cross-validation ROC-AUC was selected, with mean cross-validation F1-score as a tie-breaker. The held-out test partition was reserved for final evaluation and not used for model selection.

Reported test metrics are accuracy, precision, recall, F1-score and ROC-AUC. A confusion matrix and ROC curve are generated. Because the test set is small, the point estimates should be interpreted with substantial uncertainty. Confidence intervals and repeated external validation are recommended for future work.

### 2.6 Software and reproducibility

The workflow is implemented in Python using pandas, scikit-learn, joblib, Matplotlib and Streamlit. The fitted preprocessing/model pipeline is persisted as a model artifact. The application supports single-record inference, batch CSV scoring, dataset exploration and model evaluation. Reproduction requires the same data file, code version and compatible dependency versions.

## 3. Results

### 3.1 Data audit

| Item | Result |
|---|---:|
| Original rows | 1,025 |
| Exact duplicate rows removed | 723 |
| Rows after deduplication | 302 |
| Training rows | 241 |
| Test rows | 61 |

### 3.2 Candidate model comparison

The values below are from the current fixed split and current implementation. They should be regenerated from the exact final code before submission.

| Model | Mean CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.927 | 0.869 | 0.903 | 0.848 | 0.875 | 0.905 |
| Random Forest | 0.914 | 0.787 | 0.812 | 0.788 | 0.800 | 0.876 |
| Extra Trees | 0.912 | 0.820 | 0.806 | 0.879 | 0.841 | 0.878 |
| Gradient Boosting | 0.903 | 0.754 | 0.765 | 0.788 | 0.776 | 0.858 |

Logistic Regression had the highest mean cross-validation ROC-AUC and was selected according to the predefined rule. On the held-out test set, it achieved accuracy 0.869, precision 0.903, recall 0.848, F1-score 0.875 and ROC-AUC 0.905.

### 3.3 Confusion matrix

The current test confusion matrix is:

| | Predicted class 0 | Predicted class 1 |
|---|---:|---:|
| Actual class 0 | 25 | 3 |
| Actual class 1 | 5 | 28 |

This matrix represents 53 correct classifications out of 61 test records. It also shows five class-1 records classified as class 0. The implications of such errors cannot be assessed clinically from this dataset alone.

## 4. Discussion

The current run favours Logistic Regression under the specified cross-validation selection criterion. The result illustrates that a simpler linear classifier can be competitive with tree-based ensembles on a small structured dataset. It does not establish general superiority: the comparison is based on one fixed split and a limited set of hyperparameters.

The duplicate audit materially changes the data volume, from 1,025 raw rows to 302 unique rows. This is important for transparent reporting and suggests that the dataset's construction and provenance deserve careful investigation. The large duplicate fraction may reflect the dataset's collection or publication history; no causal explanation should be asserted without evidence.

The pipeline structure reduces a common source of leakage by fitting imputation, scaling and encoding inside each training fold. Nevertheless, pipeline design alone does not solve all validity concerns. Unknown provenance, possible dependent observations, small sample size, uncertain code definitions, absence of external validation and lack of calibration analysis all limit interpretation.

The Streamlit application makes the trained pipeline usable for individual and batch inference. This improves software reproducibility and demonstrates an end-to-end workflow, but deployment of an interface is not equivalent to clinical deployment or evidence of clinical utility.

## 5. Limitations

1. The effective dataset after exact duplicate removal contains only 302 rows, with 61 rows in the held-out test partition.
2. The original dataset source, licence, collection process and categorical codebook must be verified and cited.
3. Only one train/test split is reported. Confidence intervals, repeated nested cross-validation and independent external validation are absent.
4. Probability calibration, decision-threshold analysis, subgroup performance, fairness and robustness to distribution shift have not been assessed.
5. No clinical review, prospective study or evaluation of patient outcomes has been performed.
6. The output predicts the supplied dataset label; it is not a diagnosis or a calibrated personal probability of disease.

## 6. Future Work

Future work should verify data provenance and code definitions; evaluate on an independent, representative dataset; report uncertainty intervals; assess calibration and clinically appropriate thresholds with domain experts; examine subgroup performance and data shift; and implement monitoring, versioning and auditability. Any use that may affect healthcare decisions requires qualified clinical oversight, privacy and security review, and appropriate regulatory assessment.

## 7. Conclusion

This work presents a reproducible machine-learning workflow for classifying labels in a supplied heart-disease dataset. The pipeline includes schema validation, exact duplicate removal, leakage-conscious preprocessing, cross-validation-based model selection, held-out evaluation, saved model artifacts and a web application for individual and batch inference. Logistic Regression achieved the strongest mean cross-validation ROC-AUC in the current comparison. Because of the small effective sample and lack of external validation, the findings should be considered exploratory and should not be interpreted as evidence of clinical readiness.

## Data and Code Availability

The code and supplied dataset are packaged with the project. Before public release, the dataset's licence and redistribution permissions must be verified. Add the public repository URL, version tag, commit hash and dependency lockfile here after publication.

## Ethics and Conflicts of Interest

No patient recruitment, intervention or new human-subject data collection is described in this computational analysis. Confirm whether institutional ethics review or an exemption is required for the dataset and intended publication. The authors should declare any actual conflicts of interest and funding; do not invent declarations.

## References

**References must be completed and verified before submission.** Cite the original dataset publication/source, scikit-learn documentation or relevant methods papers, and relevant peer-reviewed literature. Do not submit this draft with an empty or fabricated reference list.
