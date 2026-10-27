"""Interactive educational demo. Run with: streamlit run app.py"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "evaluation.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"

st.set_page_config(page_title="Heart Disease Prediction | Academic Prototype", page_icon="🫀", layout="wide")
st.title("🫀 Enhanced Heart Disease Prediction")
st.caption("Heart Disease Risk Classification • Machine Learning Application")

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("Model artifacts are not available yet. Open a terminal in the project folder and run `python train_model.py`.")
    st.stop()

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

model = load_pipeline(str(MODEL_PATH))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Project overview")
    st.write("A trained scikit-learn pipeline processes the inputs and returns the model's dataset-class prediction and estimated probability.")
    st.markdown("**Model selection**")
    st.write("The model is selected using mean 5-fold cross-validation ROC-AUC on the training split, with CV F1 as a tie-breaker.")
    st.warning("This prototype is not clinically validated. Do not use it to diagnose, treat, or rule out heart disease. Seek advice from a qualified healthcare professional.")

st.subheader("Prediction demo")
st.write("Enter values using the codes and ranges in the supplied dataset. For categorical clinical codes, confirm the dataset's codebook before interpreting them.")
with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 18, 100, 52)
        sex = st.selectbox("Sex (dataset code)", [1, 0], format_func=lambda x: "Code 1" if x == 1 else "Code 0")
        cp = st.selectbox("Chest pain type (cp code)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 70, 250, 125)
        chol = st.number_input("Cholesterol (dataset units)", 80, 700, 212)
    with c2:
        fbs = st.selectbox("Fasting blood sugar flag (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG code (restecg)", [0, 1, 2])
        thalach = st.number_input("Maximum heart rate achieved (thalach)", 60, 250, 150)
        exang = st.selectbox("Exercise-induced angina flag (exang)", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with c3:
        slope = st.selectbox("ST segment slope code (slope)", [0, 1, 2])
        ca = st.selectbox("Major vessels code (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia code (thal)", [0, 1, 2, 3])
        st.info("Categorical fields are entered as numeric dataset codes. This interface does not verify that a combination is clinically plausible.")
    submitted = st.form_submit_button("Run model prediction", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    st.divider()
    st.subheader("Model output")
    m1, m2 = st.columns(2)
    m1.metric("Estimated probability of dataset class 1", f"{probability:.1%}")
    m2.metric("Predicted dataset class", str(prediction))
    st.write(f"The fitted model assigned this input to **target class {prediction}**.")
    st.caption("The probability is the model's estimated probability for target=1; it is not necessarily a calibrated personal medical risk.")
    report = row.copy()
    report["predicted_target_class"] = prediction
    report["estimated_probability_target_1"] = probability
    st.download_button(
        "Download this demo prediction (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="heart_disease_demo_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.warning("A model output cannot confirm or exclude disease. This demo must not guide healthcare decisions.")

st.divider()
st.subheader("Model evaluation summary")
best = metrics["best_model_test_metrics"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected model", metrics["best_model"])
m2.metric("Test accuracy", f'{best["test_accuracy"]:.1%}')
m3.metric("Test recall", f'{best["test_recall"]:.1%}')
m4.metric("Test ROC-AUC", f'{best["test_roc_auc"]:.3f}')
st.caption(
    f'Original rows: {metrics["dataset_rows_original"]} • Exact duplicates removed: {metrics["duplicate_rows_removed"]} • '
    f'Rows after deduplication: {metrics["rows_after_deduplication"]} • Train rows: {metrics["train_rows"]} • Test rows: {metrics["test_rows"]}.'
)
if COMPARISON_PATH.exists():
    st.subheader("Candidate model comparison")
    st.dataframe(pd.read_csv(COMPARISON_PATH).round(3), use_container_width=True, hide_index=True)
for filename, title in [("confusion_matrix.png", "Confusion matrix"), ("roc_curve.png", "ROC curve")]:
    image_path = ROOT / "models" / filename
    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), caption=title, width=650)
