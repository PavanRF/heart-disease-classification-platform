from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def test_dataset_contract():
    data = pd.read_csv(ROOT / "data" / "heart.csv")
    assert set(FEATURES + ["target"]).issubset(data.columns)
    assert set(data["target"].dropna().unique()).issubset({0, 1})

def test_saved_model_contract():
    model_path = ROOT / "models" / "heart_disease_pipeline.joblib"
    assert model_path.exists(), "Run python train_model.py first."
    model = joblib.load(model_path)
    data = pd.read_csv(ROOT / "data" / "heart.csv").drop_duplicates()
    x = data[FEATURES].iloc[[0]]
    pred = model.predict(x)
    proba = model.predict_proba(x)
    assert len(pred) == 1
    assert int(pred[0]) in (0, 1)
    assert proba.shape == (1, 2)
    assert abs(float(proba.sum()) - 1.0) < 1e-6
