from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
model_path = ROOT / "models" / "heart_disease_pipeline.joblib"
data_path = ROOT / "data" / "heart.csv"

def test_model_artifact_predicts():
    assert model_path.exists(), "Run python train_model.py first."
    data = pd.read_csv(data_path).drop_duplicates()
    row = data.drop(columns="target").iloc[[0]]
    model = joblib.load(model_path)
    prediction = model.predict(row)
    probabilities = model.predict_proba(row)
    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)
    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6

if __name__ == "__main__":
    test_model_artifact_predicts()
    print("Smoke test passed.")
