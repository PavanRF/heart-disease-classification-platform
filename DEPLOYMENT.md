# Deployment guide

## Local run (Windows PowerShell)
From the project root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python test_smoke.py
python -m streamlit run app.py
```

## Docker
Install Docker Desktop, then from the project root:
```bash
docker build -t heart-disease-classifier:0.1.0 .
docker run --rm -p 8501:8501 heart-disease-classifier:0.1.0
```
Open http://localhost:8501.

## Publish the code
1. Create a GitHub repository.
2. Review `data/heart.csv` licensing/provenance and confirm you are allowed to redistribute it. Do not upload patient-identifiable or restricted data.
3. Add source code, tests, documentation and dependency pins. Do not commit secrets or private data.
4. Push to GitHub; the included GitHub Actions workflow runs training, smoke tests and syntax checks.
5. For public hosting, use a managed container host or Streamlit-compatible host. Configure build/start commands based on `Dockerfile`; verify the host's memory/time limits and model artifact handling.

## Before a company or public clinical deployment
This repository is not clinically validated. A public URL is software deployment, not clinical approval. Do not accept identifiable patient data or market the output as a diagnosis. Obtain independent validation, legal/privacy/security review, model monitoring, and qualified clinical governance before any health-impacting use.
