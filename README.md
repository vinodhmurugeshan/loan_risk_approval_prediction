
# Real-Time Loan Risk MLOps (DVC + MLflow + Feature Store)

## 1) Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

## 2) (Optional) Initialize Git & DVC
```bash
git init
dvc init
```

dvc remote add -d localremote ./dvc_remote

## 3) Build Offline Features (training snapshot)
```bash
python -m src.build_offline_features
```

## 4) Split data (persisted & DVC-tracked)
```bash
python -m src.split_data
```

## 5) Train (MLflow experiment `loan_risk`)
```bash
python -m src.train
```

## 6) Register best & promote to Production
```bash
python -m src.register_best
```

## 7) Build Online Feature Store (SQLite)
```bash
python -m src.build_online_store
```

## 8) Real-time Inference
```bash
python -m src.predict --applicant_id A001
```

## 9) Drift Detection (Offline vs Live)
```bash
python -m src.drift_check
```
## 10)DVC Commit
dvc commit

dvc repro (to run pipeline)

## 11) MLflow UI
```bash
python -c "import mlflow; from mlflow.tracking import MlflowClient; print('Tracking:', mlflow.get_tracking_uri()); print([e.name for e in MlflowClient().search_experiments()])"

python -c "import os; print(os.path.abspath('mlflow.db'))"

mlflow ui --backend-store-uri "sqlite:///C:\Users\Vinodh.M\Documents\loan_risk_approval_prediction\mlflow.db" --port 5006

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts --port 5006
# open  http://127.0.0.1:5006
```
## 12) Git push
git init

git status

git add .

git commit -m "First Commit from Local Machine"

git log

then login to git hub and create a repo with the same name as the project folder in local machine

git remote add origin https://github.com/vinodhmurugeshan/loan_risk_approval_prediction.git
git branch -M main
git push -u origin main

Notes:

- Experiment: `loan_risk`, registered model: `loan_risk_model`.
- Online inference uses SQLite features, not raw events.
