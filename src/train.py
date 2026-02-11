
import os
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "loan_risk"
MODEL_NAME = "loan_risk_model"

def ensure_experiment(client: MlflowClient, name: str, artifact_root: str = "mlartifacts"):
    """Ensure an MLflow experiment exists, creating it if necessary."""
    exp = client.get_experiment_by_name(name)
    if exp is None:
        artifact_uri = f"file:{os.path.abspath(artifact_root)}"
        exp_id = client.create_experiment(name=name, artifact_location=artifact_uri)
        exp = client.get_experiment(exp_id)
    return exp


def train_and_log(train_df: pd.DataFrame, test_df: pd.DataFrame, C: float, exp_id: str, params: dict):
    """Train a logistic regression model, log metrics and artifacts to MLflow."""
    features = ["total_apps_120d", "avg_loan_150d", "rejected_90d"]

    X_train, y_train = train_df[features], train_df["high_risk"]
    X_test, y_test = test_df[features], test_df["high_risk"]

    with mlflow.start_run(run_name=f"logreg_C={C}", experiment_id=exp_id):
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Save and log artifacts
        os.makedirs("artifacts", exist_ok=True)
        model_path = f"artifacts/model_C_{C}.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        mlflow.log_artifact("data/processed/train.csv", artifact_path="split_snapshot")
        mlflow.log_artifact("data/processed/test.csv", artifact_path="split_snapshot")
        mlflow.log_artifact(params["paths"]["offline_features"], artifact_path="data_snapshot")

        # Log sklearn model
        mlflow.sklearn.log_model(clf, artifact_path="sk_model")

        run = mlflow.active_run()
        return run.info.run_id, {"accuracy": acc, "precision": prec, "recall": rec}


def main():
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = ensure_experiment(client, EXPERIMENT_NAME, artifact_root="mlartifacts")

    # Train across different C values
    results = []
    for C in params["train"]["Cs"]:
        run_id, metrics = train_and_log(train_df, test_df, C, exp.experiment_id, params)
        results.append((run_id, metrics, C))

    # Select best run based on accuracy and recall
    best_run_id, best_metrics, best_C = sorted(
        results, key=lambda x: (x[1]["accuracy"], x[1]["recall"]), reverse=True
    )[0]

    print(f"Best run: {best_run_id}, C={best_C}, metrics={best_metrics}")


if __name__ == "__main__":
    main()
