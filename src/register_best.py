
import yaml
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "loan_risk"
MODEL_NAME = "loan_risk_model"

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.accuracy DESC", "metrics.recall DESC"],
        max_results=5,
    )
    best_run = runs[0]
    best_run_id = best_run.info.run_id

    mv = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/sk_model",
        name=MODEL_NAME
    )

    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
    )
    print(f"Registered run {best_run_id} as {MODEL_NAME} v{mv.version} (Production)")
