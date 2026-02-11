
import argparse
import sqlite3
import yaml
import mlflow
import pandas as pd

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/loan_risk_model/Production"

def fetch_online_features(db_path, applicant_id):
    conn = sqlite3.connect(db_path)
    try:
        row = pd.read_sql_query(
            "SELECT total_apps_120d, avg_loan_150d, rejected_90d FROM features WHERE applicant_id = ?",
            conn, params=(applicant_id,)
        )
        if row.empty:
            return None
        return row[["total_apps_120d","avg_loan_150d","rejected_90d"]].iloc[0].values.reshape(1, -1)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant_id", required=True)
    args = parser.parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    X = fetch_online_features(params["paths"]["online_store"], args.applicant_id)
    if X is None:
        print(f"No online features found for applicant_id={args.applicant_id}.")
        return

    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    label = "HIGH_RISK" if pred == 1 else "LOW_RISK"

    print(f"Applicant: {args.applicant_id}")
    print(f"Prediction: {label}")
    print(f"Probability(high_risk=1): {prob:.4f}")


if __name__ == "__main__":
    main()
