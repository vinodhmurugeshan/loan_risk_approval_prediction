import os
import sqlite3
import yaml
import pandas as pd
from src.data_utils import load_historical, load_live, compute_features_for_online

DDL = """
CREATE TABLE IF NOT EXISTS features (
    applicant_id TEXT PRIMARY KEY,
    total_apps_120d INTEGER,
    avg_loan_150d REAL,
    rejected_90d INTEGER,
    feature_timestamp TEXT
);
"""

def upsert_features(conn, df: pd.DataFrame):
    """Insert or update features in the SQLite store."""
    df = df.copy()
    cols = ["applicant_id", "total_apps_120d", "avg_loan_150d", "rejected_90d", "feature_timestamp"]
    df['feature_timestamp'] =pd.to_datetime(df['feature_timestamp'] ).astype(str)
    # Ensure feature_timestamp is stringified
    if "feature_timestamp" in df.columns:
        df["feature_timestamp"] = pd.to_datetime(df["feature_timestamp"]).astype(str)

    df = df[cols]
    placeholders = ",".join(["?"] * len(cols))

    sql = f"""
    INSERT INTO features ({",".join(cols)}) VALUES ({placeholders})
    ON CONFLICT(applicant_id) DO UPDATE SET
      total_apps_120d = excluded.total_apps_120d,
      avg_loan_150d = excluded.avg_loan_150d,
      rejected_90d = excluded.rejected_90d,
      feature_timestamp = excluded.feature_timestamp;
    """

    conn.executemany(sql, df.values.tolist())
    conn.commit()


def main():
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]
    windows = params["windows"]

    # Load historical and live applications
    apps_hist, _ = load_historical(paths["applications"], paths["labels"])
    apps_live = load_live(paths["live_applications"])
    apps = pd.concat([apps_hist, apps_live], ignore_index=True)

    # Compute online features
    feats_online = compute_features_for_online(apps, windows)

    # Ensure output directory exists
    output_path = paths["online_store"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Upsert into SQLite store
    conn = sqlite3.connect(output_path)
    try:
        conn.execute(DDL)
        upsert_features(conn, feats_online)
        print(f"Upserted {len(feats_online)} rows into {output_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()