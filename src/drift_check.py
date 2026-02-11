
import yaml
import pandas as pd
from src.data_utils import load_live, compute_features_for_online, psi

DRIFT_THRESHOLD = 0.2


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]
    windows = params["windows"]

    offline = pd.read_csv(paths["offline_features"])

    live_apps = load_live(paths["live_applications"])
    live_feats = compute_features_for_online(live_apps, windows)

    offline_vals = offline[["total_apps_120d","avg_loan_150d","rejected_90d"]]
    live_vals = live_feats[["total_apps_120d","avg_loan_150d","rejected_90d"]]

    results = {}
    for col in offline_vals.columns:
        val = psi(offline_vals[col].values, live_vals[col].values)
        results[col] = val

    print("PSI by feature:", results)
    if any(v >= DRIFT_THRESHOLD for v in results.values()):
        print("DRIFT DETECTED")
    else:
        print("NO SIGNIFICANT DRIFT")


if __name__ == "__main__":
    main()
