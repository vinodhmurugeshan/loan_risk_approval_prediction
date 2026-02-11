
import pandas as pd
import numpy as np

def _parse_dates(df, col="application_date"):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df

def load_historical(app_path, labels_path):
    apps = pd.read_csv(app_path)
    apps = _parse_dates(apps, "application_date")
    labels = pd.read_csv(labels_path)
    return apps, labels

def load_live(live_path):
    live = pd.read_csv(live_path)
    live = _parse_dates(live, "application_date")
    return live

def compute_features(apps: pd.DataFrame, reference_date, windows):
    apps = apps.copy()
    apps = apps[apps["application_date"] <= reference_date]

    w_total = reference_date - pd.to_timedelta(windows["total_apps_days"], unit="D")
    w_avg   = reference_date - pd.to_timedelta(windows["avg_loan_days"], unit="D")
    w_rej   = reference_date - pd.to_timedelta(windows["rejected_days"], unit="D")

    m_total = (apps["application_date"] > w_total)
    m_avg   = (apps["application_date"] > w_avg)
    m_rej   = (apps["application_date"] > w_rej)

    total_apps = apps.loc[m_total].groupby("applicant_id").size().rename("total_apps_120d")
    avg_loan   = apps.loc[m_avg].groupby("applicant_id")["loan_amount"].mean().rename("avg_loan_150d")
    rejected_  = apps.loc[m_rej & (apps["approved"] == 0)].groupby("applicant_id").size().rename("rejected_90d")

    feats = pd.concat([total_apps, avg_loan, rejected_], axis=1).fillna(0)
    feats["total_apps_120d"] = feats["total_apps_120d"].astype(int)
    feats["avg_loan_150d"] = feats["avg_loan_150d"].astype(float)
    feats["rejected_90d"] = feats["rejected_90d"].astype(int)
    feats = feats.reset_index()
    return feats

def compute_features_for_online(apps: pd.DataFrame, windows, as_of_date=None):
    apps = apps.copy()
    apps = apps.sort_values(["applicant_id", "application_date"])
    latest_per_app = apps.groupby("applicant_id")["application_date"].max().rename("ref_date")
    apps = apps.merge(latest_per_app, on="applicant_id", how="left")

    def within_window(row, win_days):
        return row["application_date"] > (row["ref_date"] - pd.to_timedelta(win_days, unit="D"))

    apps["in_total"] = apps.apply(lambda r: within_window(r, windows["total_apps_days"]), axis=1)
    apps["in_avg"]   = apps.apply(lambda r: within_window(r, windows["avg_loan_days"]), axis=1)
    apps["in_rej"]   = apps.apply(lambda r: within_window(r, windows["rejected_days"]), axis=1)

    total_apps = apps.loc[apps["in_total"]].groupby("applicant_id").size().rename("total_apps_120d")
    avg_loan   = apps.loc[apps["in_avg"]].groupby("applicant_id")["loan_amount"].mean().rename("avg_loan_150d")
    rejected_  = apps.loc[apps["in_rej"] & (apps["approved"] == 0)].groupby("applicant_id").size().rename("rejected_90d")

    feats = pd.concat([total_apps, avg_loan, rejected_], axis=1).fillna(0).reset_index()
    feats = feats.merge(latest_per_app.reset_index(), on="applicant_id", how="left").rename(columns={"ref_date": "feature_timestamp"})
    feats["total_apps_120d"] = feats["total_apps_120d"].astype(int)
    feats["avg_loan_150d"] = feats["avg_loan_150d"].astype(float)
    feats["rejected_90d"] = feats["rejected_90d"].astype(int)
    return feats

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = np.nan_to_num(expected, nan=np.nanmedian(expected))
    actual   = np.nan_to_num(actual,   nan=np.nanmedian(actual))

    cuts = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    cuts = np.unique(cuts)

    def bucketize(arr, bins):
        idx = np.digitize(arr, bins, right=False) - 1
        idx = np.clip(idx, 0, len(bins) - 2)
        return idx

    e_idx = bucketize(expected, cuts)
    a_idx = bucketize(actual, cuts)

    e_counts = np.bincount(e_idx, minlength=len(cuts)-1).astype(float)
    a_counts = np.bincount(a_idx, minlength=len(cuts)-1).astype(float)

    e_perc = e_counts / max(e_counts.sum(), 1.0)
    a_perc = a_counts / max(a_counts.sum(), 1.0)

    a_perc = np.where(a_perc == 0, 1e-6, a_perc)
    e_perc = np.where(e_perc == 0, 1e-6, e_perc)

    psi_val = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi_val)
