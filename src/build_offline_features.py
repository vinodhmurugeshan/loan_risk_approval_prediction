import os
import yaml
from dateutil import parser
from src.data_utils import load_historical, compute_features


def get_reference_date(apps, ref_date_cfg):
    """Return the reference date either from config or max application date."""
    return parser.parse(ref_date_cfg) if ref_date_cfg else apps["application_date"].max()


def main():
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    paths = params["paths"]
    windows = params["windows"]
    ref_date_cfg = params.get("reference_date")

    # Load data
    apps, labels = load_historical(paths["applications"], paths["labels"])

    # Determine reference date
    reference_date = get_reference_date(apps, ref_date_cfg)

    # Compute features and merge with labels
    feats = compute_features(apps, reference_date, windows)
    train_df = feats.merge(labels, on="applicant_id", how="inner")

    # Ensure output directory exists
    output_path = paths["offline_features"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save results
    train_df.to_csv(output_path, index=False)
    print(f"Wrote offline features to {output_path} with shape {train_df.shape}")
    print(train_df.head())


if __name__ == "__main__":
    main()