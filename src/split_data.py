
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    offline_path = params["paths"]["offline_features"]
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    df = pd.read_csv(offline_path)
    df = df.dropna(subset=["high_risk"]).copy()

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["high_risk"]
    )

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Wrote {out_dir}/train.csv shape={train_df.shape}")
    print(f"Wrote {out_dir}/test.csv  shape={test_df.shape}")

if __name__ == "__main__":
    main()
