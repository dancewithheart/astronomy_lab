from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from datasets import PLEIADES, load_dataset
from experiments.pleiades import (
    add_absolute_g_magnitude,
    prepare_features,
)

# unusual colour–luminosity combinations
FEATURES = ["bp_rp", "absolute_g_mag"]

# unusual colour–luminosity distance and motion combinations
# FEATURES = [
#     "bp_rp",
#     "absolute_g_mag",
#     "parallax",
#     "pmra",
#     "pmdec",
# ]

TOP_N = 200
REPORT_DIR = Path("reports/gaia_anomalies")


def plot_cmd(data, candidates, title, filename):
    plt.figure(figsize=(8, 8))

    plt.scatter(
        data["bp_rp"],
        data["absolute_g_mag"],
        s=5,
        alpha=0.15,
        color="gray",
        label="All stars",
    )

    plt.scatter(
        candidates["bp_rp"],
        candidates["absolute_g_mag"],
        s=25,
        color="crimson",
        label=f"Top {len(candidates)} anomalies",
    )

    plt.xlabel("BP - RP")
    plt.ylabel("Absolute G magnitude")
    plt.title(title)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    plt.savefig(REPORT_DIR / filename, dpi=160)
    plt.close()


def main():
    print("Features")
    print(FEATURES)
    # Reuse the existing Gaia sample and transformations.
    data = load_dataset(PLEIADES)
    data = prepare_features(data)
    data = add_absolute_g_magnitude(data)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES)
        .copy()
    )

    print(f"Stars: {len(data):,}")

    # Median/IQR scaling. Particularly useful for LOF,
    # because LOF depends on distances between observations.
    X = RobustScaler().fit_transform(data[FEATURES])

    # 1. Isolation Forest
    forest = IsolationForest(
        n_estimators=250,
        max_samples=1024,
        random_state=42,
        n_jobs=-1,
    )

    forest.fit(X)

    # Negate sklearn scores:
    # larger value = more anomalous.
    data["if_score"] = -forest.score_samples(X)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(
        n_neighbors=35,
        n_jobs=-1,
    )

    lof.fit_predict(X)

    data["lof_score"] = -lof.negative_outlier_factor_

    # Rank from most unusual to least unusual.
    data["if_rank"] = (
        data["if_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    data["lof_rank"] = (
        data["lof_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    top_if = data.nsmallest(TOP_N, "if_rank")
    top_lof = data.nsmallest(TOP_N, "lof_rank")

    # 3. Compare algorithms.
    overlap = len(
        set(top_if["source_id"])
        & set(top_lof["source_id"])
    )

    print(f"Top-{TOP_N} overlap: {overlap}")

    # 4. Inspect unusual stars.
    columns = [
        "source_id",
        "bp_rp",
        "absolute_g_mag",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "ruwe",
        "if_score",
        "lof_rank",
    ]

    print("\nMost unusual stars (IsolationForest):")
    print(top_if[columns].head(20).to_string(index=False))

    # 5. Save results.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        REPORT_DIR / "anomaly-scores.csv",
        index=False,
        )

    top_if.to_csv(
        REPORT_DIR / "isolation-forest-top100.csv",
        index=False,
        )

    top_lof.to_csv(
        REPORT_DIR / "lof-top100.csv",
        index=False,
        )

    plot_cmd(
        data,
        top_if,
        "Gaia: IsolationForest anomalies",
        "isolation-forest-cmd.png",
    )

    plot_cmd(
        data,
        top_lof,
        "Gaia: Local Outlier Factor anomalies",
        "lof-cmd.png",
    )

    if_only = data[
        (data["if_rank"] <= TOP_N)
        & (data["lof_rank"] > TOP_N)
        ].sort_values("if_rank")

    lof_only = data[
        (data["lof_rank"] <= TOP_N)
        & (data["if_rank"] > TOP_N)
        ].sort_values("lof_rank")

    shared = data[
        (data["if_rank"] <= TOP_N)
        & (data["lof_rank"] <= TOP_N)
        ]

    neither = data[
        (data["if_rank"] > TOP_N)
        & (data["lof_rank"] > TOP_N)
        ]

    # Reproducible random sample of non-selected stars.
    control = neither.sample(n=200, random_state=42)

    for name, stars in {
        "if-only": if_only,
        "lof-only": lof_only,
        "shared": shared,
        "random-control": control,
    }.items():
        stars.to_csv(
            REPORT_DIR / f"{name}.csv",
            index=False,
            )
        print(f"{name}: {len(stars)}")


if __name__ == "__main__":
    main()
