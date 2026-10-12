from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


REPORT_DIR = Path("reports/gaia_dimensionality3")
SOURCE = Path("reports/gaia_anomalies/anomaly-scores.csv")

FEATURES = ["parallax", "pmra", "pmdec"]
FEATURES = [
    "bp_rp",
    "absolute_g_mag",
    "parallax",
    "pmra",
    "pmdec",
]


def plot_embedding(data, x, y, title, filename):
    # Reuse rankings from the previous experiment.
    isolation = data["if_rank"] <= 200
    lof = data["lof_rank"] <= 200

    groups = [
        (~isolation & ~lof, "Other stars", "lightgray", 3),
        (isolation & ~lof, "IsolationForest", "crimson", 16),
        (~isolation & lof, "LOF", "royalblue", 16),
        (isolation & lof, "Both", "darkorange", 30),
    ]

    plt.figure(figsize=(9, 7))

    for mask, label, color, size in groups:
        plt.scatter(
            data.loc[mask, x],
            data.loc[mask, y],
            s=size,
            c=color,
            alpha=0.6,
            label=label,
        )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    # change visualisation so extreme observation do not dominate graphs scale
    if x == "PC1":
        plt.xlim(*np.percentile(data[x], [1, 99]))
    plt.ylim(*np.percentile(data[y], [1, 99]))
    plt.tight_layout()

    plt.savefig(REPORT_DIR / filename, dpi=160)
    plt.close()


def main():
    print("Features:")
    print(FEATURES)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURES)
        .copy()
    )

    print(f"Stars: {len(data):,}", flush=True)

    # Standardize all five features.
    X = StandardScaler().fit_transform(data[FEATURES])

    # --------------------------------------------------
    # PCA: linear dimensionality reduction
    # --------------------------------------------------
    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(X)

    data["PC1"] = coordinates[:, 0]
    data["PC2"] = coordinates[:, 1]

    print("\nPCA explained variance:")
    print(pca.explained_variance_ratio_)
    print(
        "Total:",
        pca.explained_variance_ratio_.sum(),
    )

    # Which original features contribute to each axis?
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1", "PC2"],
    )

    print("\nPCA feature loadings:")
    print(loadings.round(3))

    plot_embedding(
        data,
        "PC1",
        "PC2",
        "Gaia stars: PCA",
        "pca.png",
    )

    # --------------------------------------------------
    # UMAP: nonlinear dimensionality reduction
    # --------------------------------------------------
    print("\nFitting UMAP...", flush=True)

    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    )

    coordinates = umap.fit_transform(X)

    data["UMAP1"] = coordinates[:, 0]
    data["UMAP2"] = coordinates[:, 1]

    plot_embedding(
        data,
        "UMAP1",
        "UMAP2",
        "Gaia stars: UMAP",
        "umap.png",
    )

    data[
        [
            "source_id",
            "if_rank",
            "lof_rank",
            "PC1",
            "PC2",
            "UMAP1",
            "UMAP2",
        ]
    ].to_csv(
        REPORT_DIR / "embeddings.csv",
        index=False,
        )

    print("\nSaved PCA and UMAP plots.")


if __name__ == "__main__":
    main()
