"""Compare PCA and UMAP projections of Gaia sources (3 or 5 features).

Run from the project root:
    python -m experiments.gaia_dimensionality             # 5 features
    python -m experiments.gaia_dimensionality --features 3
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

REPORT_DIR = Path("reports/gaia_dimensionality")
SOURCE = Path("reports/gaia_anomalies/anomaly-scores.csv")

FEATURE_SETS = {
    3: ["parallax", "pmra", "pmdec"],
    5: ["bp_rp", "absolute_g_mag", "parallax", "pmra", "pmdec"],
}


def plot_embedding(data: pd.DataFrame, x: str, y: str,
                   title: str, output_file: Path) -> None:
    isolation = data["if_rank"] <= 200
    lof = data["lof_rank"] <= 200

    groups = [
        (~isolation & ~lof, "Other stars", "lightgray", 3),
        (isolation & ~lof, "IsolationForest", "crimson", 16),
        (~isolation & lof, "LOF", "royalblue", 16),
        (isolation & lof, "Both", "darkorange", 30),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    for mask, label, color, size in groups:
        ax.scatter(data.loc[mask, x], data.loc[mask, y],
                   s=size, c=color, alpha=0.6, label=label)

    ax.set(xlabel=x, ylabel=y, title=title)
    if x == "PC1":
        ax.set_xlim(*np.percentile(data[x], [1, 99]))
        ax.set_ylim(*np.percentile(data[y], [1, 99]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def main(feature_count: int = 5) -> None:
    features = FEATURE_SETS[feature_count]
    out = REPORT_DIR / f"{feature_count}d"
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE, dtype={"source_id": "int64"})
    data = (data.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=features).copy())

    print("Features:", features)
    print(f"Stars: {len(data):,}")
    X = StandardScaler().fit_transform(data[features])

    pca = PCA(n_components=2)
    xy = pca.fit_transform(X)
    data[["PC1", "PC2"]] = xy
    print("\nPCA explained variance:", pca.explained_variance_ratio_)
    print("Total:", pca.explained_variance_ratio_.sum())
    print("\nPCA feature loadings:")
    print(pd.DataFrame(pca.components_.T, index=features,
                       columns=["PC1", "PC2"]).round(3))

    plot_embedding(data, "PC1", "PC2", "Gaia stars: PCA", out / "pca.png")

    print("\nFitting UMAP...", flush=True)
    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                random_state=42, n_jobs=1)
    data[["UMAP1", "UMAP2"]] = umap.fit_transform(X)
    plot_embedding(data, "UMAP1", "UMAP2", "Gaia stars: UMAP", out / "umap.png")

    cols = ["source_id", "if_rank", "lof_rank", "PC1", "PC2", "UMAP1", "UMAP2"]
    data[cols].to_csv(out / "embeddings.csv", index=False)
    (out / "features.json").write_text(
        json.dumps(features, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, choices=(3, 5), default=5,
                        help="Number of Gaia features (default: 5)")
    args = parser.parse_args()
    main(args.features)
