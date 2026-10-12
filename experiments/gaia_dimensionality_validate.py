"""Validate existing 3D/5D Gaia embeddings against a Pleiades catalogue.

Run from repository root, after saving both embeddings under distinct names.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

REPORT = Path("reports/gaia_dimensionality")
GAIA = Path("reports/gaia_anomalies/anomaly-scores.csv")
REFERENCE = Path("data/pleiades_reference.csv")


def reference_ids() -> set[int]:
    if not REFERENCE.exists():
        from experiments.pleiades_reference import load_reference_pleiades

        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        load_reference_pleiades().to_csv(REFERENCE, index=False)
    ref = pd.read_csv(REFERENCE, dtype={"source_id": "int64"})
    return set(ref["source_id"])


def load_embedding(name: str, gaia: pd.DataFrame) -> pd.DataFrame:
    path = REPORT / f"embeddings-{name}.csv"
    emb = pd.read_csv(path, dtype={"source_id": "int64"})
    result = emb.merge(
        gaia[["source_id", "parallax", "pmra", "pmdec"]],
        on="source_id", how="left", validate="one_to_one",
    )
    if result[["parallax", "pmra", "pmdec"]].isna().any().any():
        raise ValueError(f"Some sources in {path} are missing Gaia measurements")
    return result


def member_neighbor_fraction(
        data: pd.DataFrame, mask: np.ndarray, columns: list[str], k: int = 10
) -> float:
    """Among labeled member stars, fraction of their k 2D neighbours labeled member."""
    X = data[columns].to_numpy()
    member_rows = np.flatnonzero(mask)
    if len(member_rows) == 0:
        return float("nan")
    # Query 11 neighbours to include self, then explicitly remove self.
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X)
    inds = neighbours.kneighbors(X[member_rows], return_distance=False)
    other_rows = np.stack([
        row[row != i][:k] for row, i in zip(inds, member_rows)
    ])
    return float(mask[other_rows].mean())


def plot_members(data: pd.DataFrame, mask: np.ndarray, name: str, x: str, y: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(data.loc[~mask, x], data.loc[~mask, y],
               s=3, alpha=0.18, c="gray", label="Other Gaia sources")
    ax.scatter(data.loc[mask, x], data.loc[mask, y],
               s=14, alpha=0.85, c="orange", label="Published Pleiades members")
    if x == "PC1":
        ax.set_xlim(*np.quantile(data[x], [0.01, 0.99]))
        ax.set_ylim(*np.quantile(data[y], [0.01, 0.99]))
    ax.set(xlabel=x, ylabel=y, title=f"Gaia {name}: {x[:2]} with published Pleiades members")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / f"{name}-{x[:2].lower()}-pleiades.png", dpi=160)
    plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    gaia = pd.read_csv(GAIA, dtype={"source_id": "int64"})
    refs = reference_ids()
    results = {}

    for name in ("3d", "5d"):
        data = load_embedding(name, gaia)
        mask = data["source_id"].isin(refs).to_numpy()
        results[name] = (data, mask)
        print(f"\n{name}: stars={len(data):,}, reference matches={mask.sum():,}/{len(refs):,}")

        # Independent published labels: preferred validation.
        # Also reproduce the exploratory astrometric proxy comparison.
        proxy = (
                data["parallax"].between(6.76, 7.96)
                & data["pmra"].between(14.8, 24.8)
                & data["pmdec"].between(-50.35, -40.35)
        ).to_numpy()
        print(f"  Astrometric proxy: {proxy.sum():,} stars (NOT ground truth)")

        for tag, cols in (("PCA", ["PC1", "PC2"]),
                          ("UMAP", ["UMAP1", "UMAP2"])):
            p = member_neighbor_fraction(data, mask, cols)
            q = member_neighbor_fraction(data, proxy, cols)
            print(f"  {tag:4s}: published-member 10NN={p:.1%}, proxy 10NN={q:.1%}")
            plot_members(data, mask, name, *cols)

    # Additional clustering of the 3-feature UMAP embedding.
    data, mask = results["3d"]
    labels = DBSCAN(eps=0.25, min_samples=15).fit_predict(data[["UMAP1", "UMAP2"]])
    eligible = labels >= 0
    overlaps = (pd.DataFrame({"label": labels[eligible], "member": mask[eligible]})
                .groupby("label")["member"].sum())
    if overlaps.empty or overlaps.max() == 0:
        print("\nNo UMAP cluster overlaps the published reference.")
        return

    # Reference catalogue chooses *which* unsupervised cluster to describe;
    # it does not influence the PCA/UMAP embedding or DBSCAN fit.
    cluster_id = int(overlaps.idxmax())
    island = data.loc[labels == cluster_id]
    print("\nUMAP island (DBSCAN applied to 3D-feature embedding):")
    print(f"  DBSCAN label:    {cluster_id}")
    print(f"  Sources:         {len(island):,}")
    print(f"  Known members:   {int(overlaps.loc[cluster_id]):,}/{mask.sum():,} in sample")
    for col in ("parallax", "pmra", "pmdec"):
        print(f"  Median {col}: {island[col].median():.4f}")


if __name__ == "__main__":
    main()
