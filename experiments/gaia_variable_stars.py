"""Classify Gaia variable stars using supervised learning."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from astroquery.gaia import Gaia
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA = Path("data/gaia_variable_stars.csv")
REPORT = Path("reports/gaia_variable_stars")

CLASSES = ["RR", "ECL", "LPV"]

FEATURES = [
    "bp_rp",
    "phot_g_mean_mag",
    "range_mag_g_fov",
    "std_dev_mag_g_fov",
    "skewness_mag_g_fov",
    "abbe_mag_g_fov",
]


def load_data(refresh=False):
    if DATA.exists() and not refresh:
        return pd.read_csv(DATA)

    Gaia.ROW_LIMIT = -1
    frames = []

    for name in CLASSES:
        query = f"""
        SELECT TOP 3000
            v.source_id,
            v.best_class_name,
            g.bp_rp,
            g.phot_g_mean_mag,
            s.range_mag_g_fov,
            s.std_dev_mag_g_fov,
            s.skewness_mag_g_fov,
            s.abbe_mag_g_fov
        FROM gaiadr3.vari_classifier_result AS v
        JOIN gaiadr3.vari_summary AS s
            ON v.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g
            ON v.source_id = g.source_id
        WHERE v.best_class_name = '{name}'
          AND g.bp_rp IS NOT NULL
          AND s.std_dev_mag_g_fov IS NOT NULL
        """

        print(f"Downloading {name}...", flush=True)
        result = Gaia.launch_job_async(query).get_results()
        frames.append(result.to_pandas())

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates("source_id")

    DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA, index=False)

    return data


def main(refresh=False):
    REPORT.mkdir(parents=True, exist_ok=True)

    data = load_data(refresh)

    X = data[FEATURES]
    y = data["best_class_name"]

    print("\nClass distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            random_state=42,
        ),
    }

    predictions = {}
    fitted = {}

    print("\nModel comparison:")

    for name, estimator in models.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            estimator,
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        predictions[name] = pred
        fitted[name] = model

        score = f1_score(y_test, pred, average="macro")
        print(f"{name:20s} macro-F1={score:.3f}")

    # Examine Random Forest in more detail.
    forest = fitted["RandomForest"]
    pred = predictions["RandomForest"]

    print("\nRandom Forest classification report:")
    print(classification_report(y_test, pred))

    # 1. Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        labels=CLASSES,
        normalize="true",
    )

    plt.title("Variable star classification: Random Forest")
    plt.tight_layout()
    plt.savefig(REPORT / "confusion-matrix.png", dpi=160)
    plt.close()

    # 2. Permutation feature importance
    importance = permutation_importance(
        forest,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    scores = pd.Series(
        importance.importances_mean,
        index=FEATURES,
    ).sort_values()

    scores.to_csv(REPORT / "feature-importance.csv")

    scores.plot.barh()
    plt.xlabel("Decrease in macro-F1 when feature is shuffled")
    plt.title("Random Forest: feature importance")
    plt.tight_layout()
    plt.savefig(REPORT / "feature-importance.png", dpi=160)
    plt.close()

    feature_groups = {
        "Colour + brightness": [
            "bp_rp",
            "phot_g_mean_mag",
        ],
        "Variability only": [
            "range_mag_g_fov",
            "std_dev_mag_g_fov",
            "skewness_mag_g_fov",
            "abbe_mag_g_fov",
        ],
        "All features": FEATURES,
    }

    print("\nFeature ablation (Random Forest):")

    for name, columns in feature_groups.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        )

        model.fit(X_train[columns], y_train)
        pred = model.predict(X_test[columns])

        score = f1_score(y_test, pred, average="macro")
        print(f"{name:22s} macro-F1={score:.3f}")

        print(f"\nResults saved to {REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    main(refresh=args.refresh)
