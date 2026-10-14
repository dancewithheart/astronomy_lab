import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split


REPORT_DIR = Path("reports/gaia_stellar_temperature")
DATA_DIR = Path("data")

TARGET = "teff_gspspec"

FEATURE_SETS = {
    "Colour": ["bp_rp"],
    "Colour + luminosity": ["bp_rp", "absolute_g_mag"],
}


def query_gaia(radius):
    # Use the same sky centre as the Pleiades experiment.
    query = f"""
    SELECT TOP 10000
        s.source_id,
        s.bp_rp,
        s.phot_g_mean_mag,
        s.parallax,
        ap.teff_gspspec
    FROM gaiadr3.gaia_source_lite AS s
    JOIN gaiadr3.astrophysical_parameters AS ap
        ON s.source_id = ap.source_id
    WHERE
        1 = CONTAINS(
            POINT('ICRS', s.ra, s.dec),
            CIRCLE('ICRS', 56.87125, 24.10493, {radius})
        )
        AND ap.teff_gspspec IS NOT NULL
        AND s.bp_rp IS NOT NULL
        AND s.phot_g_mean_mag IS NOT NULL
        AND s.parallax > 0
        AND s.parallax_over_error >= 10
        AND s.ruwe < 1.4
    """

    print("Downloading Gaia temperatures...")
    job = Gaia.launch_job_async(query)
    return job.get_results().to_pandas()


def load_data(radius, refresh=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache = DATA_DIR / f"gaia_temperature_{radius:g}deg.csv"

    if cache.exists() and not refresh:
        data = pd.read_csv(cache)
    else:
        data = query_gaia(radius)
        data.to_csv(cache, index=False)

    # Convert apparent brightness into absolute magnitude.
    data["absolute_g_mag"] = (
            data["phot_g_mean_mag"]
            + 5 * np.log10(data["parallax"])
            - 10
    )

    columns = [
        "bp_rp",
        "absolute_g_mag",
        TARGET,
    ]

    return (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=columns)
        .reset_index(drop=True)
    )


def models():
    return {
        "Baseline": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            random_state=42,
        ),
    }


def plot_results(predictions):
    actual = predictions[TARGET]
    predicted = predictions["predicted"]
    residual = predictions["residual"]

    # Actual versus predicted temperature.
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, s=8, alpha=0.4)

    limits = [min(actual.min(), predicted.min()),
              max(actual.max(), predicted.max())]

    plt.plot(limits, limits, "k--", label="Perfect prediction")

    plt.xlabel("Gaia spectroscopic temperature [K]")
    plt.ylabel("Predicted temperature [K]")
    plt.title("Stellar temperature: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "predictions.png", dpi=160)
    plt.close()

    # Where does our model make mistakes?
    plt.figure(figsize=(7, 6))
    plt.scatter(predictions["bp_rp"], residual, s=8, alpha=0.4)
    plt.axhline(0, color="black", linestyle="--")

    plt.xlabel("BP - RP colour")
    plt.ylabel("Prediction error [K]")
    plt.title("Random Forest: residuals vs colour")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "residuals.png", dpi=160)
    plt.close()


def main(refresh=False, radius=3.0):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stars = load_data(radius, refresh=refresh)
    print(f"Stars: {len(stars):,}")

    if len(stars) < 100:
        raise ValueError("Too few stars. Try a larger query radius.")

    # Same train/test split for every experiment.
    train, test = train_test_split(
        stars,
        test_size=0.25,
        random_state=42,
    )

    results = []

    for feature_name, features in FEATURE_SETS.items():
        print(f"\nFeatures: {features}")

        X_train = train[features]
        X_test = test[features]
        y_train = train[TARGET]
        y_test = test[TARGET]

        for name, model in models().items():
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)

            mae = mean_absolute_error(y_test, predicted)
            rmse = root_mean_squared_error(y_test, predicted)
            r2 = r2_score(y_test, predicted)

            print(
                f"{name:20s}"
                f" MAE={mae:7.1f} K"
                f" RMSE={rmse:7.1f} K"
                f" R²={r2:.3f}"
            )

            results.append({
                "features": feature_name,
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            })

            # Save detailed errors for one representative model.
            if name == "RandomForest" and len(features) == 2:
                predictions = test.copy()
                predictions["predicted"] = predicted
                predictions["residual"] = predicted - y_test

                predictions.to_csv(
                    REPORT_DIR / "predictions.csv",
                    index=False,
                    )

                plot_results(predictions)

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
        )

    # errors analysis

    df = pd.read_csv("reports/gaia_stellar_temperature/predictions.csv")

    df["error"] = df["predicted"] - df["teff_gspspec"]
    df["abs_error"] = df["error"].abs()

    bins = [0, 4000, 4500, 5000, 5500, 6000, 6500, 7000, np.inf]
    df["band"] = pd.cut(df["teff_gspspec"], bins)

    print("\nErrors by temperature:")
    print(df.groupby("band", observed=True)["error"].agg(
        count="size",
        bias="mean",
        mae=lambda x: x.abs().mean(),
    ).round(1))

    print("\n10 worst predictions:")
    print(df.nlargest(10, "abs_error")[
              ["source_id", "bp_rp", "teff_gspspec", "predicted", "error"]
          ].to_string(index=False))

    # save test predictions
    out = Path("reports/gaia_stellar_temperature")
    out.mkdir(parents=True, exist_ok=True)

    df[["source_id", "teff_gspspec", "predicted"]].to_csv(
        out / "test_predictions.csv",
        index=False,
        )

    print(f"\nResults saved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()

    main(refresh=args.refresh, radius=args.radius)
