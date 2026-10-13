from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia


OUT = Path("reports/gaia_stellar_temperature")
PREDICTIONS = OUT / "test_predictions.csv"
CACHE = OUT / "gaia_quality.csv"


def download_quality(source_ids):
    frames = []
    ids = sorted(set(map(int, source_ids)))

    for start in range(0, len(ids), 200):
        chunk = ids[start:start + 200]

        print(
            f"[Gaia] quality metadata: "
            f"{min(start + 200, len(ids))}/{len(ids)}",
            flush=True,
        )

        query = f"""
            SELECT
                ap.source_id,
                ap.flags_gspspec,
                ap.teff_gspphot,
                ap.teff_gspspec_lower,
                ap.teff_gspspec_upper
            FROM gaiadr3.astrophysical_parameters AS ap
            WHERE ap.source_id IN ({",".join(map(str, chunk))})
        """

        job = Gaia.launch_job_async(query, verbose=False)
        frames.append(job.get_results().to_pandas())

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(CACHE, index=False)

    return result


def summarize(name, df):
    if df.empty:
        return {
            "group": name,
            "stars": 0,
            "MAE_K": np.nan,
            "RMSE_K": np.nan,
            "p90_error_K": np.nan,
        }

    error = df["predicted"] - df["teff_gspspec"]

    return {
        "group": name,
        "stars": len(df),
        "MAE_K": error.abs().mean(),
        "RMSE_K": np.sqrt(np.mean(error**2)),
        "p90_error_K": error.abs().quantile(0.90),
    }


def main():
    predictions = pd.read_csv(PREDICTIONS)

    if CACHE.exists():
        quality = pd.read_csv(
            CACHE,
            dtype={"flags_gspspec": "string"},
        )
    else:
        quality = download_quality(predictions["source_id"])

    missing = (
            set(predictions["source_id"])
            - set(quality["source_id"])
    )
    if missing:
        raise ValueError(f"Missing Gaia metadata for {len(missing)} stars")

    df = predictions.merge(
        quality,
        on="source_id",
        validate="one_to_one",
    )

    # GSP-Spec's first 13 quality characters concern
    # the main stellar parameters.
    flags = (
        df["flags_gspspec"]
        .astype("string")
        .str[:13]
        .fillna("")
    )

    df["flag13"] = flags

    # Temperature-related checks:
    # 1: vbroadT, 4: vradT, 7: fluxNoise, 8: extrapol
    df["temperature_flags_good"] = (
            flags.str.len().ge(13)
            & flags.str[0].eq("0")
            & flags.str[3].eq("0")
            & flags.str[6].eq("0")
            & flags.str[7].eq("0")
    )

    # Much stricter quality criterion used in published
    # Gaia sample selection.
    df["strict_flags_good"] = flags.eq("0000000000000")

    df["abs_error"] = (
            df["predicted"] - df["teff_gspspec"]
    ).abs()

    df["spec_uncertainty"] = (
                                     df["teff_gspspec_upper"]
                                     - df["teff_gspspec_lower"]
                             ) / 2

    df["phot_spec_difference"] = (
            df["teff_gspphot"] - df["teff_gspspec"]
    ).abs()

    groups = [
        summarize("All test stars", df),
        summarize(
            "Temperature flags good",
            df[df["temperature_flags_good"]],
        ),
        summarize(
            "Temperature flags warning/missing",
            df[~df["temperature_flags_good"]],
        ),
        summarize(
            "All first 13 flags zero",
            df[df["strict_flags_good"]],
        ),
    ]

    print("\nPrediction errors by Gaia quality:")
    print(
        pd.DataFrame(groups).to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    print("\n10 worst predictions:")
    worst = df.nlargest(10, "abs_error")

    print(
        worst[
            [
                "source_id",
                "teff_gspspec",
                "predicted",
                "teff_gspphot",
                "abs_error",
                "phot_spec_difference",
                "spec_uncertainty",
                "flag13",
            ]
        ].round(1).to_string(index=False)
    )

    print(
        "\nTemperature-quality warnings:"
        f"\n  Full test set: {(~df['temperature_flags_good']).sum()}/{len(df)}"
        f"\n  Worst 10:      {(~worst['temperature_flags_good']).sum()}/10"
    )

    df.to_csv(OUT / "quality_checked.csv", index=False)


if __name__ == "__main__":
    main()
