from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia


REPORT = Path("reports/gaia_stellar_temperature")


def main():
    df = pd.read_csv(REPORT / "predictions.csv")

    worst = (
        df.assign(abs_error=lambda x: x.residual.abs())
        .nlargest(10, "abs_error")
        .copy()
    )

    ids = ", ".join(map(str, worst.source_id.astype("int64")))

    query = f"""
        SELECT
            source_id,
            teff_gspspec_lower,
            teff_gspspec_upper,
            flags_gspspec,
            teff_gspphot
        FROM gaiadr3.astrophysical_parameters
        WHERE source_id IN ({ids})
    """

    quality = Gaia.launch_job_async(query).get_results().to_pandas()

    result = worst.merge(quality, on="source_id", how="left")

    result["temperature_uncertainty"] = (
                                                result.teff_gspspec_upper - result.teff_gspspec_lower
                                        ) / 2

    result["quality_flags"] = (
        result.flags_gspspec.astype("string").str[:13]
    )

    columns = [
        "source_id",
        "bp_rp",
        "teff_gspspec",
        "predicted",
        "residual",
        "temperature_uncertainty",
        "teff_gspphot",
        "quality_flags",
    ]

    print(result[columns].round(1).to_string(index=False))

    result.to_csv(REPORT / "outliers_with_quality.csv", index=False)


if __name__ == "__main__":
    main()
