from pathlib import Path

import pandas as pd
from astropy.table import Table
from astroquery.simbad import Simbad


REPORT_DIR = Path("reports/gaia_anomalies")


def load_candidates() -> pd.DataFrame:
    isolation = pd.read_csv(
        REPORT_DIR / "isolation-forest-top100.csv"
    )
    lof = pd.read_csv(
        REPORT_DIR / "lof-top100.csv"
    )

    isolation["method"] = "IsolationForest"
    lof["method"] = "LOF"

    return pd.concat(
        [isolation, lof],
        ignore_index=True,
    )


def query_simbad(source_ids: list[int]) -> pd.DataFrame:
    # SIMBAD identifiers require the Gaia DR3 prefix.
    identifiers = [
        f"Gaia DR3 {source_id}"
        for source_id in source_ids
    ]

    # Upload all identifiers in one table.
    upload = Table(
        {"gaia_id": identifiers}
    )

    query = """
            SELECT
                u.gaia_id,
                b.main_id,
                b.otype,
                b.otype_txt
            FROM TAP_UPLOAD.candidates AS u
                     JOIN ident AS i
                          ON i.id = u.gaia_id
                     JOIN basic AS b
                          ON b.oid = i.oidref \
            """

    result = Simbad.query_tap(
        query,
        candidates=upload,
    )

    if result is None or len(result) == 0:
        return pd.DataFrame(
            columns=[
                "source_id",
                "main_id",
                "otype",
                "otype_txt",
            ]
        )

    matches = result.to_pandas()

    matches["source_id"] = (
        matches["gaia_id"]
        .astype(str)
        .str.removeprefix("Gaia DR3 ")
        .astype("int64")
    )

    return (
        matches[
            ["source_id", "main_id", "otype", "otype_txt"]
        ]
        .drop_duplicates("source_id")
    )


def main():
    candidates = load_candidates()

    # Same star may appear in both top-100 lists.
    source_ids = sorted(
        candidates["source_id"].unique().tolist()
    )

    print(f"Unique candidates: {len(source_ids)}")

    reference = query_simbad(source_ids)

    result = candidates.merge(
        reference,
        on="source_id",
        how="left",
        validate="many_to_one",
    )

    result["simbad_found"] = result["main_id"].notna()

    # Keep the original anomaly rankings.
    result.to_csv(
        REPORT_DIR / "anomalies-simbad.csv",
        index=False,
        )

    print("\nSIMBAD matches:")
    print(
        result.groupby("method")["simbad_found"]
        .agg(["sum", "mean"])
    )

    print("\nKnown object types:")
    print(
        result.groupby(
            ["method", "otype_txt"],
            dropna=False,
        )
        .size()
        .sort_values(ascending=False)
        .head(30)
    )

    print("\nHighest-ranked IsolationForest candidates:")
    print(
        result[result["method"] == "IsolationForest"]
        .sort_values("if_rank")
        [
            [
                "source_id",
                "if_rank",
                "lof_rank",
                "main_id",
                "otype",
                "otype_txt",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
