from astroquery.vizier import Vizier

import pandas as pd

REFERENCE_CATALOG = "J/A+A/677/A163/members"

def load_reference_pleiades() -> pd.DataFrame:
    vizier = Vizier(
        columns=["GaiaDR3", "Cluster"],
        row_limit=-1,
    )

    tables = vizier.get_catalogs(REFERENCE_CATALOG)

    reference = tables[0].to_pandas()

    # VizieR strings can occasionally arrive as bytes.
    reference["Cluster"] = reference["Cluster"].map(
        lambda value:
        value.decode()
        if isinstance(value, bytes)
        else str(value)
    )

    pleiades = reference[
        reference["Cluster"].str.strip().str.casefold()
        == "pleiades"
        ].copy()

    pleiades = pleiades.rename(
        columns={"GaiaDR3": "source_id"}
    )

    pleiades["source_id"] = (
        pleiades["source_id"]
        .astype("int64")
    )

    return pleiades[["source_id"]].drop_duplicates()

if __name__ == "__main__":
    reference = load_reference_pleiades()

    print(len(reference))
    print(reference.head())
