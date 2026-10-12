import numpy as np
import pandas as pd
import pytest

from astrometry import add_distance, add_local_cartesian
from gaia import QueryConfig, build_query

def sample_data():
    return pd.DataFrame({
        "ra": [83.82, 83.90, 84.00],
        "dec": [-5.39, -5.30, -5.20],
        "parallax": [2.0, 4.0, 0.2],
    })

def test_build_query():
    config = QueryConfig(
        ra_deg=83.82,
        dec_deg=-5.39,
        radius_deg=2.5,
        row_limit=1234,
        ruwe_max=1.4,
        parallax_over_error_min=10.0,
    )
    query = build_query(config)
    assert "TOP 1234" in query
    assert "2.5" in query
    assert "ruwe < 1.4" in query
    assert "parallax_over_error >= 10.0" in query


def test_distance():
    result = add_distance(sample_data())
    assert result["distance_pc"].to_list() == pytest.approx([500.0, 250.0, 5000.0])

def test_depth_centered():
    result = add_local_cartesian(
        sample_data(),
        center_ra_deg=83.82,
        center_dec_deg=-5.39,
    )
    assert np.median(result["z"]) == pytest.approx(0)

def test_input_not_modified():
    original = sample_data()
    before = original.copy(deep=True)
    add_distance(original)
    pd.testing.assert_frame_equal(original, before)