import numpy as np

from gaia_style import marker_size, star_rgb_from_bprp


def test_rgb_shape_and_clipping():
    colors = star_rgb_from_bprp(
        np.array([-10.0, -0.5, 4.0, 99.0])
    )
    assert colors.shape == (4, 3)
    assert colors.dtype == np.uint8
    assert np.array_equal(colors[0], colors[1])
    assert np.array_equal(colors[2], colors[3])


def test_marker_size_increases_with_brightness():
    brightness = np.array([0.0, 0.25, 0.5, 1.0])
    sizes = marker_size(brightness)
    assert np.all(np.diff(sizes) > 0)
    assert sizes[0] == 0.4
    assert sizes[-1] == 2.0
