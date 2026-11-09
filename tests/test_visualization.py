import numpy as np

from visualization import get_4_class_colormap
from config import NUM_CLASSES


def test_colormap_4_class():
    cmap, norm = get_4_class_colormap()
    assert cmap.N == NUM_CLASSES
    # Check boundaries: -0.5, 0.5, 1.5, 2.5, 3.5
    np.testing.assert_array_equal(norm.boundaries, np.array([-0.5, 0.5, 1.5, 2.5, 3.5]))
