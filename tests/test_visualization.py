import sys
from pathlib import Path
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from visualization import get_4_class_colormap
from config import VIS_PALETTE_4_CLASS, NUM_CLASSES


def test_colormap_4_class():
    cmap, norm = get_4_class_colormap()
    assert cmap.N == NUM_CLASSES
    # Check boundaries: -0.5, 0.5, 1.5, 2.5, 3.5
    np.testing.assert_array_equal(norm.boundaries, np.array([-0.5, 0.5, 1.5, 2.5, 3.5]))
