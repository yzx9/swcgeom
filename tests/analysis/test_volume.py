"""Test volume."""

from io import StringIO

import numpy as np

from swcgeom.analysis.volume import get_volume
from swcgeom.core import Tree


def test_volume():
    swc = """
1 1 0 0 0 1 -1
2 1 2 0 0 1  1
3 1 4 0 0 1  2
"""

    tree = Tree.from_swc(StringIO(swc))
    volume = get_volume(tree)
    assert np.isclose(volume, 16 / 3 * np.pi)
