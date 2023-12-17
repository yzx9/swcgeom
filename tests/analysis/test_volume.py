"""Test volume."""

from io import StringIO

import numpy as np
import numpy.testing as npt
import pytest

from swcgeom.analysis.volume import get_volume
from swcgeom.core import Tree


class TestVolume:
    @pytest.mark.parametrize(
        "swc, expected",
        [
            (
                """
1 1 0 0 0 1 -1
2 1 2 0 0 1  1
3 1 4 0 0 1  2
""",
                16 / 3 * np.pi,
            ),
            (
                """
1 1 0 0 0 1 -1
2 1 2 0 0 1  1
3 1 2 0 0 1  1
""",
                12 / 3 * np.pi,
            ),
        ],
    )
    def test_close(self, swc, expected):
        tree = Tree.from_swc(StringIO(swc))
        volume = get_volume(tree)
        npt.assert_allclose(volume, expected, rtol=1e-3)

    @pytest.mark.parametrize(
        "swc, lowbound, upbound",
        [
            (
                """
1 1 0 0 0 1 -1
2 1 2 0 0 1  1
3 1 0 2 0 1  1
""",
                (16 / 3) * np.pi - 1,
                (16 / 3) * np.pi,
            ),
        ],
    )
    def test_between(self, swc, lowbound, upbound):
        tree = Tree.from_swc(StringIO(swc))
        volume = get_volume(tree)
        assert lowbound <= volume <= upbound
