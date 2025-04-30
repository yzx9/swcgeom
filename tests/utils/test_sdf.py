# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Test SDF class."""

import numpy as np
import numpy.testing as npt
import pytest

from swcgeom.utils.sdf import SDFFrustumCone, SDFSphere


class TestSphere:
    """Test SDFSphere."""

    @pytest.mark.parametrize(
        "center, raidus, p, expected",
        [
            ((0, 0, 0), 1, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [-1, 0, 1]),
            ((1, 1, 1), 1, [[0, 0, 0], [1, 0, 0]], [np.sqrt(3) - 1, np.sqrt(2) - 1]),
        ],
    )  # fmt: skip
    def test_distance(self, center, raidus, p, expected):
        sphere = SDFSphere(center, raidus)
        npt.assert_allclose(sphere.distance(p), expected)

    @pytest.mark.parametrize(
        "center, raidus, p, expected",
        [
            ((0, 0, 0), 1, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, True, False]),
            ((1, 1, 1), 1, [[0, 0, 0], [1, 0, 0]], [False, False]),
        ],
    )  # fmt: skip
    def test_is_in(self, center, raidus, p, expected):
        sphere = SDFSphere(center, raidus)
        npt.assert_equal(sphere.is_in(p), expected)


class TestSDFFrustumCone:
    """Test SDFFrustumCone."""

    @pytest.mark.parametrize(
        "a, b, ra, rb, p, expected",
        [
            ((0, 0, 0), (0, 0, 2), 2, 1,
                [[0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
                [        1 ,        0 ,       -1 ,        0 ,        1 ]),
        ],
    )  # fmt: skip
    def test_distance(self, a, b, ra, rb, p, expected):
        frustum_cone = SDFFrustumCone(a, b, ra, rb)
        npt.assert_allclose(frustum_cone.distance(p), expected)

    @pytest.mark.parametrize(
        "a, b, ra, rb, p, expected",
        [
            ((0, 0, 0), (0, 0, 2), 2, 1,
                [[0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
                [    False ,     True ,     True ,     True ,    False ]),
        ],
    )  # fmt: skip
    def test_is_in(self, a, b, ra, rb, p, expected):
        frustum_cone = SDFFrustumCone(a, b, ra, rb)
        npt.assert_allclose(frustum_cone.is_in(p), expected)
