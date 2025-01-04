# Copyright 2022-2025 Zexin Yuan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
