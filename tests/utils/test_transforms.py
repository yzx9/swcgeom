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


"""Test transforms."""

import numpy as np

from swcgeom.utils.transforms import to_homogeneous


def test_to_to_homogeneous():
    test_cases = [
        # fmt:off
        ((1, 2, 3), 1, (1, 2, 3, 1)),
        ((1, 2, 3), 0, (1, 2, 3, 0)),
        ((1, 2, 3, 4), 1, (1, 2, 3, 4)),
        ((1, 2, 3, 4), 0, (1, 2, 3, 4)),
        # fmt:on
    ]

    for xyz, w, except_xyz4 in test_cases:
        xyz4 = to_homogeneous(xyz, w)
        assert np.allclose(
            xyz4, except_xyz4
        ), f"Test Failed: Expected xyz4: {except_xyz4}, Got: {xyz4}"

    # Test batch
    xyz = np.array([[1, 2, 3], [4, 5, 6]])
    xyz4 = to_homogeneous(xyz, 1)
    assert xyz4.shape == (2, 4)
    assert np.allclose(xyz4[..., :3], xyz)

    xyz = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    xyz4 = to_homogeneous(xyz, 1)
    assert xyz4.shape == (2, 2, 4)
    assert np.allclose(xyz4[..., :3], xyz)
