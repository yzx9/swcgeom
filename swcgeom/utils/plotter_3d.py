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


"""3D Plotting utils."""

import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

__all__ = ["draw_lines_3d"]


def draw_lines_3d(
    ax: Axes3D,
    lines: npt.NDArray[np.floating],
    joinstyle="round",
    capstyle="round",
    **kwargs,
):
    """Draw lines.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    **kwargs : dict[str, Unknown]
        Forwarded to `~mpl_toolkits.mplot3d.art3d.Line3DCollection`.
    """

    line_collection = Line3DCollection(
        lines, joinstyle=joinstyle, capstyle=capstyle, **kwargs
    )  # type: ignore
    return ax.add_collection3d(line_collection)
