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

    line_collection = Line3DCollection(lines, joinstyle=joinstyle, capstyle=capstyle, **kwargs)  # type: ignore
    return ax.add_collection3d(line_collection)
