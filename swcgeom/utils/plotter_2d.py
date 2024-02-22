"""Rendering related utils."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from swcgeom.utils.renderer import Camera
from swcgeom.utils.transforms import to_homogeneous, translate3d

__all__ = ["draw_lines", "draw_direction_indicator", "draw_circles", "get_fig_ax"]


def draw_lines(
    ax: Axes, lines: npt.NDArray[np.floating], camera: Camera, **kwargs
) -> LineCollection:
    """Draw lines.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    camera : Camera
        Camera position.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.
    """

    T = camera.MVP
    T = translate3d(*camera.position).dot(T)  # keep origin

    starts, ends = lines[:, 0], lines[:, 1]
    starts, ends = to_homogeneous(starts, 1), to_homogeneous(ends, 1)
    starts, ends = np.dot(T, starts.T).T[:, 0:2], np.dot(T, ends.T).T[:, 0:2]

    edges = np.stack([starts, ends], axis=1)
    return ax.add_collection(LineCollection(edges, **kwargs))  # type: ignore


def draw_direction_indicator(
    ax: Axes, camera: Camera, loc: Tuple[float, float]
) -> None:
    x, y = loc
    direction = camera.MV.dot(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )

    arrow_length, text_offset = 0.05, 0.05  # TODO: may still overlap
    text_colors = [["x", "red"], ["y", "green"], ["z", "blue"]]
    for (dx, dy, dz, _), (text, color) in zip(direction, text_colors):
        if 1 - abs(dz) < 1e-5:
            continue

        ax.arrow(
            x,
            y,
            arrow_length * dx,
            arrow_length * dy,
            head_length=0.02,
            head_width=0.01,
            color=color,
            transform=ax.transAxes,
        )

        ax.text(
            x + (arrow_length + text_offset) * dx,
            y + (arrow_length + text_offset) * dy,
            text,
            color=color,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )


def draw_circles(
    ax: Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    cmap: str | Colormap = "viridis",
) -> PatchCollection:
    """Draw a sequential of circles."""

    y_min = y.min() if y_min is None else y_min
    y_max = y.max() if y_max is None else y_max
    norm = Normalize(y_min, y_max)

    color_map = cmap if isinstance(cmap, Colormap) else cm.get_cmap(name=cmap)
    colors = color_map(norm(y))

    circles = [
        Circle((0, 0), xi, color=color) for xi, color in reversed(list(zip(x, colors)))
    ]
    patches = PatchCollection(circles, match_original=True)
    patches.set_cmap(color_map)
    patches.set_norm(norm)
    patches: PatchCollection = ax.add_collection(patches)  # type: ignore

    ax.set_aspect(1)
    ax.autoscale()
    return patches


def get_fig_ax(
    fig: Optional[Figure] = None, ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    if fig is None and ax is not None:
        fig = ax.get_figure()
        assert fig is not None, "expecting a figure from the axes"

    fig = fig or plt.gcf()
    ax = ax or plt.gca()
    return fig, ax
