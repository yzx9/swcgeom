"""Painter utils."""

from typing import Dict, NamedTuple, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import numpy as np
import numpy.typing as npt

from .transforms import (
    model_view_trasformation,
    orthographic_projection_simple,
    to_homogeneous,
    translate3d,
)

__all__ = [
    "Vector3D",
    "Camera",
    "palette",
    "draw_lines",
    "draw_xyz_axes",
    "draw_circles",
    "get_fig_ax",
]

Vector3D = Tuple[float, float, float]
Camera = NamedTuple("Camera", position=Vector3D, look_at=Vector3D, up=Vector3D)


class Palette:
    default: Dict[int, str]
    vaa3d: Dict[int, str]

    def __init__(self):
        default = [
            "#F596AA",  # momo,
            "#867835",  # kimirucha,
            "#E2943B",  # kuchiba,
            "#00896C",  # aotake,
            "#B9887D",  # mizugaki,
            "#2EA9DF",  # tsuyukusa,
            "#66327C",  # sumire,
            "#52433D",  # benikeshinezumi,
        ]
        self.default = dict(enumerate(default))

        vaa3d = [
            "#ffffff",  # white, 0-undefined
            "#141414",  # black, 1-soma
            "#c81400",  # red, 2-axon
            "#0014c8",  # blue, 3-dendrite
            "#c800c8",  # purple, 4-apical dendrite
            # the following is Hanchuan’s extended color. 090331
            "#00c8c8",  # cyan, 5
            "#dcc800",  # yellow, 6
            "#00c814",  # green, 7
            "#bc5e25",  # coffee, 8
            "#b4c878",  # asparagus, 9
            "#fa6478",  # salmon, 10
            "#78c8c8",  # ice, 11
            "#6478c8",  # orchid, 12
            # the following is Hanchuan’s further extended color. 111003
            "#ff80a8",  # 13
            "#80ffa8",  # 14
            "#80a8ff",  # 15
            "#a8ff80",  # 16
            "#ffa880",  # 17
            "#a880ff",  # 18
            "#000000",  # 19 # totally black. PHC, 2012-02-15
            # the following (20-275) is used for matlab heat map. 120209 by WYN
            "#000083",
        ]
        self.vaa3d = dict(enumerate(vaa3d))


palette = Palette()


def draw_lines(
    lines: npt.NDArray[np.floating], ax: Axes, camera: Camera, **kwargs
) -> LineCollection:
    """Draw lines.

    Parameters
    ----------
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    ax : ~matplotlib.axes.Axes
    camera : Camera
        Camera position.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.
    """

    starts, ends = lines[:, 0], lines[:, 1]
    starts, ends = to_homogeneous(starts, 1), to_homogeneous(ends, 1)

    # model/view transformation
    T = model_view_trasformation(*camera)
    T = translate3d(*camera[0]).dot(T)  # keep origin

    # projection transformation
    T = orthographic_projection_simple().dot(T)

    starts = np.dot(T, starts.T).T[:, 0:2]
    ends = np.dot(T, ends.T).T[:, 0:2]

    edges = np.stack([starts, ends], axis=1)
    collection = LineCollection(edges, **kwargs)  # type: ignore
    ax.add_collection(collection)  # type: ignore
    return collection


def draw_xyz_axes(
    ax: Axes, camera: Camera, position: Tuple[float, float] = (0.8, 0.10)
) -> None:
    x, y = position
    arrow_length = 0.05

    direction = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )
    direction = model_view_trasformation(*camera).dot(direction)

    for (dx, dy, dz, _), (text, color) in zip(
        direction[:3],
        [["x", "red"], ["y", "green"], ["z", "blue"]],
    ):
        if 1 - abs(dz) < 1e-5:
            continue

        ax.arrow(
            x,
            y,
            arrow_length * dx,
            arrow_length * dy,
            head_length=0.04,
            head_width=0.03,
            color=color,
            transform=ax.transAxes,
        )
        ax.text(
            x + (arrow_length + 0.06) * dx,
            y + (arrow_length + 0.06) * dy,
            text,
            color=color,
            transform=ax.transAxes,
        )


def draw_circles(
    fig: Figure,
    ax: Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    /,
    y_min: float | None = None,
    y_max: float | None = None,
    cmap: str = "viridis",
) -> None:
    """Draw a sequential of circles."""
    y_min = y.min() if y_min is None else y_min
    y_max = y.max() if y_max is None else y_max
    norm = Normalize(y_min, y_max)

    color_map = cm.get_cmap(name=cmap)
    colors = color_map(norm(y))

    for xi, color in reversed(list(zip(x, colors))):
        circle = Circle((0, 0), xi, color=color)
        ax.add_patch(circle)

    sm = cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    fig.colorbar(sm)

    ax.set_aspect(1)
    ax.autoscale()


def get_fig_ax(
    fig: Figure | None = None, ax: Axes | None = None
) -> Tuple[Figure, Axes]:
    if fig is None:
        fig = plt.gcf() if ax is None else ax.get_figure()

    if ax is None:
        ax = fig.gca()

    return fig, ax
