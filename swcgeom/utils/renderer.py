"""Painter utils."""

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

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
    "get_fig_ax",
]

Vector3D = Tuple[float, float, float]
Camera = NamedTuple("Camera", position=Vector3D, look_at=Vector3D, up=Vector3D)


@dataclass
class Palette:
    """Palette dataclasss."""

    momo: str = "#F596AA"
    mizugaki: str = "#B9887D"
    kuchiba: str = "#E2943B"
    kimirucha: str = "#867835"
    aotake: str = "#00896C"
    tsuyukusa: str = "#2EA9DF"
    sumire: str = "#66327C"
    benikeshinezumi: str = "#52433D"


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
    ax: Axes, camera: Camera, position: Tuple[float, float] = (0.85, 0.85)
) -> None:
    x, y = position
    arrow_length = 0.1

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
        if 1 - abs(dz) < 1e-2:
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
            x + (arrow_length + 0.1) * dx,
            y + (arrow_length + 0.1) * dy,
            text,
            color=color,
            transform=ax.transAxes,
        )


def get_fig_ax(
    fig: Figure | None = None, ax: Axes | None = None
) -> Tuple[Figure, Axes]:
    fig = plt.gcf() if fig is None else fig
    ax = fig.gca() if ax is None else ax
    return fig, ax
