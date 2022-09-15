"""Painter utils."""

from dataclasses import dataclass
from typing import Tuple, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

import numpy as np
import numpy.typing as npt

__all__ = ["palette", "get_fig_ax", "draw_lines", "draw_xyz_axes"]


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


def get_fig_ax(
    fig: Figure | None = None, ax: Axes | None = None
) -> Tuple[Figure, Axes]:
    fig = cast(Figure, plt.gcf()) if fig is None else fig
    ax = cast(Axes, fig.gca()) if ax is None else ax
    return fig, ax


def draw_lines(ax: Axes, lines: npt.NDArray[np.floating], **kwargs) -> LineCollection:
    """Draw lines.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.
    """

    starts, ends = lines[:, 0], lines[:, 1]

    # transform
    # TODO: support camare view
    starts = starts[:, 0:2]
    ends = ends[:, 0:2]

    edges = np.stack((starts, ends), axis=1)
    collection = LineCollection(edges, **kwargs)  # type: ignore

    ax.add_collection(collection)  # type: ignore
    return collection


def draw_xyz_axes(ax: Axes) -> None:
    x, y = 0.85, 0.05

    dx = [0.1, 0]
    dy = [0, 0.1]

    settings = [{"text": "x", "color": "red"}, {"text": "y", "color": "green"}]

    for i, setting in enumerate(settings):
        ax.arrow(
            x,
            y,
            dx[i],
            dy[i],
            head_length=0.04,
            head_width=0.03,
            color=setting["color"],
            transform=ax.transAxes,
        )
        ax.text(
            x + dx[i] + 0.03,
            y + dy[i] + 0.03,
            setting["text"],
            color=setting["color"],
            transform=ax.transAxes,
        )
