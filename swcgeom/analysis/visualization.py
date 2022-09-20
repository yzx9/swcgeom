"""Painter utils."""


from typing import Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import Branch, SWCLike
from ..transforms import BranchStandardizer
from ..utils import draw_lines, draw_xyz_axes, get_fig_ax, palette

__all__ = ["draw"]

DEFAULT_COLOR = palette.momo


def draw(
    swc: SWCLike,
    *,
    color: Dict[int, str] | str | None = None,
    standardize: bool = True,
    first: bool = True,
    fig: Figure | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Draw neuron tree.

    Parameters
    ----------
    color : Dict[int, str] | str, optional
        Color map. If is dict, segments will be colored by the type of
        parent node.If is string, the value will be use for any type.
    ax : ~matplotlib.axes.Axes, optional
        A subplot of `~matplotlib`. If `None`, a new one will be
        created.
    standardize : bool, default `True`
        Standardize input, enable for branch only.
    first : bool, default to `True`
        If multiple neuron plotted on same axes, set to `False` on
        subsequent calls.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.

    Returns
    -------
    ax : ~matplotlib.axes.Axes
        If provided, return as-is.
    collection : ~matplotlib.collections.LineCollection
        Drawn line collection.
    """

    if standardize and isinstance(swc, Branch):
        standardizer = BranchStandardizer()
        swc = standardizer(swc)

    if isinstance(color, dict):
        types = swc.type()[:-1]  # colored by type of parent node
        color_map = list(map(lambda t: color.get(t, DEFAULT_COLOR), types))
    else:
        color_map = color if color is not None else DEFAULT_COLOR

    if callable((get_segments := getattr(swc, "get_segments", None))):
        lines = get_segments().xyz()
    else:
        xyz = swc.xyz()
        starts, ends = swc.id()[1:], swc.pid()[1:]
        lines = np.stack([xyz[starts], xyz[ends]], axis=1)

    fig, ax = get_fig_ax(fig, ax)
    draw_lines(ax, lines, color=color_map, **kwargs)
    ax.autoscale()
    if first:
        ax.set_aspect(1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.text(0.05, 0.95, r"$\mu m$", transform=ax.transAxes)

        draw_xyz_axes(ax)

    return fig, ax
