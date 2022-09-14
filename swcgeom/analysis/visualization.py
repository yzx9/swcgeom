"""Painter utils."""


from typing import Dict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..core import Branch, SWCLike
from ..transforms import BranchStandardizer
from ..utils import draw_lines, palette

__all__ = ["draw"]

DEFAULT_COLOR = palette.momo


def draw(
    swc: SWCLike,
    color: Dict[int, str] | str | None = None,
    ax: Axes | None = None,
    standardize: bool = True,
    **kwargs,
) -> tuple[Axes, LineCollection]:
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

    xyz = np.stack([swc.x(), swc.y(), swc.z()], axis=1)  # (N, 3)
    starts, ends = swc.id()[1:], swc.pid()[1:]
    segments = np.stack([xyz[starts], xyz[ends]], axis=1)

    if isinstance(color, dict):
        types = swc.type()[:-1]  # colored by type of parent node
        color_map = list(map(lambda t: color.get(t, DEFAULT_COLOR), types))
    else:
        color_map = color if color is not None else DEFAULT_COLOR

    return draw_lines(segments, ax=ax, color=color_map, **kwargs)
