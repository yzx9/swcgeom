"""Painter utils."""


import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..core import Branch, SWCLike
from ..transforms import BranchStandardizer
from ..utils import draw_lines, palette

__all__ = ["draw"]


def draw(
    swc: SWCLike,
    color: str | None = palette.momo,
    ax: Axes | None = None,
    standardize: bool = True,
    **kwargs,
) -> tuple[Axes, LineCollection]:
    """Draw neuron tree.

    Parameters
    ----------
    color : str, optional
        Color of branch. If `None`, the default color will be enabled.
    ax : ~matplotlib.axes.Axes, optional
        A subplot of `~matplotlib`. If `None`, a new one will be created.
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
    return draw_lines(segments, ax=ax, color=color, **kwargs)
