"""Painter utils."""


import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..core import Branch, Tree
from ..transforms import BranchStandardizer
from ..utils import draw_lines, palette

__all__ = ["draw", "draw_branch"]


def draw(
    tree: Tree, color: str | None = palette.momo, ax: Axes | None = None, **kwargs
) -> tuple[Axes, LineCollection]:
    """Draw neuron tree.

    Parameters
    ----------
    color : str, optional
        Color of branch. If `None`, the default color will be enabled.
    ax : ~matplotlib.axes.Axes, optional
        A subplot of `~matplotlib`. If `None`, a new one will be created.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.

    Returns
    -------
    ax : ~matplotlib.axes.Axes
        If provided, return as-is.
    collection : ~matplotlib.collections.LineCollection
        Drawn line collection.
    """
    xyz = tree.xyz()  # (N, 3)
    segments = np.array([xyz[range(tree.number_of_nodes())], xyz[tree.pid()]])
    return draw_lines(segments, ax=ax, color=color, **kwargs)


def draw_branch(
    branch: Branch,
    color: str = palette.mizugaki,
    ax: Axes | None = None,
    standardize: bool = True,
    **kwargs,
) -> tuple[Axes, LineCollection]:
    """Draw neuron branch.

    Parameters
    ----------
    color : str, optional
        Color of branch. If `None`, the default color will be enabled.
    ax : ~matplotlib.axes.Axes, optional
        A subplot. If `None`, a new one will be created.
    standardize : bool, default `True`
        Standardize branch, see also self.standardize.
    **kwargs : dict[str, Any]
        Forwarded to `matplotlib.collections.LineCollection`.

    Returns
    -------
    ax : ~matplotlib.axes.Axes
        If provided, return as-is.
    collection : ~matplotlib.collections.LineCollection
        Drawn line collection.
    """

    xyz = branch.xyz()
    if standardize:
        T = BranchStandardizer.get_branch_standardize_matrix(xyz)
        xyz = np.dot(xyz, T)

    segments = np.array([xyz[:-1], xyz[1:]]).swapaxes(0, 1)
    return draw_lines(segments, color=color, ax=ax, **kwargs)
