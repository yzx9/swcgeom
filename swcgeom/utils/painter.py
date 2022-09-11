"""Painter utils."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

import numpy as np
import numpy.typing as npt

from ..core import Branch, Tree
from .branch import get_branch_standardize_matrix

__all__ = ["palette", "draw_branch", "draw_tree"]


@dataclass
class Palette:
    """Palette dataclasss."""

    momo: str = "#F596AA"
    mizugaki: str = "#B9887D"


palette = Palette()


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
        T = get_branch_standardize_matrix(xyz)
        xyz = np.dot(xyz, T)

    lines = np.array([xyz[:-1], xyz[1:]]).swapaxes(0, 1)
    return _draw_lines(lines, color=color, ax=ax, **kwargs)


def draw_tree(
    self: Tree, color: str | None = palette.momo, ax: Axes | None = None, **kwargs
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
    xyz = self.xyz()  # (N, 3)
    edges = np.array([xyz[range(self.number_of_nodes())], xyz[self.pid()]])
    return _draw_lines(edges, ax=ax, color=color, **kwargs)


def _draw_lines(
    lines: npt.NDArray[np.floating], ax: Axes | None = None, **kwargs
) -> tuple[Axes, LineCollection]:
    """Draw lines.

    Parameters
    ----------
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    ax : ~matplotlib.axes.Axes, optional
        A subplot. If `None`, a new one will be created.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.

    Returns
    -------
    ax : ~matplotlib.axes.Axes
        If provided, return as-is.
    collection : ~matplotlib.collections.LineCollection
        Drawn line collection.
    """

    starts = lines[:, 0]
    ends = lines[:, 1]

    # transform
    # TODO: support camare view
    starts = starts[:, 0:2]
    ends = ends[:, 0:2]

    edges = np.stack((starts, ends), axis=1)
    collection = LineCollection(edges, **kwargs)  # type: ignore

    if ax is None:
        fig, ax = plt.subplots(1, 1)  # pylint: disable=unused-variable
    ax.add_collection(collection)  # type: ignore
    ax.set_aspect(1)
    ax.autoscale()
    ax.axis("off")
    if ax is None:
        plt.show()

    return ax, collection
