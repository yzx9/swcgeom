from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection


@dataclass
class Palette:
    momo: str = "#F596AA"
    mizugaki: str = "#B9887D"


palette = Palette()


def draw_lines(
    lines: npt.NDArray[np.floating], ax: plt.Axes | None = None, **kwargs
) -> tuple[plt.Axes, LineCollection]:
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
        fig, ax = plt.subplots(1, 1)
    ax.add_collection(collection)
    ax.set_aspect(1)
    ax.autoscale()
    ax.axis("off")
    if ax is None:
        plt.show()

    return ax, collection
