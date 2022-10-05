"""Sholl analysis."""

import math
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import Tree
from ..utils import draw_circles, get_fig_ax

__all__ = ["Sholl"]


class Sholl:
    """Sholl analysis.

    .. [1] Dendritic organization in the neurons of the visual and motor
       cortices of the cat J. Anat., 87 (1953), pp. 387-406
    """

    count: npt.NDArray[np.int32]
    step: float

    def __init__(self, tree: Tree, step: float = 1) -> None:
        xyz = tree.get_segments().xyz() - tree.soma().xyz()  # shift
        radius = np.linalg.norm(xyz, axis=2)
        max_radius = math.ceil(np.max(radius[:, 1]))
        count = [
            np.count_nonzero(np.logical_and(radius[:, 0] <= i, radius[:, 1] > i))
            for i in np.arange(0, max_radius, step)
        ]

        self.count = np.array(count, dtype=np.int32)
        self.step = step

    def __getitem__(self, idx: int) -> int:
        return self.count[idx] if 0 <= idx < len(self.count) else 0

    def get_count(self) -> npt.NDArray[np.int32]:
        return self.count.copy()

    def get_distribution(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        y = self.get_count()
        x = self.step * np.arange(len(y))
        return x, y

    def avg(self) -> float:
        return np.average(self.get_count()).item()

    def std(self) -> float:
        return np.std(self.get_count()).item()

    def sum(self) -> int:
        return np.sum(self.get_count()).item()

    def plot(
        self,
        plot_type: Literal["bar", "linechart", "circles"] = "linechart",
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        fig, ax = get_fig_ax(fig, ax)
        x, y = self.get_distribution()

        match plot_type:
            case "bar":
                kwargs.setdefault("width", self.step)
                ax.bar(x, y, **kwargs)
            case "linechart":
                ax.plot(x, y, **kwargs)
            case "circles":
                kwargs.setdefault("y_min", 0)
                draw_circles(fig, ax, x, y, **kwargs)
            case _:
                raise ValueError(f"unsupported plot type: {plot_type}")

        return fig, ax
