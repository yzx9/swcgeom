"""Sholl analysis."""

import math
import warnings
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

    count: npt.NDArray[np.int64]
    step: float
    steps: npt.NDArray[np.float32]

    def __init__(self, tree: Tree, step: float = 1) -> None:
        xyz = tree.get_segments().xyz() - tree.soma().xyz()  # shift
        r = np.linalg.norm(xyz, axis=2)
        steps = np.arange(step, int(np.ceil(r.max())), step)
        intersections = [np.logical_and(r[:, 0] <= i, r[:, 1] > i) for i in steps]
        count = np.count_nonzero(intersections, axis=1)

        self.count = count
        self.step = step
        self.steps = steps

    def __getitem__(self, idx: int) -> int:
        return self.count[idx] if 0 <= idx < len(self.count) else 0

    def get(self) -> npt.NDArray[np.int64]:
        return self.count.copy()

    def get_count(self) -> npt.NDArray[np.int32]:
        warnings.warn(
            "`Sholl.get_count` has been renamed to `get` since v0.5.0, "
            "and will be removed in next version",
            DeprecationWarning,
        )
        return self.get().astype(np.int32)

    def avg(self) -> float:
        warnings.warn(
            "`Sholl.avg` will be removed in next version",
            DeprecationWarning,
        )
        return self.count.mean()

    def std(self) -> float:
        warnings.warn(
            "`Sholl.std` will be removed in next version",
            DeprecationWarning,
        )
        return self.count.std()

    def sum(self) -> int:
        warnings.warn(
            "`Sholl.sum` will be removed in next version",
            DeprecationWarning,
        )
        return self.count.sum()

    def plot(
        self,
        plot_type: str | None = None,
        kind: Literal["bar", "linechart", "circles"] = "linechart",
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        if plot_type is not None:
            warnings.warn(
                "`plot_type` has been renamed to `kind` since v0.5.0, "
                "and will be removed in next version",
                DeprecationWarning,
            )
            kind = plot_type  # type: ignore

        x, y = self.steps, self.count
        fig, ax = get_fig_ax(fig, ax)
        match kind:
            case "bar":
                kwargs.setdefault("width", self.step)
                ax.bar(x, y, **kwargs)
            case "linechart":
                ax.plot(x, y, **kwargs)
            case "circles":
                kwargs.setdefault("y_min", 0)
                draw_circles(fig, ax, x, y, **kwargs)
            case _:
                raise ValueError(f"unsupported plot kind: {kind}")

        return fig, ax
