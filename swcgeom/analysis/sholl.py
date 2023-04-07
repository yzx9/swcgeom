"""Sholl analysis."""

import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import Tree
from ..utils import draw_circles, get_fig_ax

__all__ = ["Sholl"]


class Sholl:
    """Sholl analysis.

    References
    ----------
    [1] Dendritic organization in the neurons of the visual and motor
    cortices of the cat J. Anat., 87 (1953), pp. 387-406
    """

    rs: npt.NDArray[np.float32]
    rmax: float

    # compat
    step: float | None = None

    def __init__(
        self,
        tree: Tree,
        step: Optional[float] = None,
    ) -> None:
        xyz = tree.get_segments().xyz() - tree.soma().xyz()  # shift
        self.rs = np.linalg.norm(xyz, axis=2)
        self.rmax = self.rs.max()

        if step is not None:
            warnings.warn(
                "`Sholl(x, step=...)` has been replaced by "
                "`Sholl(x).get(steps=...)` since v0.6.0 because it has "
                "been change to dynamic calculate, and will be removed "
                "in next version",
                DeprecationWarning,
            )
            self.step = step

    def get(self, steps: int | npt.ArrayLike = 20) -> npt.NDArray[np.int64]:
        xs, rs = self._get_rs(steps=steps), self.rs
        intersections = [np.logical_and(rs[:, 0] <= i, rs[:, 1] > i) for i in xs]
        return np.count_nonzero(intersections, axis=1)

    def intersect(self, r: float) -> int:
        return np.count_nonzero(np.logical_and(self.rs[:, 0] <= r, self.rs[:, 1] > r))

    def plot(  # pylint: disable=too-many-arguments
        self,
        steps: List[float] | int = 20,
        plot_type: str | None = None,
        kind: Literal["bar", "linechart", "circles"] = "bar",
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot Sholl analysis.

        Parameters
        ----------
        steps : int or List[float], default to 20
            Steps of raius of circle. If steps is int, then it will be
            evenly divided into n radii.
        kind : "bar" | "linechart" | "circles", default `bar`
        fig : ~matplotlib.figure.Figure
        ax : ~matplotlib.axes.Axes
        """
        if plot_type is not None:
            warnings.warn(
                "`plot_type` has been renamed to `kind` since v0.5.0, "
                "and will be removed in next version",
                DeprecationWarning,
            )
            kind = plot_type  # type: ignore

        xs = self._get_rs(steps=steps)
        ys = self.get(steps=xs)
        fig, ax = get_fig_ax(fig, ax)
        match kind:
            case "bar":
                sns.barplot(x=xs, y=ys, ax=ax, **kwargs)
                ax.set_ylabel("Count of Intersections")
            case "linechart":
                sns.lineplot(x=xs, y=ys, ax=ax, **kwargs)
                ax.set_ylabel("Count of Intersections")
            case "circles":
                kwargs.setdefault("y_min", 0)
                patches = draw_circles(ax, xs, ys, **kwargs)
                fig.colorbar(patches, ax=ax)
            case _:
                raise ValueError(f"unsupported kind: {kind}")

        ax.set_xlabel("Radial Distance")
        return fig, ax

    @staticmethod
    def get_rs(rmax: float, steps: int | npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Function to calculate the list of radius used by the sholl."""
        if isinstance(steps, int):
            s = rmax / (steps + 1)
            return np.arange(s, rmax, s)

        return np.array(steps)

    def _get_rs(self, steps: int | npt.ArrayLike) -> npt.NDArray[np.float32]:
        if self.step is not None:  # compat
            return np.arange(self.step, int(np.ceil(self.rmax)), self.step)

        return self.get_rs(self.rmax, steps)

    def get_count(self) -> npt.NDArray[np.int32]:
        warnings.warn(
            "`Sholl.get_count` has been renamed to `get` since v0.5.0, "
            "and will be removed in next version",
            DeprecationWarning,
        )
        return self.get().astype(np.int32)

    def avg(self) -> float:
        warnings.warn(
            "`Sholl.avg` has been deprecated since v0.6.0 and will be "
            "removed in next version, use `Shool(x).get().mean()` "
            "instead",
            DeprecationWarning,
        )
        return self.get().mean()

    def std(self) -> float:
        warnings.warn(
            "`Sholl.std` has been deprecate since v0.6.0 and will be "
            "removed in next version, use `Shool(x).get().std()` "
            "instead",
            DeprecationWarning,
        )
        return self.get().std()

    def sum(self) -> int:
        warnings.warn(
            "`Sholl.sum` has been deprecate since v0.6.0 and will be "
            "removed in next version, use `Shool(x).get().sum()` "
            "instead",
            DeprecationWarning,
        )
        return self.get().sum()
