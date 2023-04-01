"""Minimum spanning tree."""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy import ma

from ..core import Tree
from ..core.swc_utils import SWCNames, get_names
from .base import Transform

__all__ = ["PointsToCuntzMST", "PointsToMST"]


class PointsToCuntzMST(Transform[npt.NDArray[np.float32], Tree]):
    """Create tree from points.

    Creates trees corresponding to the minimum spanning tree keeping
    the path length to the root small (with balancing factor bf).

    References
    ----------
    [1] Cuntz, Hermann, Alexander Borst, and Idan Segev. “Optimization
    Principles of Dendritic Structure.” Theoretical Biology and Medical
    Modelling 4, no. 1 (June 8, 2007): 21. https://doi.org/10.1186/1742-4682-4-21.
    """

    def __init__(
        self, *, bf: float = 0.4, furcations: int = 2, exclude_soma: bool = True
    ) -> None:
        """
        Parameters
        ----------
        bf : float, default `0.4`
            Balancing factor between 0~1.
        furcations : int, default `2`
            Suppress multi-furcations which more than k. If set to -1,
            no suppression.
        exclude_soma : bool, default `True`
            Suppress multi-furcations exclude soma.
        """
        self.bf = np.clip(bf, 0, 1)
        self.furcations = furcations
        self.exclude_soma = exclude_soma

    def __call__(  # pylint: disable=too-many-locals
        self,
        points: npt.NDArray[np.floating],
        soma: Optional[npt.ArrayLike] = None,
        *,
        names: Optional[SWCNames] = None,
    ) -> Tree:
        """
        Paramters
        ---------
        points : array of shape (N, 3)
            Positions of points cloud.
        soma : array of shape (3,), default `None`
            Position of soma. If none, use the first point as soma.
        names : SWCNames, optional
        """
        names = get_names(names)
        if soma is not None:
            soma = np.array(soma)
            assert soma.shape == (3,)
            points = np.concatenate([[soma], points])

        n = points.shape[0]
        dis = np.linalg.norm(
            points.reshape((-1, 1, 3)) - points.reshape((1, -1, 3)), axis=2
        )

        pid = np.full(n, fill_value=-1)
        acc = np.zeros(n)
        furcations = np.zeros(n, dtype=np.int32)
        conn = np.zeros(n, dtype=np.bool_)  # connect points
        conn[0] = True
        mask = np.ones((n, n), dtype=np.bool_)
        mask[0, :] = False
        mask[0, 0] = True
        for _ in range(n - 1):  # for tree: e = n-1
            cost = ma.array(dis + self.bf * acc, mask=mask)
            (i, j) = np.unravel_index(cost.argmin(), cost.shape)

            furcations[i] += 1
            if (
                self.furcations != -1
                and furcations[i] >= self.furcations
                and (not self.exclude_soma or i != 0)
            ):
                mask[i, :] = True  # avoid it link to any point
                mask[:, i] = True  # avoid any point lint to it

            pid[j] = i
            acc[j] = acc[i] + dis[i, j]
            conn[j] = True
            mask[j, :] = conn  # enable link to other points
            mask[:, j] = True  # avoid any point lint to it

        dic = {
            names.id: np.arange(n),
            names.type: np.full(n, fill_value=7),  # TODO
            names.x: points[:, 0],
            names.y: points[:, 1],
            names.z: points[:, 2],
            names.r: 1,
            names.pid: pid,
        }
        dic[names.type][0] = 1
        df = pd.DataFrame.from_dict(dic)
        return Tree.from_data_frame(df, names=names)

    def __repr__(self) -> str:
        return (
            f"PointsToCuntzMST"
            f"-bf-{self.bf}"
            f"-furcations-{self.furcations}"
            f"-{'exclude-soma' if self.exclude_soma else 'include-soma'}"
        )


class PointsToMST(PointsToCuntzMST):  # pylint: disable=too-few-public-methods
    """Create minimum spanning tree from points."""

    def __init__(self, k_furcations: int = 2) -> None:
        """
        Parameters
        ----------
        k_furcations : int, default `2`
            Suppress multifurcations which more than k. If set to -1,
            no suppression.
        """
        super().__init__(bf=0, furcations=k_furcations)

    def __repr__(self) -> str:
        return (
            f"PointsToMST"
            f"-furcations-{self.furcations}"
            f"-{'exclude-soma' if self.exclude_soma else 'include-soma'}"
        )
