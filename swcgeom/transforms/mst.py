"""Minimum spanning tree."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy import ma
from numpy import typing as npt

from swcgeom.core import Tree, sort_tree
from swcgeom.core.swc_utils import SWCNames, SWCTypes, get_names, get_types
from swcgeom.transforms.base import Transform

__all__ = ["PointsToCuntzMST", "PointsToMST"]


class PointsToCuntzMST(Transform[npt.NDArray[np.float32], Tree]):
    """Create tree from points.

    Creates trees corresponding to the minimum spanning tree keeping
    the path length to the root small (with balancing factor bf).

    References
    ----------
    [1] Cuntz, H., Forstner, F., Borst, A. & Häusser, M. One Rule to
    Grow Them Al: A General Theory of Neuronal Branching and Its
    Practical Application. PLOS Comput Biol 6, e1000877 (2010).
    [2] Cuntz, H., Borst, A. & Segev, I. Optimization principles of
    dendritic structure. Theor Biol Med Model 4, 21 (2007).
    """

    def __init__(
        self,
        *,
        bf: float = 0.4,
        furcations: int = 2,
        exclude_soma: bool = True,
        sort: bool = True,
        names: Optional[SWCNames] = None,
        types: Optional[SWCTypes] = None,
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
        names : SWCNames, optional
        types : SWCTypes, optional
        """
        self.bf = np.clip(bf, 0, 1)
        self.furcations = furcations
        self.exclude_soma = exclude_soma
        self.sort = sort
        self.names = get_names(names)
        self.types = get_types(types)

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
        if names is None:
            names = self.names
        else:
            warnings.warn(
                "`PointsToCuntzMST(...)(names=...)` has been "
                "replaced by `PointsToCuntzMST(...,names=...)` since "
                "v0.12.0, and will be removed in next version",
                DeprecationWarning,
            )
            names = get_names(names)  # TODO: remove it

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
            names.type: np.full(n, fill_value=self.types.glia_processes),
            names.x: points[:, 0],
            names.y: points[:, 1],
            names.z: points[:, 2],
            names.r: 1,
            names.pid: pid,
        }
        dic[names.type][0] = self.types.soma
        df = pd.DataFrame.from_dict(dic)
        t = Tree.from_data_frame(df, names=names)
        if self.sort:
            t = sort_tree(t)
        return t

    def extra_repr(self) -> str:  # TODO: names, types
        return f"bf={self.bf:.4f}, furcations={self.furcations}, exclude_soma={self.exclude_soma}, sort={self.sort}"


class PointsToMST(PointsToCuntzMST):  # pylint: disable=too-few-public-methods
    """Create minimum spanning tree from points."""

    def __init__(
        self,
        furcations: int = 2,
        *,
        k_furcations: Optional[int] = None,
        exclude_soma: bool = True,
        names: Optional[SWCNames] = None,
        types: Optional[SWCTypes] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        furcations : int, default `2`
            Suppress multifurcations which more than k. If set to -1,
            no suppression.
        exclude_soma : bool, default `True`
            Suppress multi-furcations exclude soma.
        names : SWCNames, optional
        types : SWCTypes, optional
        """

        if k_furcations is not None:
            warnings.warn(
                "`PointsToMST(k_furcations=...)` has been renamed to "
                "`PointsToMST(furcations=...)` since v0.12.0, and will "
                "be removed in next version",
                DeprecationWarning,
            )
            furcations = k_furcations

        super().__init__(
            bf=0,
            furcations=furcations,
            exclude_soma=exclude_soma,
            names=names,
            types=types,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"furcations-{self.furcations}, exclude-soma={self.exclude_soma}"
