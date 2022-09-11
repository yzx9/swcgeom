"""Transformation in branch."""

from typing import cast

import numpy as np
import numpy.typing as npt

from ..core import Branch
from ..utils import get_branch_standardize_matrix
from .base import Transform

__all__ = ["BranchResamplerLinear", "BranchStandardizer"]


class _BranchResampler(Transform[Branch, Branch]):
    r"""Resample branch."""

    def __call__(self, x: Branch) -> Branch:
        xyzr = x.xyzr()
        new_xyzr = self.resample(xyzr)
        return Branch.from_numpy(new_xyzr)

    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError()


class BranchResamplerLinear(_BranchResampler):
    r"""Resampling by linear interpolation, DO NOT keep original node."""

    def __init__(self, n_nodes: int) -> None:
        """Resample branch to special num of nodes.

        Parameters
        ----------
        n_nodes : int
            Number of nodes after resample.
        """
        super().__init__()
        self.n_nodes = n_nodes

    def __repr__(self) -> str:
        return f"BranchResamplerLinear-{self.n_nodes}"

    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Resampling by linear interpolation, DO NOT keep original node.

        Parameters
        ----------
        xyzr : np.ndarray[np.float32]
            The array of shape (N, 4).

        Returns
        -------
        coordinates : ~numpy.NDArray[float64]
            An array of shape (n_nodes, 4).
        """

        xp = np.cumsum(np.linalg.norm(xyzr[1:, :3] - xyzr[:-1, :3], axis=1))
        xp = np.insert(xp, 0, 0)
        xvals = np.linspace(0, xp[-1], self.n_nodes)

        x = np.interp(xvals, xp, xyzr[:, 0])
        y = np.interp(xvals, xp, xyzr[:, 1])
        z = np.interp(xvals, xp, xyzr[:, 2])
        r = np.interp(xvals, xp, xyzr[:, 3])
        return cast(npt.NDArray[np.float32], np.stack([x, y, z, r], axis=1))


class BranchStandardizer(Transform[Branch, Branch]):
    r"""Standarize branch.

    Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y,
    and scale max radius to 1.
    """

    def __call__(self, x: Branch) -> Branch:
        xyzr = x.xyzr()
        xyz, r = xyzr[:, 0:3], xyzr[:, 3:4]
        T = get_branch_standardize_matrix(xyzr)

        ones = np.ones([xyz.shape[0], 1])
        xyz4 = np.concatenate([xyz, ones], axis=1).transpose()  # (4, N)
        new_xyz = np.dot(T, xyz4)[0:3, :].transpose()
        new_xyzr = np.concatenate([new_xyz, r / r.max()], axis=1)
        return Branch.from_numpy(new_xyzr)
