"""Transformation in branch."""

from typing import cast

import numpy as np
import numpy.typing as npt

from ..core import Branch
from ..utils import (
    angle,
    rotate3d_x,
    rotate3d_y,
    rotate3d_z,
    scale3d,
    to_homogeneous,
    translate3d,
)
from .base import Transform

__all__ = ["BranchLinearResampler", "BranchStandardizer"]


class _BranchResampler(Transform[Branch, Branch]):
    r"""Resample branch."""

    def __call__(self, x: Branch) -> Branch:
        xyzr = x.xyzr()
        new_xyzr = self.resample(xyzr)
        return Branch.from_xyzr(new_xyzr)

    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        raise NotImplementedError()


class BranchLinearResampler(_BranchResampler):
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
        return f"BranchLinearResampler-{self.n_nodes}"

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
        T = self.get_matrix(xyz)

        xyz4 = to_homogeneous(xyz, 1).transpose()  # (4, N)
        new_xyz = np.dot(T, xyz4)[:3].transpose()
        new_xyzr = np.concatenate([new_xyz, r / r.max()], axis=1)
        return Branch.from_xyzr(new_xyzr)

    @staticmethod
    def get_matrix(xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        r"""Get standarize transformation matrix.

        Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y.

        Parameters
        ----------
        xyz : np.ndarray[np.float32]
            The `x`, `y`, `z` matrix of shape (N, 3) of branch.

        Returns
        -------
        T : np.ndarray[np.float32]
            An homogeneous transfomation matrix of shape (4, 4).
        """

        assert (
            xyz.ndim == 2 and xyz.shape[1] == 3
        ), f"xyz should be of shape (N, 3), got {xyz.shape}"

        xyz = xyz[:, :3]
        T = np.identity(4)
        v = np.concatenate([xyz[-1] - xyz[0], np.zeros((1))])[:, None]

        # translate to the origin
        T = translate3d(-xyz[0, 0], -xyz[0, 1], -xyz[0, 2]).dot(T)

        # scale to unit vector
        s = (1 / np.linalg.norm(v[:3, 0])).item()
        T = scale3d(s, s, s).dot(T)

        # rotate v to the xz-plane, v should be (x, 0, z) now
        vy = np.dot(T, v)[:, 0]
        # when looking at the xz-plane along the positive y-axis, the
        # coordinates should be (z, x)
        T = rotate3d_y(angle([vy[2], vy[0]], [0, 1])).dot(T)

        # rotate v to the x-axis, v should be (1, 0, 0) now
        vx = np.dot(T, v)[:, 0]
        T = rotate3d_z(angle([vx[0], vx[1]], [1, 0])).dot(T)

        # rotate the farthest point to the xy-plane
        if xyz.shape[0] > 2:
            xyz4 = to_homogeneous(xyz, 1).transpose()  # (4, N)
            new_xyz4 = np.dot(T, xyz4)  # (4, N)
            max_index = np.argmax(np.linalg.norm(new_xyz4[1:3, :], axis=0)[1:-1]) + 1
            max_xyz4 = xyz4[:, max_index].reshape(4, 1)
            max_xyz4_t = np.dot(T, max_xyz4)  # (4, 1)
            angle_x = angle(max_xyz4_t[1:3, 0], [1, 0])
            T = rotate3d_x(angle_x).dot(T)

        return T
