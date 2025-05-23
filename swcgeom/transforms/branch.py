
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Transformation in branch."""

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import signal
from typing_extensions import override

from swcgeom.core import Branch, DictSWC
from swcgeom.transforms.base import Transform
from swcgeom.utils import (
    angle,
    rotate3d_x,
    rotate3d_y,
    rotate3d_z,
    scale3d,
    to_homogeneous,
    translate3d,
)

__all__ = ["BranchLinearResampler", "BranchConvSmoother", "BranchStandardizer"]


class _BranchResampler(Transform[Branch, Branch], ABC):
    r"""Resample branch."""

    @override
    def __call__(self, x: Branch) -> Branch:
        xyzr = self.resample(x.xyzr())
        return Branch.from_xyzr(xyzr)

    @abstractmethod
    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...


class BranchLinearResampler(_BranchResampler):
    r"""Resampling by linear interpolation, DO NOT keep original node."""

    def __init__(self, n_nodes: int) -> None:
        """Resample branch to special num of nodes.

        Args:
            n_nodes: Number of nodes after resample.
        """
        super().__init__()
        self.n_nodes = n_nodes

    @override
    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Resampling by linear interpolation, DO NOT keep original node.

        Args:
            xyzr: The array of shape (N, 4).

        Returns:
            coordinates: An array of shape (n_nodes, 4).
        """

        xp = np.cumsum(np.linalg.norm(xyzr[1:, :3] - xyzr[:-1, :3], axis=1))
        xp = np.insert(xp, 0, 0)
        xvals = np.linspace(0, xp[-1], self.n_nodes)

        x = np.interp(xvals, xp, xyzr[:, 0])
        y = np.interp(xvals, xp, xyzr[:, 1])
        z = np.interp(xvals, xp, xyzr[:, 2])
        r = np.interp(xvals, xp, xyzr[:, 3])
        return cast(npt.NDArray[np.float32], np.stack([x, y, z, r], axis=1))

    @override
    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}"


class BranchIsometricResampler(_BranchResampler):
    def __init__(self, distance: float, *, adjust_last_gap: bool = True) -> None:
        super().__init__()
        self.distance = distance
        self.adjust_last_gap = adjust_last_gap

    @override
    def resample(self, xyzr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Resampling by isometric interpolation, DO NOT keep original node.

        Args:
            xyzr: The array of shape (N, 4).

        Returns:
            new_xyzr: An array of shape (n_nodes, 4).
        """

        # Compute the cumulative distances between consecutive points
        diffs = np.diff(xyzr[:, :3], axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        total_length = cumulative_distances[-1]
        n_nodes = int(np.ceil(total_length / self.distance)) + 1

        # Determine the new distances
        if self.adjust_last_gap and n_nodes > 1:
            new_distances = np.linspace(0, total_length, n_nodes)
        else:
            new_distances = np.arange(0, total_length, self.distance)
            # keep endpoint
            new_distances = np.concatenate([new_distances, total_length])

        # Interpolate the new points
        new_xyzr = np.zeros((n_nodes, 4), dtype=np.float32)
        new_xyzr[:, :3] = np.array(
            [
                np.interp(new_distances, cumulative_distances, xyzr[:, i])
                for i in range(3)
            ]
        ).T
        new_xyzr[:, 3] = np.interp(new_distances, cumulative_distances, xyzr[:, 3])
        return new_xyzr

    @override
    def extra_repr(self) -> str:
        return f"distance={self.distance},adjust_last_gap={self.adjust_last_gap}"


class BranchConvSmoother(Transform[Branch, Branch[DictSWC]]):
    r"""Smooth the branch by sliding window."""

    def __init__(self, n_nodes: int = 5) -> None:
        """
        Args:
            n_nodes: Window size.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.kernel = np.ones(n_nodes)

    @override
    def __call__(self, x: Branch) -> Branch[DictSWC]:
        x = x.detach()
        c = signal.convolve(np.ones(x.number_of_nodes()), self.kernel, mode="same")
        for k in ["x", "y", "z"]:
            v = x.get_ndata(k)
            s = signal.convolve(v, self.kernel, mode="same")
            x.attach.ndata[k][1:-1] = (s / c)[1:-1]

        return x

    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}"


class BranchStandardizer(Transform[Branch, Branch[DictSWC]]):
    r"""Standardize branch.

    Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y, and scale max
    radius to 1.
    """

    @override
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
        r"""Get standardize transformation matrix.

        Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y.

        Args:
            xyz: The `x`, `y`, `z` matrix of shape (N, 3) of branch.

        Returns:
            T: An homogeneous transformation matrix of shape (4, 4).
        """

        assert xyz.ndim == 2 and xyz.shape[1] == 3, (
            f"xyz should be of shape (N, 3), got {xyz.shape}"
        )

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
