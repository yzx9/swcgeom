from typing import Iterable, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from typing_extensions import Self  # TODO: move to typing in python 3.11

from ..utils import painter, transforms
from .Tree import Tree

_Scale = tuple[float, float, float, float]


class Branch(list[Tree.Node]):
    """A branch of neuron tree.

    Notes
    =====
    Only a part of data of branch nodes is valid, such as `x`, `y`, `z` and
    `r`, but the `id` and `type` is usually invalid.
    """

    scale: _Scale

    def __init__(self, nodes: Iterable[Tree.Node]) -> None:
        super().__init__(nodes)
        self.scale = (1, 1, 1, 1)

    def x(self) -> npt.NDArray[np.float64]:
        """Get the `x` of branch, shape of (n_sample,)"""
        return np.array([n.x for n in self], dtype=np.float64)

    def y(self) -> npt.NDArray[np.float64]:
        """Get the `y` of branch, shape of (n_sample,)"""
        return np.array([n.y for n in self], dtype=np.float64)

    def z(self) -> npt.NDArray[np.float64]:
        """Get the `z` of branch, shape of (n_sample,)"""
        return np.array([n.z for n in self], dtype=np.float64)

    def r(self) -> npt.NDArray[np.float64]:
        """Get the `r` of branch, shape of (n_sample,)"""
        return np.array([n.r for n in self], dtype=np.float64)

    def xyz(self) -> npt.NDArray[np.float64]:
        """Get the `x`, `y`, `z` of branch, shape of (n_sample, 3)"""
        return np.array([n.xyz() for n in self], dtype=np.float64)

    def xyzr(self) -> npt.NDArray[np.float64]:
        """Get the `x`, `y`, `z`, `r` of branch, shape of (n_sample, 4)"""
        return np.array([n.xyzr() for n in self], dtype=np.float64)

    def draw(
        self,
        color: str = painter.palette.mizugaki,
        ax: plt.Axes | None = None,
        standardize: bool = True,
        **kwargs,
    ) -> tuple[plt.Axes, LineCollection]:
        """Draw neuron branch.

        Parameters
        ==========
        color : str, optional.
            Color of branch. If `None`, the default color will be enabled.
        ax : ~matplotlib.axes.Axes, optional.
            A subplot. If `None`, a new one will be created.
        standardize : bool, default to `True`.
            Standardize branch, see also self.standardize.
        **kwargs : dict[str, Unknown].
            Forwarded to `matplotlib.collections.LineCollection`.

        Returns
        =======
        ax : ~matplotlib.axes.Axes.
            If provided, return as-is.
        collection : ~matplotlib.collections.LineCollection.
            Drawn line collection.
        """

        if standardize:
            xyzr, _ = self._standardize()
            xyz = xyzr[:, :3]
        else:
            xyz = self.xyz()

        lines = np.array([xyz[:-1], xyz[1:]]).swapaxes(0, 1)
        return painter.draw_lines(lines, color=color, ax=ax, **kwargs)

    # fmt:off
    @overload
    def resample(self, *, num: int) -> Self: pass
    @overload
    def resample(self, mode: Literal["linear"], *, num: int) -> Self: pass
    # fmt:on

    def resample(self, mode: str = "linear", **kwargs) -> Self:
        """Resample branch to special num of nodes.

        Parameters
        ==========
        mode : str, default to `linear`.
            Resample mode.

        Modes
        =====

        Linear
        ------
        Resampling by linear interpolation, DO NOT keep original node.

        num : int.
            Number of nodes after resample.

        See Also
        ========
        cls.resample_linear
        """

        xyzr = self.xyzr()
        if mode == "linear":
            new_xyzr = self.resample_linear(xyzr, **kwargs)
        else:
            raise Exception(f"unsupported reample mode '{mode}'")

        return Branch.from_numpy(new_xyzr)

    def standardize(self) -> Self:
        """Standarize a branch.

        Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y,
        and scale max radius to 1.
        """

        new_xyzr, scale = self._standardize()
        branch = Branch.from_numpy(new_xyzr)
        branch.scale = scale
        return branch

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1))

    def straight_line_distance(self) -> float:
        """Distance between start point and end point."""
        return float(np.linalg.norm(self[-1].xyz() - self[0].xyz()))

    def _standardize(self) -> tuple[npt.NDArray[np.float64], _Scale]:
        xyzr = self.xyzr()
        xyz, r = xyzr[:, 0:3], xyzr[:, 3:4]
        T = np.identity(4)
        v = np.concatenate([self[-1].xyz() - self[0].xyz(), np.zeros((1))])[:, None]

        # translate to the origin
        translate = transforms.translate3d(-self[0].x, -self[0].y, -self[0].z)
        T = np.dot(translate, T)

        # scale to unit vector
        s = 1 / self.straight_line_distance()
        scale = transforms.scale3d(s, s, s)
        T = np.dot(scale, T)

        # rotate v to the xz-plane, v should be (x, 0, z) now
        vy = np.dot(T, v)[:, 0]
        # when looking at the xz-plane along the positive y-axis, the
        # coordinates should be (z, x)
        rotate_y = transforms.rotate3d_y(transforms.angle([vy[2], vy[0]], [0, 1]))
        T = np.dot(rotate_y, T)

        # rotate v to the x-axis, v should be (1, 0, 0) now
        vx = np.dot(T, v)[:, 0]
        rotate_z = transforms.rotate3d_z(transforms.angle([vx[0], vx[1]], [1, 0]))
        T = np.dot(rotate_z, T)

        # rotate the farthest point to the xy-plane
        ones = np.ones([xyz.shape[0], 1])
        xyz4 = np.concatenate([xyz, ones], axis=1).transpose()  # (4, N)
        if xyz.shape[0] > 2:
            new_xyz4 = np.dot(T, xyz4)  # (4, N)
            max_index = np.argmax(np.linalg.norm(new_xyz4[1:3, :], axis=0)[1:-1]) + 1
            max_xyz4 = xyz4[:, max_index].reshape(4, 1)
            max_xyz4_t = np.dot(T, max_xyz4)  # (4, 1)
            angle_x = transforms.angle(max_xyz4_t[1:3, 0], [1, 0])
            rotate_x = transforms.rotate3d_x(angle_x)
            T = np.dot(rotate_x, T)

        # scale max radius to 1
        scale_r = 1 / r.max()

        # generate new branch
        new_xyz = np.dot(T, xyz4)[0:3, :].transpose()
        new_r = r * scale_r
        new_xyzr = np.concatenate([new_xyz, new_r], axis=1)
        sx, sy, sz, sr = self.scale
        return new_xyzr, (sx * s, sy * s, sz * s, sr * scale_r)

    def __str__(self) -> str:
        return f"Neuron branch with {len(self)} nodes."

    @classmethod
    def from_numpy(cls, xyzr: npt.NDArray[np.float64]) -> Self:
        """Create a branch from ~numpy.ndarray.

        Parameters
        ==========
        xyzr : npt.NDArray[np.float64].
            Collection of nodes. If shape of (n, 4), both `x`, `y`, `z`, `r`
            of nodes is enabled. If shape of (n, 3), only `x`, `y`, `z` is
            enabled and `r` will fill by 1.
        """
        assert xyzr.ndim == 2
        assert xyzr.shape[1] >= 3

        if xyzr.shape[1] == 3:
            ones = np.ones([xyzr.shape[0], 1])
            xyzr = np.concatenate([xyzr, ones], axis=1)

        nodes = [
            Tree.Node(i, 0, x, y, z, r, i - 1)
            for i, (x, y, z, r) in zip(range(xyzr.shape[0]), xyzr[:, 0:4])
        ]
        return cls(nodes)

    @classmethod
    def from_numpy_batch(cls, xyzr_batch: npt.NDArray[np.float64]) -> list[Self]:
        """Create list of branch form ~numpy.ndarray.

        Parameters
        ==========
        xyzr: npt.NDArray[np.float64].
            Batch of collection of nodes. If shape of (bs, n, 4), both `x`,
            `y`, `z`, `r` of nodes is enabled. If shape of (bs, n, 3), only
            `x`, `y`, `z` is enabled and `r` will fill by 1.
        """

        assert xyzr_batch.ndim == 3
        assert xyzr_batch.shape[1] >= 3

        if xyzr_batch.shape[2] == 3:
            ones = np.ones([xyzr_batch.shape[0], xyzr_batch.shape[1], 1])
            xyzr_batch = np.concatenate([xyzr_batch, ones], axis=2)

        branches = list[Branch]()
        for xyzr in xyzr_batch:
            nodes = [
                Tree.Node(i, 0, x, y, z, r, i - 1)
                for i, (x, y, z, r) in zip(range(xyzr.shape[0]), xyzr[:, 0:4])
            ]
            branches.append(Branch(nodes))

        return branches

    @staticmethod
    def resample_linear(
        xyzr: npt.NDArray[np.float64], num: int
    ) -> npt.NDArray[np.float64]:
        """Resampling by linear interpolation, DO NOT keep original node.

        Parameters
        ==========
        num : int.
            Number of nodes after resample.
        """

        xp = np.cumsum(np.linalg.norm(xyzr[1:, :3] - xyzr[:-1, :3], axis=1))
        xp = np.insert(xp, 0, 0)
        xvals = np.linspace(0, xp[-1], num)

        x = np.interp(xvals, xp, xyzr[:, 0])
        y = np.interp(xvals, xp, xyzr[:, 1])
        z = np.interp(xvals, xp, xyzr[:, 2])
        r = np.interp(xvals, xp, xyzr[:, 3])
        return np.array([x, y, z, r])
