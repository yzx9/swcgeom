"""Branch is a set of node points."""

from typing import Any, Literal, NamedTuple, cast, overload

import matplotlib.axes
import matplotlib.collections
import numpy as np
import numpy.typing as npt
from typing_extensions import Self  # TODO: move to typing in python 3.11

from ..utils import padding1d, painter, transforms
from ._node import NodeAttached

Scale = NamedTuple("Scale", x=float, y=float, z=float, r=float)


class Branch:
    """A branch of neuron tree.

    Notes
    -----
    Only a part of data of branch nodes is valid, such as `x`, `y`, `z` and
    `r`, but the `id` and `type` is usually invalid.
    """

    class Node(NodeAttached["Branch"]):
        """Node of branch."""

    ndata: dict[str, npt.NDArray[Any]]

    def __init__(
        self,
        n_nodes: int,
        *,
        typee: npt.NDArray[np.int32] | None = None,
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        self.ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, typee, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
            **kwargs,
        }

    def __len__(self) -> int:
        """Get the number of nodes."""
        return self.x().shape[0]

    def __repr__(self) -> str:
        return f"Neuron branch with {len(self)} nodes."

    def __getitem__(self, idx: int) -> Node:
        return self.Node(self, idx)

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x-coordinates of nodes of branch.

        Returns
        -------
        x : np.ndarray[np.float32]
            An array of shape (n_sample,).
        """
        return self.ndata["x"]

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y-coordinates of nodes of branch.

        Returns
        -------
        y : np.ndarray[np.float32]
            An array of shape (n_sample,).
        """
        return self.ndata["y"]

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z-coordinates of nodes of branch.

        Returns
        -------
        z : np.ndarray[np.float32]
            An array of shape (n_sample,).
        """
        return self.ndata["z"]

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of nodes of branch.

        Returns
        -------
        r : np.ndarray[np.float32]
            An array of shape (n_sample,).
        """
        return self.ndata["r"]

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z` of branch.

        Returns
        -------
        xyz : np.ndarray[np.float32]
            An array of shape (n_sample, 3).
        """
        return np.array([self.x(), self.y(), self.z()])

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z`, `r` of branch.

        Returns
        -------
        xyzr : np.ndarray[np.float32]
            An array of shape (n_sample, 4).
        """
        return np.array([self.x(), self.y(), self.z(), self.r()])

    def draw(
        self,
        color: str = painter.palette.mizugaki,
        ax: matplotlib.axes.Axes | None = None,
        standardize: bool = True,
        **kwargs,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.collections.LineCollection]:
        """Draw neuron branch.

        Parameters
        ----------
        color : str, optional
            Color of branch. If `None`, the default color will be enabled.
        ax : ~matplotlib.axes.Axes, optional
            A subplot. If `None`, a new one will be created.
        standardize : bool, default `True`
            Standardize branch, see also self.standardize.
        **kwargs : dict[str, Any]
            Forwarded to `matplotlib.collections.LineCollection`.

        Returns
        -------
        ax : ~matplotlib.axes.Axes
            If provided, return as-is.
        collection : ~matplotlib.collections.LineCollection
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
    def resample(self, *, num: int) -> Self: ...
    @overload
    def resample(self, mode: Literal["linear"], *, num: int) -> Self: ...
    # fmt:on

    def resample(self, mode: str = "linear", **kwargs) -> Self:
        """Resample branch to special num of nodes.

        Parameters
        ----------
        mode : str, default `linear`
            Resample mode.

            Linear mode :
                Resampling by linear interpolation, DO NOT keep original node.

                num : int
                    Number of nodes after resample.

        See Also
        --------
        cls.resample_linear : linear mode.
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

        new_xyzr, _ = self._standardize()
        return Branch.from_numpy(new_xyzr)

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()

    def straight_line_distance(self) -> float:
        """Distance between start point and end point."""
        return np.linalg.norm(self[-1].xyz() - self[0].xyz()).item()

    def _standardize(self) -> tuple[npt.NDArray[np.float32], Scale]:
        xyzr = self.xyzr()
        xyz, r = xyzr[:, 0:3], xyzr[:, 3:4]
        T, s = self.get_standardize_matrix(xyzr)

        ones = np.ones([xyz.shape[0], 1])
        xyz4 = np.concatenate([xyz, ones], axis=1).transpose()  # (4, N)
        new_xyz = np.dot(T, xyz4)[0:3, :].transpose()
        new_xyzr = np.concatenate([new_xyz, r * s.r], axis=1)

        return new_xyzr, s

    @classmethod
    def from_numpy(cls, xyzr: npt.NDArray[np.float32]) -> Self:
        """Create a branch from ~numpy.ndarray.

        Parameters
        ----------
        xyzr : npt.NDArray[np.float32]
            Collection of nodes. If shape (n, 4), both `x`, `y`, `z`, `r` of
            nodes is enabled. If shape (n, 3), only `x`, `y`, `z` is enabled
            and `r` will fill by 1.
        """
        assert xyzr.ndim == 2
        assert xyzr.shape[1] >= 3

        if xyzr.shape[1] == 3:
            ones = np.ones([xyzr.shape[0], 1], dtype=np.float32)
            xyzr = np.concatenate([xyzr, ones], axis=1)

        return cls(
            xyzr.shape[0], x=xyzr[:, 0], y=xyzr[:, 1], z=xyzr[:, 2], r=xyzr[:, 3]
        )

    @classmethod
    def from_numpy_batch(cls, xyzr_batch: npt.NDArray[np.float32]) -> list[Self]:
        """Create list of branch form ~numpy.ndarray.

        Parameters
        ----------
        xyzr : npt.NDArray[np.float32]
            Batch of collection of nodes. If shape (bs, n, 4), both `x`, `y`,
            `z`, `r` of nodes is enabled. If shape (bs, n, 3), only `x`, `y`,
            `z` is enabled and `r` will fill by 1.
        """

        assert xyzr_batch.ndim == 3
        assert xyzr_batch.shape[1] >= 3

        if xyzr_batch.shape[2] == 3:
            ones = np.ones(
                [xyzr_batch.shape[0], xyzr_batch.shape[1], 1], dtype=np.float32
            )
            xyzr_batch = np.concatenate([xyzr_batch, ones], axis=2)

        branches = list[Branch]()
        for xyzr in xyzr_batch:
            branches.append(
                cls(
                    xyzr.shape[0],
                    x=xyzr[:, 0],
                    y=xyzr[:, 1],
                    z=xyzr[:, 2],
                    r=xyzr[:, 3],
                )
            )

        return branches

    @staticmethod
    def resample_linear(
        xyzr: npt.NDArray[np.float32], num: int
    ) -> npt.NDArray[np.float32]:
        """Resampling by linear interpolation, DO NOT keep original node.

        Parameters
        ----------
        xyzr : np.ndarray[np.float32]
            The array of shape (N, 4).
        num : int
            Number of nodes after resample.

        Returns
        -------
        coords : ~numpy.NDArray[float64]
            An array of shape (num, 4).
        """

        xp = np.cumsum(np.linalg.norm(xyzr[1:, :3] - xyzr[:-1, :3], axis=1))
        xp = np.insert(xp, 0, 0)
        xvals = np.linspace(0, xp[-1], num)

        x = np.interp(xvals, xp, xyzr[:, 0])
        y = np.interp(xvals, xp, xyzr[:, 1])
        z = np.interp(xvals, xp, xyzr[:, 2])
        r = np.interp(xvals, xp, xyzr[:, 3])
        return cast(npt.NDArray[np.float32], np.stack([x, y, z, r], axis=1))

    @staticmethod
    def get_standardize_matrix(
        xyzr: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], Scale]:
        """Get standarize transformation matrix.

        Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y,
        and scale max radius to 1.

        Parameters
        ----------
        xyzr : np.ndarray[np.float32]
            The `x`, `y`, `z`, `r` matrix of shape (N, 4) of branch.

        Returns
        -------
        T : np.ndarray[np.float32]
            An homogeneous transfomation matrix of shape (4, 4).
        scale : float
            Scale ratio.
        """
        assert xyzr.ndim == 2
        assert xyzr.shape[1] == 4

        xyz, r = xyzr[:, :3], xyzr[:, 3:]
        T = np.identity(4)
        v = np.concatenate([xyz[-1] - xyz[0], np.zeros((1))])[:, None]

        # translate to the origin
        translate = transforms.translate3d(-xyz[0, 0], -xyz[0, 1], -xyz[0, 2])
        T = np.dot(translate, T)

        # scale to unit vector
        s = float(1 / np.linalg.norm(v[:3, 0]))
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
        if xyz.shape[0] > 2:
            ones = np.ones([xyz.shape[0], 1])
            xyz4 = np.concatenate([xyz, ones], axis=1).transpose()  # (4, N)
            new_xyz4 = np.dot(T, xyz4)  # (4, N)
            max_index = np.argmax(np.linalg.norm(new_xyz4[1:3, :], axis=0)[1:-1]) + 1
            max_xyz4 = xyz4[:, max_index].reshape(4, 1)
            max_xyz4_t = np.dot(T, max_xyz4)  # (4, 1)
            angle_x = transforms.angle(max_xyz4_t[1:3, 0], [1, 0])
            rotate_x = transforms.rotate3d_x(angle_x)
            T = np.dot(rotate_x, T)

        # scale max radius to 1
        sr = 1 / r.max()

        return T, Scale(s, s, s, sr)
