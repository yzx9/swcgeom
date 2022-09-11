"""Branch is a set of node points."""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from typing_extensions import Self  # TODO: move to typing in python 3.11

from ..utils import padding1d
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
        ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, typee, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
        }
        kwargs.update(ndata)
        self.ndata = kwargs

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

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()

    def straight_line_distance(self) -> float:
        """Distance between start point and end point."""
        return np.linalg.norm(self[-1].xyz() - self[0].xyz()).item()

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
            branch = cls(
                xyzr.shape[0],
                x=xyzr[:, 0],
                y=xyzr[:, 1],
                z=xyzr[:, 2],
                r=xyzr[:, 3],
            )
            branches.append(branch)

        return branches
