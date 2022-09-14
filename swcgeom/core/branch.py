"""Branch is a set of node points."""

from typing import Any, Dict, Generic, Iterable, List, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self  # TODO: move to typing in python 3.11

from ..utils import padding1d
from .swc import SWC, SWCTypeVar
from .node import NodeAttached

__all__ = ["Branch", "BranchAttached"]


class _Branch(SWC):
    class Node(NodeAttached["_Branch"]):
        """Node of branch."""

    def __iter__(self) -> Iterable[Node]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        return self.id().shape[0]

    def __repr__(self) -> str:
        return f"Neuron branch with {len(self)} nodes."

    # fmt:off
    @overload
    def __getitem__(self, key: int) -> Node: ...
    @overload
    def __getitem__(self, key: slice) -> List[Node]: ...
    @overload
    def __getitem__(self, key: str) -> npt.NDArray[Any]: ...
    # fmt:on
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.get_node(i) for i in range(*key.indices(len(self)))]

        if isinstance(key, int):
            length = len(self)
            if key < -length or key >= length:
                raise IndexError(f"The index ({key}) is out of range.")

            if key < 0:  # Handle negative indices
                key += length

            return self.get_node(key)

        if isinstance(key, str):
            return self.get_ndata(key)

        raise TypeError("Invalid argument type.")

    def get_keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def get_node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()

    def straight_line_distance(self) -> float:
        """Distance between start point and end point."""
        return np.linalg.norm(self[-1].xyz() - self[0].xyz()).item()


class Branch(_Branch):
    r"""A branch of neuron tree.

    Notes
    -----
    Only a part of data of branch nodes is valid, such as `x`, `y`, `z` and
    `r`, but the `id` and `pid` is usually invalid.
    """

    ndata: Dict[str, npt.NDArray]

    def __init__(
        self,
        n_nodes: int,
        *,
        type: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        super().__init__()
        ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, type, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
        }
        kwargs.update(ndata)
        self.ndata = kwargs
        self.source = None  # TODO

    def get_keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.ndata[key]

    @classmethod
    def from_xyzr(cls, xyzr: npt.NDArray[np.float32]) -> Self:
        """Create a branch from ~numpy.ndarray.

        Parameters
        ----------
        xyzr : npt.NDArray[np.float32]
            Collection of nodes. If shape (n, 4), both `x`, `y`, `z`, `r` of
            nodes is enabled. If shape (n, 3), only `x`, `y`, `z` is enabled
            and `r` will fill by 1.
        """
        assert xyzr.ndim == 2 and xyzr.shape[1] in (
            3,
            4,
        ), f"xyzr should be of shape (N, 3) or (N, 4), got {xyzr.shape}"

        if xyzr.shape[1] == 3:
            ones = np.ones([xyzr.shape[0], 1], dtype=np.float32)
            xyzr = np.concatenate([xyzr, ones], axis=1)

        return cls(
            xyzr.shape[0], x=xyzr[:, 0], y=xyzr[:, 1], z=xyzr[:, 2], r=xyzr[:, 3]
        )

    @classmethod
    def from_xyzr_batch(cls, xyzr_batch: npt.NDArray[np.float32]) -> list[Self]:
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


class BranchAttached(_Branch, Generic[SWCTypeVar]):
    """Node attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, idx: npt.NDArray[np.int32]) -> None:
        super().__init__()
        self.attach = attach
        self.idx = idx

    def get_keys(self) -> Iterable[str]:
        return self.attach.get_keys()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.attach.get_ndata(key)[self.idx]

    # def detach(self) -> Branch: # TODO
