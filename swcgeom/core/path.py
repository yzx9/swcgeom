"""Nueron node."""

from typing import Dict, Generic, Iterable, List, overload

import numpy as np
import numpy.typing as npt

from ..utils import padding1d
from .node import NodeAttached
from .swc import SWCLike, SWCTypeVar

__all__ = ["PathBase", "Path", "PathAttached"]


class PathBase(SWCLike):
    """Path of neuron tree.

    A path is a linear set of points without bifurcations.
    """

    class Node(NodeAttached["PathBase"]):
        """Node of neuron tree."""

    def __iter__(self) -> Iterable[Node]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        return self.id().shape[0]

    def __repr__(self) -> str:
        return f"Neuron path with {len(self)} nodes."

    @overload
    def __getitem__(self, key: int) -> Node:
        ...

    @overload
    def __getitem__(self, key: slice) -> List[Node]:
        ...

    @overload
    def __getitem__(self, key: str) -> npt.NDArray:
        ...

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

    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray:
        raise NotImplementedError()

    def get_node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample,).

        Returns a consecutively incremented id.

        See Also
        --------
        self.origin_id
        """
        return np.arange(len(self.origin_id()), dtype=np.int32)

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of shape (n_sample,).

        Returns a consecutively incremented pid.

        See Also
        --------
        self.origin_pid
        """
        return np.arange(-1, len(self.origin_id()) - 1, dtype=np.int32)

    def origin_id(self) -> npt.NDArray[np.int32]:
        """Get the original id."""
        return self.get_ndata("id")

    def origin_pid(self) -> npt.NDArray[np.int32]:
        """Get the original pid."""
        return self.get_ndata("pid")

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()

    def straight_line_distance(self) -> float:
        """Straight-line distance of path.

        The end-to-end straight-line distance between start point and
        end point.
        """
        return np.linalg.norm(self[-1].xyz() - self[0].xyz()).item()

    def tortuosity(self) -> float:
        """Tortuosity of path.

        The straight-line distance between two consecutive branch
        points divided by the length of the neuronal path between
        those points.
        """
        return self.straight_line_distance() / self.length()


class Path(PathBase):
    r"""A path of neuron tree."""

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
        self.source = ""  # TODO

    def keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.ndata[key]


class PathAttached(PathBase, Generic[SWCTypeVar]):
    """Path attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, idx: npt.ArrayLike) -> None:
        super().__init__()
        self.attach = attach
        self.idx = np.array(idx, dtype=np.int32)

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> Path:
        return Path(len(self), **{k: self[k] for k in self.keys()})
