"""Nueron path."""

import warnings
from typing import Generic, Iterable, Iterator, List, overload

import numpy as np
import numpy.typing as npt

from swcgeom.core.node import Node
from swcgeom.core.swc import DictSWC, SWCLike, SWCTypeVar

__all__ = ["Path"]


class Path(SWCLike, Generic[SWCTypeVar]):
    """Neuron path.

    A path is a linear set of points without bifurcations.
    """

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    class Node(Node["Path"]):
        """Node of neuron tree."""

    def __init__(self, attach: SWCTypeVar, idx: npt.ArrayLike) -> None:
        super().__init__()
        self.attach = attach
        self.names = attach.names
        self.idx = np.array(idx, dtype=np.int32)
        self.source = self.attach.source

    def __iter__(self) -> Iterator[Node]:
        return (self.node(i) for i in range(len(self)))

    def __len__(self) -> int:
        return self.id().shape[0]

    def __repr__(self) -> str:
        return f"Neuron path with {len(self)} nodes."

    # fmt:off
    @overload
    def __getitem__(self, key: int) -> Node: ...
    @overload
    def __getitem__(self, key: slice) -> List[Node]: ...
    @overload
    def __getitem__(self, key: str) -> npt.NDArray: ...
    # fmt:on

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.node(i) for i in range(*key.indices(len(self)))]

        if isinstance(key, (int, np.integer)):
            length = len(self)
            if key < -length or key >= length:
                raise IndexError(f"The index ({key}) is out of range.")

            if key < 0:  # Handle negative indices
                key += length

            return self.node(key)

        if isinstance(key, str):
            return self.get_ndata(key)

        raise TypeError("Invalid argument type.")

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def get_node(self, idx: int | np.integer) -> Node:
        """Get the count of intersection.

        .. deprecated:: 0.16.0
            Use :meth:`path.node` instead.
        """

        warnings.warn(
            "`Path.get_node` has been deprecated since v0.16.0 and "
            "will be removed in future version",
            DeprecationWarning,
        )
        return self.node(idx)

    def node(self, idx: int | np.integer) -> Node:
        return self.Node(self, idx)

    def detach(self) -> "Path[DictSWC]":
        """Detach from current attached object."""
        # pylint: disable-next=consider-using-dict-items
        attact = DictSWC(
            **{k: self.get_ndata(k) for k in self.keys()},
            source=self.source,
            names=self.names,
        )
        attact.ndata[self.names.id] = self.id()
        attact.ndata[self.names.pid] = self.pid()
        return Path(attact, self.id())

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
        return self.get_ndata(self.names.id)

    def origin_pid(self) -> npt.NDArray[np.int32]:
        """Get the original pid."""
        return self.get_ndata(self.names.pid)

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()

    def straight_line_distance(self) -> float:
        """Straight-line distance of path.

        The end-to-end straight-line distance between start point and
        end point.
        """
        return np.linalg.norm(self.node(-1).xyz() - self.node(0).xyz()).item()

    def tortuosity(self) -> float:
        """Tortuosity of path.

        The straight-line distance between two consecutive branch
        points divided by the length of the neuronal path between
        those points.
        """
        if (length := self.length()) == 0:
            return 1
        return self.straight_line_distance() / length
