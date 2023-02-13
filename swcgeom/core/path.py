"""Nueron node."""

from typing import Generic, Iterable, Iterator, List, overload

import numpy as np
import numpy.typing as npt

from .node import Node
from .swc import DictSWC, SWCLike, SWCTypeVar

__all__ = ["Path"]


class Path(SWCLike, Generic[SWCTypeVar]):
    """Neural path.

    A path is a linear set of points without bifurcations.
    """

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    class Node(Node["Path"]):
        """Node of neuron tree."""

    def __init__(self, attach: SWCTypeVar, idx: npt.ArrayLike) -> None:
        super().__init__()
        self.attach = attach
        self.idx = np.array(idx, dtype=np.int32)

    def __iter__(self) -> Iterator[Node]:
        return (self[i] for i in range(len(self)))

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
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def get_node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def detach(self) -> "Path":
        """Detach from current attached object."""
        attact = DictSWC(**{k: self[k] for k in self.keys()})
        return Path(attact, self.idx.copy())

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
        if (length := self.length()) == 0:
            return 1
        return self.straight_line_distance() / length
