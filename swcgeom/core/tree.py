"""Neuron tree."""

from typing import Any, Callable, Iterable, List, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt

from ..utils import padding1d
from .node import NodeAttached
from .branch import BranchAttached
from .swc import SWC

__all__ = ["Tree"]

T, K = TypeVar("T"), TypeVar("K")


class Tree(SWC):
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(NodeAttached["Tree"]):
        """Node of neuron tree."""

    class Branch(BranchAttached["Branch"]):
        """Branch of neuron tree."""

    ndata: dict[str, npt.NDArray[Any]]
    source: str | None

    def __init__(
        self,
        n_nodes: int,
        *,
        type: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        pid: npt.NDArray[np.int32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        if pid is None:
            pid = np.arange(-1, n_nodes - 1, step=1, dtype=np.int32)

        ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, type, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": padding1d(n_nodes, pid, dtype=np.int32),
        }
        kwargs.update(ndata)
        self.ndata = kwargs
        self.source = None

    def __len__(self) -> int:
        return self.number_of_nodes()

    def __repr__(self) -> str:
        n_nodes, n_edges = self.number_of_nodes(), self.number_of_edges()
        return f"Neuron Tree with {n_nodes} nodes and {n_edges} edges"

    # fmt:off
    @overload
    def __getitem__(self, key: slice) -> List[Node]: ...
    @overload
    def __getitem__(self, key: int) -> Node: ...
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
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.ndata[key]

    def get_node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def number_of_nodes(self) -> int:
        """Get the number of nodes."""
        return self.id().shape[0]

    def number_of_edges(self) -> int:
        """Get the number of edges."""
        return self.number_of_nodes() - 1

    TraverseEnter = Callable[[Node, T | None], T]
    TraverseLeave = Callable[[Node, list[T]], T]

    # fmt:off
    @overload
    def traverse(self, *, enter: TraverseEnter[T]) -> None: ...
    @overload
    def traverse(self, *, enter: TraverseEnter[T] | None = None, leave: TraverseLeave[K]) -> K: ...
    # fmt:on

    def traverse(
        self,
        *,
        enter: TraverseEnter[T] | None = None,
        leave: TraverseLeave[K] | None = None,
    ) -> K | None:
        """Traverse each nodes.

        Parameters
        ----------
        enter : Callable[[Node, list[T]], T], optional
            The callback when entering each node, it accepts two parameters,
            the first parameter is the current node, the second parameter is
            the parent's information T, and the root node receives an None.
        leave : Callable[[Node, T | None], T], optional
            The callback when leaving each node. When leaving a node, subtree
            has already been traversed. Callback accepts two parameters, the
            first parameter is the current node, the second parameter is the
            children's information T, and the leaf node receives an empty list.
        """

        children_map = dict[int, list[int]]()
        for idx, pid in enumerate(self.pid()):
            children_map.setdefault(pid, [])
            children_map[pid].append(idx)

        def dfs(
            idx: int,
            enter: Tree.TraverseEnter[T] | None,
            leave: Tree.TraverseLeave[K] | None,
            pre: T | None,
        ) -> K | None:
            cur = enter(self[idx], pre) if enter is not None else None
            children = [dfs(i, enter, leave, cur) for i in children_map.get(idx, [])]
            children = cast(list[K], children)
            return leave(self[idx], children) if leave is not None else None

        return dfs(0, enter, leave, None)

    def copy(self) -> "Tree":
        """Make a copy."""
        new_tree = Tree(len(self), **{k: v.copy() for k, v in self.ndata.items()})
        new_tree.source = self.source
        return new_tree
