"""Neuron tree."""

import os
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils import padding1d
from .branch import BranchAttached
from .node import NodeAttached
from .segment import SegmentAttached
from .swc import SWCLike

__all__ = ["Tree"]

T, K = TypeVar("T"), TypeVar("K")


SWCNameMap = TypedDict(
    "SWCNameMap",
    {"id": str, "type": str, "x": str, "y": str, "z": str, "r": str, "pid": str},
    total=False,
)


class Tree(SWCLike):
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(NodeAttached["Tree"]):
        """Node of neuron tree."""

        def parent(self) -> Union["Tree.Node", None]:
            return Tree.Node(self.attach, self.pid) if self.pid != -1 else None

        def children(self) -> List["Tree.Node"]:
            pid = self.attach.pid()
            return [Tree.Node(self.attach, idx) for idx in pid[pid == self.id]]

        def is_soma(self) -> bool:
            return self.id == -1

        def is_bifurcation(self) -> bool:
            return len(self.children()) > 1

        def is_tip(self) -> bool:
            return len(self.children()) == 0

        def get_branch(self) -> "Tree.Branch":
            nodes: List["Tree.Node"] = [self]
            while (
                not nodes[-1].is_bifurcation()
                and (parent := nodes[-1].parent()) is not None
            ):
                nodes.append(parent)

            nodes.reverse()
            while not nodes[-1].is_bifurcation() and not nodes[-1].is_tip():
                nodes.append(nodes[-1].children()[0])

            return Tree.Branch(self.attach, [n.id for n in nodes])

    class Segment(SegmentAttached["Tree"]):
        """Segment of neuron tree."""

    class Branch(BranchAttached["Tree"]):
        """Branch of neuron tree."""

    ndata: dict[str, npt.NDArray]

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

    def __iter__(self) -> Iterable[Node]:
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        n_nodes, n_edges = self.number_of_nodes(), self.number_of_edges()
        return f"Neuron Tree with {n_nodes} nodes and {n_edges} edges"

    # fmt:off
    @overload
    def __getitem__(self, key: slice) -> List[Node]: ...
    @overload
    def __getitem__(self, key: int) -> Node: ...
    @overload
    def __getitem__(self, key: str) -> npt.NDArray: ...
    # fmt:on
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.node(i) for i in range(*key.indices(len(self)))]

        if isinstance(key, int):
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
        return self.ndata.keys()

    def node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.ndata[key]

    def get_segments(self) -> List[Segment]:
        # pylint: disable-next=not-an-iterable
        return [self.Segment(self, n.pid, n.id) for n in self[1:]]

    def get_branches(self) -> List[Branch]:
        Info = Tuple[List[Tree.Branch], List[int]]

        def collect_branches(node: "Tree.Node", pre: List[Info]) -> Info:
            if len(pre) == 1:
                branches, child = pre[0]
                child.append(node.id)
                return branches, child

            branches: List[Tree.Branch] = []

            for sub_branches, child in pre:
                child.append(node.id)
                child.reverse()
                sub_branches.append(Tree.Branch(self, np.array(child, dtype=np.int32)))
                sub_branches.reverse()
                branches.extend(sub_branches)

            return branches, [node.id]

        # pylint: disable-next=unpacking-non-sequence
        branches, _ = self.traverse(leave=collect_branches)
        return branches

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

    @staticmethod
    def from_swc(swc_file: str, name_map: SWCNameMap | None = None) -> "Tree":
        """Read neuron tree from swc file.

        Parameters
        ----------
        swc_file : str
            Path of swc file, the id should be consecutively incremented.
        name_map : dict[str, str], optional
            Map standard name to actual name. The standard names are `id`,
            `type`, `x`, `y`, `z`, `r` and `pid`.
        """

        cols: List[Tuple[str, npt.DTypeLike]] = [
            ("id", np.int32),
            ("type", np.int32),
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("r", np.float32),
            ("pid", np.int32),
        ]

        def get_name(k: str) -> str:
            return name_map[k] if name_map is not None and k in name_map else k

        df = pd.read_csv(
            swc_file,
            sep=" ",
            comment="#",
            names=[get_name(k) for k, v in cols],
            dtype=cast(Any, {get_name(k): v for k, v in cols}),
        ).rename({get_name(k): k for k, v in cols})

        root = df.loc[0]["id"]
        if root != 0:
            df["id"] = df["id"] - root
            df["pid"] = df["pid"] - root

        df.loc[0, "pid"] = -1

        tree = Tree(df.shape[0], **{k: df[k].to_numpy() for k, v in cols})
        tree.source = os.path.abspath(swc_file)
        return tree
