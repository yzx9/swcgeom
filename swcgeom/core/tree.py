"""Neuron tree."""

import itertools
import logging
import os
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from ..utils import padding1d
from .branch import Branch
from .node import Node
from .path import Path
from .segment import Segment, Segments
from .swc import SWCLike, eswc_cols, read_swc, swc_cols

__all__ = ["Tree"]

T, K = TypeVar("T"), TypeVar("K")


class Tree(SWCLike):
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(Node["Tree"]):
        """Neural node."""

        def parent(self) -> Union["Tree.Node", None]:
            return Tree.Node(self.attach, self.pid) if self.pid != -1 else None

        def children(self) -> List["Tree.Node"]:
            return [Tree.Node(self.attach, idx) for idx in self.child_ids()]

        def is_soma(self) -> bool:
            return self.id == 0

        def is_tip(self) -> bool:
            return self.id not in self.attach.pid()

        def get_branch(self) -> "Tree.Branch":
            logging.info(
                "`tree.get_branch` has been renamed to `tree.branch` and will be removed in next version"
            )
            return self.branch()

        def branch(self) -> "Tree.Branch":
            nodes: List["Tree.Node"] = [self]
            while not (nodes[-1].is_soma() or nodes[-1].is_bifurcation()):
                nodes.append(cast(Tree.Node, nodes[-1].parent()))

            nodes.reverse()
            while not (nodes[-1].is_bifurcation() or nodes[-1].is_tip()):
                nodes.append(nodes[-1].children()[0])

            return Tree.Branch(self.attach, [n.id for n in nodes])

        def radial_distance(self) -> float:
            """The end-to-end straight-line distance to soma."""
            return self.distance(self.attach.soma())

    class Path(Path["Tree"]):
        """Neural path."""

    class Segment(Segment["Tree"]):
        """Neural segment."""

    class Branch(Branch["Tree"]):
        """Neural branch."""

    ndata: dict[str, npt.NDArray]

    def __init__(
        self,
        n_nodes: int,
        *,
        id: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        type: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        pid: npt.NDArray[np.int32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        id = np.arange(0, n_nodes, step=1, dtype=np.int32) if id is None else id
        pid = np.arange(-1, n_nodes - 1, step=1, dtype=np.int32) if pid is None else pid

        ndata = {
            "id": padding1d(n_nodes, id, dtype=np.int32),
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

    def soma(self) -> Node:
        return self.node(0)

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.ndata[key]

    def get_bifurcations(self) -> List[Node]:
        """Get all node of bifurcations."""
        bifurcations: List[int] = []

        def collect_bifurcations(n: Tree.Node, children: List[None]) -> None:
            if len(children) > 1:
                bifurcations.append(n.id)

        self.traverse(leave=collect_bifurcations)
        return [self.node(i) for i in bifurcations]

    def get_tips(self) -> List[Node]:
        """Get all node of tips."""
        tip_ids = np.setdiff1d(self.id(), self.pid(), assume_unique=True)
        return [self.node(i) for i in tip_ids]

    def get_segments(self) -> Segments[Segment]:
        # pylint: disable-next=not-an-iterable
        return Segments([self.Segment(self, n.pid, n.id) for n in self[1:]])

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

    def get_paths(self) -> List[Path]:
        """Get all path from soma to tips."""
        path_dic: Dict[int, List[int]] = {}
        Paths = List[List[int]]

        def assign_path(n: Tree.Node, pre_path: List[int] | None) -> List[int]:
            path = [] if pre_path is None else pre_path.copy()
            path.append(n.id)
            path_dic[n.id] = path
            return path

        def collect_path(n: Tree.Node, children: List[Paths]) -> Paths:
            if len(children) == 0:
                return [path_dic[n.id]]

            return list(itertools.chain(*children))

        paths = self.traverse(enter=assign_path, leave=collect_path)
        # pylint: disable-next=not-an-iterable
        return [self.Path(self, idx) for idx in paths]

    # fmt: off
    @overload
    def traverse(self, *, enter: Callable[[Node, T | None], T], mode: Literal["dfs"] = ...) -> None: ...
    @overload
    def traverse(self, *, leave: Callable[[Node, list[K]], K], mode: Literal["dfs"] = ...) -> K: ...
    @overload
    def traverse(self, *,
        enter: Callable[[Node, T | None], T], leave: Callable[[Node, list[K]], K], mode: Literal["dfs"] = ...,
    ) -> K: ...
    # fmt: on

    def traverse(self, *, enter=None, leave=None, mode="dfs"):
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

        match mode:
            case "dfs":
                return self._traverse_dfs(enter=enter, leave=leave)
            case _:
                raise ValueError(f"unsupported mode: `{mode}`")

    def _traverse_dfs(self, *, enter=None, leave=None):
        """Traverse each nodes by dfs."""
        children_map = dict[int, list[int]]()
        for idx, pid in enumerate(self.pid()):
            children_map.setdefault(pid, [])
            children_map[pid].append(idx)

        # manual stack to avoid stack overflow in long projection
        stack: List[Tuple[int, bool]] = [(0, True)]  # (idx, first)
        params = {0: None}
        vals = {}

        while len(stack) != 0:
            idx, first = stack.pop()
            if first:
                pre = params.pop(idx)
                cur = enter(self[idx], pre) if enter is not None else None
                stack.append((idx, False))
                for child in children_map.get(idx, []):
                    stack.append((child, True))
                    params[child] = cur
            else:
                children = [vals.pop(i) for i in children_map.get(idx, [])]
                vals[idx] = leave(self[idx], children) if leave is not None else None

        return vals[0]

    def copy(self) -> "Tree":
        """Make a copy."""
        new_tree = Tree(len(self), **{k: v.copy() for k, v in self.ndata.items()})
        new_tree.source = self.source
        return new_tree

    def length(self) -> float:
        """Get length of tree."""
        return sum(s.length() for s in self.get_segments())

    @staticmethod
    def from_swc(swc_file: str, **kwargs) -> "Tree":
        """Read neuron tree from swc file.

        See Also
        --------
        ~swcgeom.read_swc
        """

        df = read_swc(swc_file, **kwargs)
        tree = Tree(df.shape[0], **{k: df[k].to_numpy() for k, v in swc_cols})
        tree.source = os.path.abspath(swc_file)
        return tree

    @staticmethod
    def from_eswc(swc_file: str, **kwargs) -> "Tree":
        """Read neuron tree from eswc file.

        See Also
        --------
        ~swcgeom.read_swc
        """

        kwargs.setdefault("extra_cols", [])
        kwargs["extra_cols"].extend(k for k, t in eswc_cols)
        return Tree.from_swc(swc_file, **kwargs)
