"""Neuron tree."""

import itertools
import os
import warnings
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import Self

from ..utils import padding1d
from .branch import Branch
from .node import Node
from .path import Path
from .segment import Segment, Segments
from .swc import DictSWC, eswc_cols
from .swc_utils import SWCNames, get_names, read_swc, traverse

__all__ = ["Tree"]

T, K = TypeVar("T"), TypeVar("K")


class Tree(DictSWC):
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(Node["Tree"]):
        """Neural node."""

        def parent(self) -> Union["Tree.Node", None]:
            return Tree.Node(self.attach, self.pid) if self.pid != -1 else None

        def children(self) -> List["Tree.Node"]:
            children = self.attach.id()[self.attach.pid() == self.id]
            return [Tree.Node(self.attach, idx) for idx in children]

        def get_branch(self) -> "Tree.Branch":
            warnings.warn(
                "`Tree.Node.get_branch` has been renamed to "
                "`Tree.Node.branch` since v0.3.1 and will be removed "
                "in next version",
                DeprecationWarning,
            )
            return self.branch()

        def branch(self) -> "Tree.Branch":
            ns: List["Tree.Node"] = [self]
            while not ns[-1].is_bifurcation() and (p := ns[-1].parent()) is not None:
                ns.append(p)

            ns.reverse()
            while not (ns[-1].is_bifurcation() or ns[-1].is_tip()):
                ns.append(ns[-1].children()[0])

            return Tree.Branch(self.attach, [n.id for n in ns])

        def is_soma(self) -> bool:
            return self.id == 0

        def radial_distance(self) -> float:
            """The end-to-end straight-line distance to soma."""
            return self.distance(self.attach.soma())

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
        def traverse(self, **kwargs):  # type: ignore
            """Traverse from node.

            See Also
            --------
            ~Tree.traverse
            """
            return self.attach.traverse(root=self.idx, **kwargs)

    class Path(Path["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural path."""

    class Segment(Segment["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural segment."""

    class Branch(Branch["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural branch."""

    def __init__(
        self,
        n_nodes: int,
        *,
        # pylint: disable-next=redefined-builtin
        id: Optional[npt.NDArray[np.int32]] = None,
        # pylint: disable-next=redefined-builtin
        type: Optional[npt.NDArray[np.int32]] = None,
        x: Optional[npt.NDArray[np.float32]] = None,
        y: Optional[npt.NDArray[np.float32]] = None,
        z: Optional[npt.NDArray[np.float32]] = None,
        r: Optional[npt.NDArray[np.float32]] = None,
        pid: Optional[npt.NDArray[np.int32]] = None,
        names: Optional[SWCNames] = None,
        **kwargs: npt.NDArray,
    ) -> None:
        names = get_names(names)
        id = np.arange(0, n_nodes, step=1, dtype=np.int32) if id is None else id
        pid = np.arange(-1, n_nodes - 1, step=1, dtype=np.int32) if pid is None else pid

        ndata = {
            names.id: padding1d(n_nodes, id, dtype=np.int32),
            names.type: padding1d(n_nodes, type, dtype=np.int32),
            names.x: padding1d(n_nodes, x),
            names.y: padding1d(n_nodes, y),
            names.z: padding1d(n_nodes, z),
            names.r: padding1d(n_nodes, r, padding_value=1),
            names.pid: padding1d(n_nodes, pid, dtype=np.int32),
        }
        super().__init__(**ndata, **kwargs, names=names)

    def __iter__(self) -> Iterator[Node]:
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
        return self.ndata.keys()

    def node(self, idx: int | np.integer) -> Node:
        return self.Node(self, idx)

    def soma(self) -> Node:
        return self.node(0)

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
        return Segments(self.Segment(self, n.pid, n.id) for n in self[1:])

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
        return [self.Path(self, idx) for idx in paths]

    # fmt: off
    @overload
    def traverse(self, *,
                 enter: Callable[[Node, T | None], T],
                 root: int | np.integer = ..., mode: Literal["dfs"] = ...) -> None: ...
    @overload
    def traverse(self, *,
                 enter: Optional[Callable[[Node, T | None], T]] = ...,
                 leave: Callable[[Node, List[K]], K],
                 root: int | np.integer = ..., mode: Literal["dfs"] = ...) -> K: ...
    # fmt: on

    def traverse(self, *, enter=None, leave=None, **kwargs):
        """Traverse nodes.

        Parameters
        ----------
        enter : (n: Node, parent: T | None) => T, optional
        leave : (n: Node, children: List[T]) => T, optional

        See Also
        --------
        ~swc_utils.traverse
        """

        def wrap(fn) -> Callable | None:
            if fn is None:
                return None

            def fn_wrapped(idx, *args, **kwargs):
                return fn(self[idx], *args, **kwargs)

            return fn_wrapped

        topology = (self.id(), self.pid())
        enter, leave = wrap(enter), wrap(leave)
        return traverse(topology, enter=enter, leave=leave, **kwargs)  # type: ignore

    def length(self) -> float:
        """Get length of tree."""
        return sum(s.length() for s in self.get_segments())

    @classmethod
    def from_data_frame(
        cls, df: pd.DataFrame, source: str = "", *, names: Optional[SWCNames] = None
    ) -> "Tree":
        """Read neuron tree from data frame."""
        names = get_names(names)
        tree = Tree(
            df.shape[0], **{k: df[k].to_numpy() for k in names.cols()}, names=names
        )
        tree.source = source
        return tree

    @classmethod
    def from_swc(cls, swc_file: str, **kwargs) -> Self:
        """Read neuron tree from swc file.

        See Also
        --------
        ~swcgeom.core.swc_utils.read_swc
        """

        df = read_swc(swc_file, **kwargs)
        source = os.path.abspath(swc_file)
        return cls.from_data_frame(df, source)

    @classmethod
    def from_eswc(
        cls, swc_file: str, extra_cols: Optional[List[str]] = None, **kwargs
    ) -> Self:
        """Read neuron tree from eswc file.

        See Also
        --------
        ~swcgeom.Tree.from_swc
        """
        extra_cols = extra_cols or []
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(swc_file, extra_cols=extra_cols, **kwargs)
