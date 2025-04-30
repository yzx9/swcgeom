
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Neuron tree."""

import itertools
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Literal, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import deprecated

from swcgeom.core.branch import Branch
from swcgeom.core.compartment import Compartment, Compartments
from swcgeom.core.node import Node
from swcgeom.core.path import Path
from swcgeom.core.swc import DictSWC, eswc_cols
from swcgeom.core.swc_utils import SWCNames, get_names, read_swc, traverse
from swcgeom.core.tree_utils_impl import Mapping, get_subtree_impl
from swcgeom.utils import PathOrIO, padding1d

__all__ = ["Tree"]

T = TypeVar("T")
K = TypeVar("K")


class Tree(DictSWC):
    """A neuron tree, which should be a binary tree in most cases."""

    class Node(Node["Tree"]):
        """Neural node."""

        def parent(self) -> Union["Tree.Node", None]:
            return Tree.Node(self.attach, self.pid) if self.pid != -1 else None

        def children(self) -> list["Tree.Node"]:
            children = self.attach.id()[self.attach.pid() == self.id]
            return [Tree.Node(self.attach, idx) for idx in children]

        def branch(self) -> "Tree.Branch":
            ns: list["Tree.Node"] = [self]
            while not ns[-1].is_furcation() and (p := ns[-1].parent()) is not None:
                ns.append(p)

            ns.reverse()
            while not (ns[-1].is_furcation() or ns[-1].is_tip()):
                ns.append(ns[-1].children()[0])

            return Tree.Branch(self.attach, [n.id for n in ns])

        def radial_distance(self) -> float:
            """The end-to-end straight-line distance to soma."""
            return self.distance(self.attach.soma())

        def subtree(self, *, out_mapping: Mapping | None = None) -> "Tree":
            """Get subtree from node.

            Args:
                out_mapping: Map from new id to old id.
            """

            n_nodes, ndata, source, names = get_subtree_impl(
                self.attach, self.id, out_mapping=out_mapping
            )
            return Tree(n_nodes, **ndata, source=source, names=names)

        def is_root(self) -> bool:
            return self.parent() is None

        def is_soma(self) -> bool:  # TODO: support multi soma, e.g. 3 points
            return self.type == self.attach.types.soma and self.is_root()

        @overload
        def traverse(
            self, *, enter: Callable[[Node, T | None], T], mode: Literal["dfs"] = ...
        ) -> None: ...
        @overload
        def traverse(
            self, *, leave: Callable[[Node, list[K]], K], mode: Literal["dfs"] = ...
        ) -> K: ...
        @overload
        def traverse(
            self,
            *,
            enter: Callable[[Node, T | None], T],
            leave: Callable[[Node, list[K]], K],
            mode: Literal["dfs"] = ...,
        ) -> K: ...
        def traverse(self, **kwargs):  # type: ignore
            """Traverse from node.

            See Also:
                ~Tree.traverse
            """
            return self.attach.traverse(root=self.idx, **kwargs)

    class Path(Path["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural path."""

    class Compartment(Compartment["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural compartment."""

    Segment = Compartment  # Alias

    class Branch(Branch["Tree"]):
        # TODO: should returns `Tree.Node`
        """Neural branch."""

    def __init__(
        self,
        n_nodes: int,
        *,
        source: str = "",
        comments: Iterable[str] | None = None,
        names: SWCNames | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        names = get_names(names)

        if names.id not in kwargs:
            kwargs[names.id] = np.arange(0, n_nodes, step=1, dtype=np.int32)

        if names.pid not in kwargs:
            kwargs[names.pid] = np.arange(-1, n_nodes - 1, step=1, dtype=np.int32)

        ndata = {
            names.id: padding1d(n_nodes, kwargs.pop(names.id, None), dtype=np.int32),
            names.type: padding1d(
                n_nodes, kwargs.pop(names.type, None), dtype=np.int32
            ),
            names.x: padding1d(n_nodes, kwargs.pop(names.x, None), dtype=np.float32),
            names.y: padding1d(n_nodes, kwargs.pop(names.y, None), dtype=np.float32),
            names.z: padding1d(n_nodes, kwargs.pop(names.z, None), dtype=np.float32),
            names.r: padding1d(
                n_nodes, kwargs.pop(names.r, None), dtype=np.float32, padding_value=1
            ),
            names.pid: padding1d(n_nodes, kwargs.pop(names.pid, None), dtype=np.int32),
        }
        # ? padding other columns
        super().__init__(
            **ndata, **kwargs, source=source, comments=comments, names=names
        )

    def __iter__(self) -> Iterator[Node]:
        return (self[i] for i in range(len(self)))

    def __repr__(self) -> str:
        n_nodes, n_edges = self.number_of_nodes(), self.number_of_edges()
        return f"Neuron Tree with {n_nodes} nodes and {n_edges} edges"

    @overload
    def __getitem__(self, key: slice) -> list[Node]: ...
    @overload
    def __getitem__(self, key: int) -> Node: ...
    @overload
    def __getitem__(self, key: str) -> npt.NDArray: ...
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

    def soma(self, type_check: bool = True) -> Node:
        """Get soma of neuron."""
        # TODO: find soma, see also: https://neuromorpho.org/myfaq.jsp
        n = self.node(0)
        if type_check and n.type != self.types.soma:
            raise ValueError(f"no soma found in: {self.source}")
        return n

    def get_furcations(self) -> list[Node]:
        """Get all node of furcations."""
        furcations: list[int] = []

        def collect_furcations(n: Tree.Node, children: list[None]) -> None:
            if len(children) > 1:
                furcations.append(n.id)

        self.traverse(leave=collect_furcations)
        return [self.node(i) for i in furcations]

    @deprecated("Use `get_furcations` instead")
    def get_bifurcations(self) -> list[Node]:
        """Get all node of furcations.

        .. deprecated:: 0.17.2
            Deprecated due to the wrong spelling of furcation. For now, it is just an
            alias of `get_furcations` and raise a warning. It will be change to raise
            an error in the future.
        """
        return self.get_furcations()

    def get_tips(self) -> list[Node]:
        """Get all node of tips."""
        tip_ids = np.setdiff1d(self.id(), self.pid(), assume_unique=True)
        return [self.node(i) for i in tip_ids]

    def get_compartments(self) -> Compartments[Compartment]:
        return Compartments(self.Compartment(self, n.pid, n.id) for n in self[1:])

    def get_segments(self) -> Compartments[Compartment]:  # Alias
        return self.get_compartments()

    def get_branches(self) -> list[Branch]:
        def collect_branches(
            node: "Tree.Node", pre: list[tuple[list[Tree.Branch], list[int]]]
        ) -> tuple[list[Tree.Branch], list[int]]:
            if len(pre) == 1:
                branches, child = pre[0]
                child.append(node.id)
                return branches, child

            branches: list[Tree.Branch] = []

            for sub_branches, child in pre:
                child.append(node.id)
                child.reverse()
                sub_branches.append(Tree.Branch(self, np.array(child, dtype=np.int32)))
                sub_branches.reverse()
                branches.extend(sub_branches)

            return branches, [node.id]

        branches, _ = self.traverse(leave=collect_branches)
        return branches

    def get_paths(self) -> list[Path]:
        """Get all path from soma to tips."""
        path_dic: dict[int, list[int]] = {}

        def assign_path(n: Tree.Node, pre_path: list[int] | None) -> list[int]:
            path = [] if pre_path is None else pre_path.copy()
            path.append(n.id)
            path_dic[n.id] = path
            return path

        def collect_path(
            n: Tree.Node, children: list[list[list[int]]]
        ) -> list[list[int]]:
            if len(children) == 0:
                return [path_dic[n.id]]

            return list(itertools.chain(*children))

        paths = self.traverse(enter=assign_path, leave=collect_path)
        return [self.Path(self, idx) for idx in paths]

    def get_neurites(self, type_check: bool = True) -> Iterable["Tree"]:
        """Get neurites from soma."""
        return (n.subtree() for n in self.soma(type_check).children())

    def get_dendrites(self, type_check: bool = True) -> Iterable["Tree"]:
        """Get dendrites."""
        types = [self.types.apical_dendrite, self.types.basal_dendrite]
        children = self.soma(type_check).children()
        return (n.subtree() for n in children if n.type in types)

    @overload
    def traverse(
        self,
        *,
        enter: Callable[[Node, T | None], T],
        root: int | np.integer = ...,
        mode: Literal["dfs"] = ...,
    ) -> None: ...
    @overload
    def traverse(
        self,
        *,
        leave: Callable[[Node, list[K]], K],
        root: int | np.integer = ...,
        mode: Literal["dfs"] = ...,
    ) -> K: ...
    @overload
    def traverse(
        self,
        *,
        enter: Callable[[Node, T | None], T],
        leave: Callable[[Node, list[K]], K],
        root: int | np.integer = ...,
        mode: Literal["dfs"] = ...,
    ) -> K: ...
    def traverse(self, *, enter=None, leave=None, **kwargs):
        """Traverse nodes.

        Args:
            enter: (n: Node, parent: T | None) => T
            leave: (n: Node, children: list[T]) => T

        See Also:
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

    @staticmethod
    def from_data_frame(
        df: pd.DataFrame,
        source: str = "",
        *,
        comments: Iterable[str] | None = None,
        names: SWCNames | None = None,
    ) -> "Tree":
        """Read neuron tree from data frame."""
        names = get_names(names)
        tree = Tree(
            df.shape[0],
            **{k: df[k].to_numpy() for k in names.cols()},
            source=source,
            comments=comments,
            names=names,
        )
        return tree

    @classmethod
    def from_swc(cls, swc_file: PathOrIO, **kwargs) -> "Tree":
        """Read neuron tree from swc file.

        See Also:
            ~swcgeom.core.swc_utils.read_swc
        """

        try:
            df, comments = read_swc(swc_file, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(f"fails to read swc: {swc_file}") from e

        source = os.path.abspath(swc_file) if isinstance(swc_file, str) else ""
        return cls.from_data_frame(df, source=source, comments=comments)

    @classmethod
    def from_eswc(
        cls, swc_file: str, extra_cols: list[str] | None = None, **kwargs
    ) -> "Tree":
        """Read neuron tree from eswc file.

        See Also:
            ~swcgeom.Tree.from_swc
        """
        extra_cols = extra_cols or []
        extra_cols.extend(k for k, t in eswc_cols)
        return cls.from_swc(swc_file, extra_cols=extra_cols, **kwargs)
