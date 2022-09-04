"""Neuron tree."""

import os
from typing import Any, Callable, TypedDict, TypeVar, cast, overload

import matplotlib.axes
import matplotlib.collections
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils import painter

__all__ = ["Tree", "Node", "NodeDetached"]

T, K = TypeVar("T"), TypeVar("K")


class NodeBase:
    """Node of neuron tree."""

    def __str__(self) -> str:
        return self.format_swc()

    def __getitem__(self, k: str) -> Any:
        raise NotImplementedError()

    def __setitem__(self, k: str, v: Any) -> Any:
        raise NotImplementedError()

    # fmt: off
    @property
    def type(self) -> int: return self["type"]
    @type.setter
    def type(self, v: int): self["type"] = v

    @property
    def x(self) -> float: return self["x"]
    @x.setter
    def x(self, v: float): self["x"] = v

    @property
    def y(self) -> float: return self["y"]
    @y.setter
    def y(self, v: float): self["y"] = v

    @property
    def z(self) -> float: return self["z"]
    @z.setter
    def z(self, v: float): self["z"] = v

    @property
    def r(self) -> float: return self["r"]
    @r.setter
    def r(self, v: float): self["r"] = v
    # fmt: on

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z` of branch, an array of shape (3,)"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z`, `r` of branch, an array of shape (4,)"""
        return np.array([self.x, self.y, self.z, self.r], dtype=np.float32)

    def distance(self, b: "NodeBase") -> float:
        """Get the distance of two nodes."""
        return np.linalg.norm(self.xyz() - b.xyz()).item()

    def format_swc(self) -> str:
        """Get the SWC format string."""
        x, y, z, r = [f"{f:.4f}" for f in [self.x, self.y, self.z, self.r]]
        items = [0, self.type, x, y, z, r, 0]  # TODO: id, pid
        return " ".join(map(str, items))


class NodeDetached(NodeBase):
    """Detached node that do not depend on the tree."""

    ndata: dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.ndata = {}

    def __getitem__(self, k: str) -> Any:
        return self.ndata[k]

    def __setitem__(self, k: str, v: Any) -> Any:
        self.ndata[k] = v


class Node(NodeBase):
    """Node of neuron tree"""

    tree: "Tree"
    idx: int

    def __init__(self, tree: "Tree", idx: int) -> None:
        super().__init__()
        self.tree = tree
        self.idx = idx

    def __getitem__(self, k: str) -> Any:
        return self.tree.ndata[k][self.idx]

    def __setitem__(self, k: str, v: Any) -> None:
        self.tree.ndata[k][self.idx] = v


class Tree:
    """A neuron tree, which should be a binary tree in most cases."""

    ndata: dict[str, npt.NDArray]
    # Need edata?

    source: str | None

    def __init__(
        self,
        *,
        ids: npt.NDArray[np.int32] | None = None,
        types: npt.NDArray[np.int32] | None = None,
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        pid: npt.NDArray[np.int32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        n_nodes = self.number_of_nodes()
        self.ndata = {
            "id": np.zeros(n_nodes, dtype=np.int32) if ids is None else ids,
            "type": np.zeros(n_nodes, dtype=np.int32) if types is None else types,
            "x": np.zeros(n_nodes, dtype=np.float32) if x is None else x,
            "y": np.zeros(n_nodes, dtype=np.float32) if y is None else y,
            "z": np.zeros(n_nodes, dtype=np.float32) if z is None else z,
            "r": np.ones(n_nodes, dtype=np.float32) if r is None else r,
            "pid": np.ones(n_nodes, dtype=np.int32) if pid is None else pid,
            **kwargs,
        }

        self.source = None

    def __len__(self) -> int:
        """Get number of nodes."""
        return self.number_of_nodes()

    def __repr__(self) -> str:
        nodes, edges = self.number_of_nodes(), self.number_of_edges()
        return f"Neuron Tree with {nodes} nodes and {edges} edges"

    def __getitem__(self, idx: int) -> Node:
        """Get node by id."""
        return Node(self, idx)

    # fmt:off
    def id(self)   -> npt.NDArray[np.int32]:   return self.ndata["id"]
    def type(self) -> npt.NDArray[np.int32]:   return self.ndata["type"]
    def x(self)    -> npt.NDArray[np.float32]: return self.ndata["x"]
    def y(self)    -> npt.NDArray[np.float32]: return self.ndata["y"]
    def z(self)    -> npt.NDArray[np.float32]: return self.ndata["z"]
    def r(self)    -> npt.NDArray[np.float32]: return self.ndata["r"]
    def pid(self)  -> npt.NDArray[np.int32]:   return self.ndata["pid"]
    # fmt:on

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get array of shape(N, 3)"""
        return np.array([self.x(), self.y(), self.z()])

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get array of shape(N, 4)"""
        return np.array([self.x(), self.y(), self.z(), self.r()])

    def number_of_nodes(self) -> int:  # pylint: disable=missing-function-docstring
        return self.id().shape[0]

    def number_of_edges(self) -> int:  # pylint: disable=missing-function-docstring
        return self.number_of_nodes() - 1

    def to_swc(self, swc_path: str) -> None:
        """Write swc file."""
        ids = self.id()
        types = self.type()
        xyzr = self.xyzr()
        pid = self.pid()

        def get_line_str(idx: int) -> str:
            x, y, z, r = [f"{f:.4f}" for f in xyzr[idx]]
            items = [ids[idx], types[idx], x, y, z, r, pid[idx]]
            return " ".join(map(str, items))

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write(f"# source: {self.source if self.source else 'Unknown'}\n")
            f.write("# id type x y z r pid\n")
            f.writelines(map(get_line_str, ids))

    def draw(
        self,
        color: str | None = painter.palette.momo,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.collections.LineCollection]:
        """Draw neuron tree.

        Parameters
        ----------
        color : str, optional
            Color of branch. If `None`, the default color will be enabled.
        ax : ~matplotlib.axes.Axes, optional
            A subplot of `~matplotlib`. If `None`, a new one will be created.
        **kwargs : dict[str, Unknown]
            Forwarded to `~matplotlib.collections.LineCollection`.

        Returns
        -------
        ax : ~matplotlib.axes.Axes
            If provided, return as-is.
        collection : ~matplotlib.collections.LineCollection
            Drawn line collection.
        """
        xyz = self.xyz()  # (N, 3)
        edges = np.array([xyz[range(self.number_of_nodes())], xyz[self.pid()]])
        return painter.draw_lines(edges, ax=ax, color=color, **kwargs)

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
        for pid in self.pid():
            if pid == -1:
                continue

            children_map.setdefault(pid, [])
            children_map[pid].append(pid)

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

    @classmethod
    def copy(cls) -> "Tree":
        """Make a copy."""

        new_tree = cls(**{k: v.copy() for k, v in cls.ndata.items()})
        new_tree.source = cls.source
        return new_tree

    @classmethod
    def normalize(cls) -> "Tree":
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1."""
        new_tree = cls.copy()
        for key in ["x", "y", "z", "r"]:  # TODO: does r is the same?
            v_max = np.max(new_tree.ndata[key])
            v_min = np.min(new_tree.ndata[key])
            new_tree.ndata[key] = (new_tree.ndata[key] - v_min) / v_max

        return new_tree


SWCNameMap = TypedDict(
    "SWCNameMap",
    {"id": str, "type": str, "x": str, "y": str, "z": str, "r": str, "pid": str},
    total=False,
)


def from_swc(swc_path: str, name_map: SWCNameMap | None = None) -> Tree:
    """Read neuron tree from swc file.

    Parameters
    ----------
    swc_path : str
        Path of swc file, the id should be consecutively incremented.
    name_map : dict[str, str]
        Map standard name to actual name. The standard names are `id`,
        `type`, `x`, `y`, `z`, `r` and `pid`.
    """

    def get_name(key: str) -> str:
        return name_map[key] if name_map and key in name_map else key

    cols = {
        "id": np.int32,
        "type": np.int32,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "r": np.float32,
        "pid": np.int32,
    }
    names = [get_name(k) for k in cols]
    dtype = {get_name(k): v for k, v in cols.items()}
    df = pd.read_csv(swc_path, sep=" ", comment="#", names=names, dtype=dtype)

    root = df.iloc[0][get_name("id")]
    if root != 0:
        df[get_name("id")] = df[get_name("id")] - root
        df[get_name("pid")] = df[get_name("pid")] - root

    if df.iloc[0][get_name("pid")] != -1:
        df.iloc[0][get_name("pid")] = -1

    tree = Tree(**{k: df[get_name(k)].to_numpy() for k in cols})
    tree.source = os.path.abspath(swc_path)
    return tree
