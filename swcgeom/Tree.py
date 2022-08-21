import copy
import os
import random
from functools import reduce
from typing import Any, Callable, Iterator, TypeVar, cast, overload

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.collections import LineCollection

from . import painter

_DATA = "data"
T = TypeVar("T")
K = TypeVar("K")


class Tree:
    """A neuron tree, which should be a binary tree in most cases."""

    class Node:
        """Node of neuron tree"""

        id: int
        type: int
        x: float
        y: float
        z: float
        r: float
        pid: int
        data: dict[str, Any]

        def __init__(
            self, id: int, type: int, x: float, y: float, z: float, r: float, pid: int
        ) -> None:
            self.id = id
            self.type = type
            self.x = x
            self.y = y
            self.z = z
            self.r = r
            self.pid = pid
            self.data = {}

        def xyz(self) -> npt.NDArray[np.float64]:
            """Get the `x`, `y`, `z` of branch, shape of (3,)"""
            return np.array([self.x, self.y, self.z], dtype=np.float64)

        def xyzr(self) -> npt.NDArray[np.float64]:
            """Get the `x`, `y`, `z`, `r` of branch, shape of (4,)"""
            return np.array([self.x, self.y, self.z, self.r], dtype=np.float64)

        def distance(self, b: "Tree.Node") -> float:
            """Get the distance of two nodes."""
            return np.linalg.norm(self.xyz() - b.xyz()).item()

        def __str__(self) -> str:
            x, y, z, r = ["%.4f" % f for f in [self.x, self.y, self.z, self.r]]
            items = [self.id, self.type, x, y, z, r, self.pid]
            return " ".join(map(str, items))

        def __setitem__(self, key: str, val: Any) -> None:
            self.data[key] = val

        def __getitem__(self, key: str) -> Any:
            return self.data[key]

        @classmethod
        def from_dataframe_row(cls, row: tuple[Any, ...]) -> "Tree.Node":
            """Read node from row of pd.DataFrame"""
            keys = ("id", "type", "x", "y", "z", "r", "pid")
            return cls(*[getattr(row, key) for key in keys])

    root: int
    G: nx.DiGraph
    scale: tuple[float, float, float, float]
    _source: str | None = None

    def __init__(self):
        self.G = nx.DiGraph()
        self.root = 1  # default to 1
        self.scale = (1, 1, 1, 1)

    def to_swc(self, swc_path: str) -> None:
        """Write swc file."""
        with open(swc_path, "w") as f:
            if self._source is not None:
                f.write(f"# source: {self._source}\n")

            f.write("# id type x y z r pid\n")
            f.writelines((str(n) + "\n" for n in self))

    def copy(self, G: bool = True) -> "Tree":
        """Make a copy.

        Parameters
        ----------
        G : bool.
            Skip copy G if false, default to true.
        """
        newTree = Tree()
        newTree._source = self._source
        if G:
            newTree.G = copy.deepcopy(self.G)
        return newTree

    def draw(
        self, color=painter.palette.momo, ax: plt.Axes | None = None, **kwargs
    ) -> tuple[plt.Axes, LineCollection]:
        """Draw neuron tree.

        Parameters
        ----------
        color : str, default to painter.palette.momo.
            Color of branch.
        ax : ~matplotlib.axes.Axes, optional.
            A subplot of `~matplotlib`. If None, a new one will be created.
        **kwargs : dict[str, Unknown].
            Forwarded to `matplotlib.collections.LineCollection`.

        Returns
        -------
        ax : ~matplotlib.axes.Axes.
            If provided, return as-is.
        collection : ~matplotlib.collections.LineCollection.
            Drawn line collection.
        """
        edges = np.array([[self[a].xyz(), self[b].xyz()] for a, b in self.G.edges])
        return painter.draw_lines(edges, ax=ax, color=color, **kwargs)

    TraverseEnter = Callable[[Node, T | None], T]
    TraverseLeave = Callable[[Node, list[T]], T]

    # fmt:off
    @overload
    def traverse(self, *, enter: TraverseEnter[T]) -> None: ...
    @overload
    def traverse(
        self, *, enter: TraverseEnter[T] | None = None, leave: TraverseLeave[K]
    ) -> K: ...
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
        enter : Callable[[Node, list[T]], T], optional.
            The callback when entering each node, it accepts two parameters,
            the first parameter is the current node, the second parameter is
            the parent's information T, and the root node receives an None.
        leave : Callable[[Node, T | None], T], optional.
            The callback when leaving each node. When leaving a node, subtree
            has already been traversed. Callback accepts two parameters, the
            first parameter is the current node, the second parameter is the
            children's information T, and the leaf node receives an empty list.
        """
        return self._traverse(enter, leave, self.root, None)

    def length(self) -> float:
        """Sum of length of all segments."""
        return self.traverse(
            leave=lambda n, acc: sum(acc)
            + sum(map(lambda b: n.distance(self[b]), self.G.neighbors(n.id)))
        )

    def random_cut(self, keep_percent: float) -> "Tree":
        """Random cut tree.

        Parameters
        ----------
        keep_percent : float.
            The percent of preserved segment length.

        Returns
        -------
        tree : Tree.
            A new tree.
        """
        tree = self.copy()
        length = (1 - keep_percent) * self.length()
        while tree.G.size() > 1 and length > 0:
            nodes = [i for i, d in tree.G.out_degree() if d == 0]
            i = nodes[random.randint(0, len(nodes) - 1)]
            predecessors = list(tree.G.predecessors(i))
            if len(predecessors) == 0:
                continue
            length -= tree[predecessors[0]].distance(tree[i])
            tree.G.remove_node(i)

        return tree

    def normalize(self) -> None:
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1"""

        _min, _max = self.traverse(
            leave=lambda a, children: reduce(
                lambda acc, cur: (
                    np.min(np.stack([acc[0], cur[0]]).transpose(), axis=1),
                    np.max(np.stack([acc[1], cur[1]]).transpose(), axis=1),
                ),
                children,
                (a.xyzr(), a.xyzr()),
            )
        )
        scale = 1 / (_max - _min)
        self.scale = tuple(scale)

        def scaler(a: Tree.Node, _):
            a.x, a.y, a.z, a.r = (a.xyzr() - _min) * scale

        self.traverse(leave=scaler)

    def standardize(self) -> None:
        """TODO"""
        pass

    def _add_edge(self, id: int, childId: int) -> None:
        self.G.add_edge(id, childId)

    def _add_node(self, node: Node) -> None:
        self.G.add_node(node.id, **{_DATA: node})

    def _traverse(
        self,
        enter: TraverseEnter[T] | None,
        leave: TraverseLeave[K] | None,
        id: int,
        pre: T | None,
    ) -> K | None:
        cur = enter(self[id], pre) if enter is not None else None
        children = [self._traverse(enter, leave, i, cur) for i in self.G.neighbors(id)]
        return leave(self[id], cast(list[K], children)) if leave is not None else None

    def __getitem__(self, id: int) -> Node:
        return self.G.nodes.data()[id][_DATA]  # type: ignore

    def __iter__(self) -> Iterator[Node]:
        return (self[i] for i in self.G)

    def __len__(self) -> int:
        return len(self.G)

    def __str__(self) -> str:
        nodes, edges = self.G.number_of_nodes(), self.G.number_of_edges()
        return f"Neuron Tree with {nodes} nodes and {edges} edges"

    @classmethod
    def from_swc(cls, swc_path: str) -> "Tree":
        """Read neuron tree from swc file."""

        self = cls()
        self._source = os.path.abspath(swc_path)
        df = pd.read_csv(
            swc_path,
            sep=" ",
            comment="#",
            names=["id", "type", "x", "y", "z", "r", "pid"],
            dtype={
                "id": np.int16,
                "type": np.int8,
                "x": np.float64,
                "y": np.float64,
                "z": np.float64,
                "r": np.float64,
                "pid": np.int16,
            },
        )

        nodes = [cls.Node.from_dataframe_row(r) for r in df.itertuples()]
        self.G.add_nodes_from([(n.id, {_DATA: n}) for n in nodes])
        self.G.add_edges_from([(n.pid, n.id) for n in nodes if n.pid != -1])
        self.root = df.iloc[0]["id"]
        return self
