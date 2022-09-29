"""Dgl grapth transforms."""

from typing import List, cast

import dgl

from ...core import Tree
from .. import Transform

__all__ = ["ToDGLGraph"]


class ToDGLGraph(Transform[Tree, dgl.DGLGraph]):
    """Transofrm tree to dgl graph.

    Notes
    -----
    You SHOULD initially decide to constrct your own transformation,
    even if we are attempting to provide additional capabilities for
    you because there are so many custom initial options in graph and
    this class is more of a toy and template.
    """

    keys: List[str] | None
    to_bidirected: bool

    def __init__(
        self,
        to_bidirected: bool = False,
        keys: List[str] | None = None,
    ) -> None:
        """Transofrm tree to dgl graph.

        Parameters
        ----------
        to_bidirected : bool, default to `False`
            If True, return bidirected graph.
        keys : List[str], optional
            Copy these keys as ndata of graph.
        """

        self.to_bidirected = to_bidirected
        self.keys = keys

    def __call__(self, x: Tree) -> dgl.DGLGraph:
        g = dgl.graph((x.id()[1:], x.pid()[1:]), num_nodes=x.number_of_nodes())
        if self.to_bidirected:
            g = cast(dgl.DGLGraph, dgl.to_bidirected(g))

        if self.keys:
            for k in self.keys:
                g.ndata[k] = x[k]

        return g

    def __repr__(self) -> str:
        return (
            "ToDGLGraph"
            + ("-ToBidirected" if self.to_bidirected else "")
            + (("-" + "-".join(self.keys)) if self.keys else "")
        )
