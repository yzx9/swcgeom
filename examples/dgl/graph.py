# SPDX-FileCopyrightText: 2022-2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: unlicense

"""Dgl grapth transforms."""

from typing import cast

import dgl

from swcgeom import Tree
from swcgeom.transforms import Transform


class ToDGLGraph(Transform[Tree, dgl.DGLGraph]):
    """Transform tree to dgl graph.

    NOTE: You SHOULD initially decide to construct your own transformation, even if we
    are attempting to provide additional capabilities for you because there are so
    many custom initial options in graph and this class is more of a toy and template.
    """

    keys: list[str] | None
    to_bidirected: bool

    def __init__(
        self, to_bidirected: bool = False, keys: list[str] | None = None
    ) -> None:
        """Transform tree to dgl graph.

        Args:
            to_bidirected: Whether to return bidirected graph.
            keys: Copy these keys as ndata of graph.
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
