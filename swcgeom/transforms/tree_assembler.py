"""Assemble a tree."""

from copy import copy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from swcgeom.core import Tree
from swcgeom.core.swc_utils import (
    SWCNames,
    get_names,
    link_roots_to_nearest_,
    sort_nodes_,
)
from swcgeom.transforms.base import Transform

EPS = 1e-5


class LinesToTree(Transform[List[pd.DataFrame], Tree]):
    """Assemble lines to swc."""

    def __init__(self, *, thre: float = 0.2, undirected: bool = True):
        """
        Parameters
        ----------
        thre : float, default `0.2`
            Connection threshold.
        undirected : bool, default `True`
            Both ends of a line can be considered connection point. If
            `False`, only the starting point.
        """
        super().__init__()
        self.thre = thre
        self.undirected = undirected

    def __call__(
        self, lines: Iterable[pd.DataFrame], *, names: Optional[SWCNames] = None
    ):
        return self.assemble(lines, names=names)

    def assemble(
        self,
        lines: Iterable[pd.DataFrame],
        *,
        undirected: bool = True,
        names: Optional[SWCNames] = None,
    ) -> pd.DataFrame:
        """Assemble lines to a tree.

        Assemble all the lines into a set of subtrees, and then connect
        them.

        Parameters
        ----------
        lines : List of ~pd.DataFrame
            An array of tables containing a line, columns should follwing
            the swc.
        undirected : bool, default `True`
            Forwarding to `self.try_assemble`.
        names : SWCNames, optional
            Forwarding to `self.try_assemble`.

        Returns
        -------
        tree : ~pd.DataFrame

        See Also
        --------
        self.try_assemble
        """

        tree, lines = self.try_assemble(
            lines, sort_nodes=False, undirected=undirected, names=names
        )
        while len(lines) > 0:
            t, lines = self.try_assemble(
                lines,
                id_offset=len(tree),
                sort_nodes=False,
                undirected=undirected,
                names=names,
            )
            tree = pd.concat([tree, t])

        tree = tree.reset_index()
        link_roots_to_nearest_(tree)
        sort_nodes_(tree)
        return tree

    def try_assemble(
        self,
        lines: Iterable[pd.DataFrame],
        *,
        id_offset: int = 0,
        undirected: bool = True,
        sort_nodes: bool = True,
        names: Optional[SWCNames] = None,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Trying assemble lines to a tree.

        Treat the first line as a tree, find a line whose shortest distance
        between the tree and the line is less than threshold, merge it into
        the tree, repeat until there are no line to merge, return tree and
        the remaining lines.

        Parameters
        ----------
        lines : List of ~pd.DataFrame
            An array of tables containing a line, columns should follwing
            the swc.
        id_offset : int, default `0`
            The offset of the line node id.
        undirected : bool, default `True`
            Both ends of a line can be considered connection point. If
            `False`, only the starting point.
        sort_nodes : bool, default `True`
            sort nodes of subtree.
        names : SWCNames, optional

        Returns
        -------
        tree : ~pandas.DataFrame
        remaining_lines : List of ~pandas.DataFrame
        """

        names = get_names(names)
        lines = copy(list(lines))

        tree = lines[0]
        tree[names.id] = id_offset + np.arange(len(tree))
        tree[names.pid] = tree[names.id] - 1
        tree.at[0, names.pid] = -1
        del lines[0]

        while True:
            for i, line in enumerate(lines):
                for p in [0, -1] if undirected else [0]:
                    xyz = [names.x, names.y, names.z]
                    vs = tree[xyz] - line.iloc[p][xyz]
                    dis = np.linalg.norm(vs, axis=1)
                    ind = np.argmin(dis)
                    if dis[ind] > self.thre:
                        continue

                    if dis[ind] < EPS:
                        line = line.drop((p + len(line)) % len(line)).reset_index(
                            drop=True
                        )

                    line[names.id] = id_offset + len(tree) + np.arange(len(line))
                    line[names.pid] = line[names.id] + (-1 if p == 0 else 1)
                    line.at[(p + len(line)) % len(line), names.pid] = tree.iloc[ind][
                        names.id
                    ]
                    tree = pd.concat([tree, line])
                    del lines[i]
                    break
                else:
                    continue

                break
            else:
                break

        if sort_nodes:
            sort_nodes_(tree)

        return tree, lines

    def extra_repr(self):
        return f"thre={self.thre}, undirected={self.undirected}"
