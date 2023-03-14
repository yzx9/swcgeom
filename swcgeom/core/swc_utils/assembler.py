"""Assemble lines to swc."""

from copy import copy
from typing import List, Tuple

import numpy as np
import pandas as pd

from .normalizer import link_roots_to_nearest_, sort_nodes_


def assemble_lines(lines: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Assemble lines to a tree.

    Assemble all the lines into a set of subtrees, and then connect
    them.

    Parameters
    ----------
    lines : List of ~pd.DataFrame
        An array of tables containing a line, columns should follwing
        the swc.
    **kwargs
        Forwarding to `try_assemble_lines`

    Returns
    -------
    tree : ~pd.DataFrame

    See Also
    --------
    ~swcgeom.core.swc_utils.try_assemble_lines
    """

    tree, lines = try_assemble_lines(lines, sort_nodes=False, **kwargs)
    while len(lines) > 0:
        t, lines = try_assemble_lines(
            lines, id_offset=len(tree), sort_nodes=False, **kwargs
        )
        tree = pd.concat([tree, t])

    tree = tree.reset_index()
    link_roots_to_nearest_(tree)
    sort_nodes_(tree)
    return tree


def try_assemble_lines(
    lines: List[pd.DataFrame],
    undirected: bool = True,
    thre: float = 0.2,
    id_offset: int = 0,
    sort_nodes: bool = True,
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
    undirected : bool, default `True`
        Both ends of a line can be considered connection point. If
        `False`, only the starting point.
    thre : float, default `0.2`
        Connection threshold.
    id_offset : int, default `0`
        The offset of the line node id.
    sort_nodes : bool, default `True`
        sort nodes of subtree

    Returns
    -------
    tree : ~pd.DataFrame
    remaining_lines : List of ~pd.DataFrame
    """
    lines = copy(lines)
    tree = lines[0]
    tree["id"] = id_offset + np.arange(len(tree))
    tree["pid"] = tree["id"] - 1
    tree.at[0, "pid"] = -1
    del lines[0]

    while True:
        for i, line in enumerate(lines):
            for p in [0, -1] if undirected else [0]:
                vs = tree[["x", "y", "z"]] - line.iloc[p][["x", "y", "z"]]
                dis = np.linalg.norm(vs, axis=1)
                ind = np.argmin(dis)
                if dis[ind] > thre:
                    continue

                line["id"] = id_offset + len(tree) + np.arange(len(line))
                line["pid"] = line["id"] + (-1 if p == 0 else 1)
                line.at[(p + len(line)) % len(line), "pid"] = tree.iloc[ind]["id"]
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
