"""Assemble lines to swc."""

import warnings
from copy import copy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import SWCNames, get_names
from .normalizer import link_roots_to_nearest_, sort_nodes_

__all__ = ["assemble_lines", "try_assemble_lines"]


def assemble_lines(*args, **kwargs) -> pd.DataFrame:
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
    warnings.warn(
        "`assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations, and this will be removed in next "
        "version.",
        DeprecationWarning,
    )
    return assemble_lines_impl(*args, **kwargs)


def try_assemble_lines(*args, **kwargs) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
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
    names : SWCNames, optional

    Returns
    -------
    tree : ~pd.DataFrame
    remaining_lines : List of ~pd.DataFrame
    """
    warnings.warn(
        "`try_assemble_lines` has been replaced by "
        "`~.transforms.LinesToTree` because it can be easy assemble "
        "with other tansformations, and this will be removed in next "
        "version.",
        DeprecationWarning,
    )
    return try_assemble_lines_impl(*args, **kwargs)


# TODO: move the following codes to `transforms` module

EPS = 1e-5


def assemble_lines_impl(lines: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
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


def try_assemble_lines_impl(  # pylint: disable=too-many-arguments
    lines: List[pd.DataFrame],
    undirected: bool = True,
    thre: float = 0.2,
    id_offset: int = 0,
    sort_nodes: bool = True,
    *,
    names: Optional[SWCNames] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    names = get_names(names)
    lines = copy(lines)

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
                if dis[ind] > thre:
                    continue

                if dis[ind] < EPS:
                    line = line.drop((p + len(line)) % len(line)).reset_index(drop=True)

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
