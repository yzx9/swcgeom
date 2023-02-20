"""SWC format."""

import warnings
from copy import copy
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = []  # do not export anything


def mark_roots_as_somas(
    df: pd.DataFrame, update_type: int | Literal[False] = 1
) -> None:
    """Merge multiple roots in swc.

    The first root are reserved and others are linked to it.
    """
    roots = df["pid"] == -1
    root_loc = roots.argmax()
    root_id = df.loc[root_loc, "id"]
    df["pid"] = np.where(df["pid"] != -1, df["pid"], root_id)
    if update_type is not False:
        df["type"] = np.where(df["pid"] != -1, df["type"], update_type)
    df.loc[root_loc, "pid"] = -1


def link_roots_to_nearest(df: pd.DataFrame) -> None:
    """Merge multiple roots in swc.

    The first root are reserved, and the others was.
    """
    dsu = _get_dsu(df)
    roots = df[df["pid"] == -1].iterrows()
    next(roots)  # type: ignore # skip the first one
    for i, row in roots:
        vs = df[["x", "y", "z"]] - row[["x", "y", "z"]]
        dis = np.linalg.norm(vs.to_numpy(), axis=1)
        subtree = dsu == dsu[i]  # type: ignore
        dis = np.where(subtree, np.Infinity, dis)  # avoid link to same tree
        dsu = np.where(subtree, dsu[dis.argmin()], dsu)  # merge set
        df.loc[i, "pid"] = df["id"].iloc[dis.argmin()]  # type: ignore


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

    tree, lines = try_assemble_lines(lines, sort=False, **kwargs)
    while len(lines) > 0:
        t, lines = try_assemble_lines(lines, id_offset=len(tree), sort=False, **kwargs)
        tree = pd.concat([tree, t])

    tree = tree.reset_index()
    link_roots_to_nearest(tree)
    sort_nodes(tree)
    return tree


def try_assemble_lines(
    lines: List[pd.DataFrame],
    undirected: bool = True,
    thre: float = 0.2,
    id_offset: int = 0,
    sort: bool = True,
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
    sort : bool, default `True`
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
                dis = np.linalg.norm(
                    tree[["x", "y", "z"]] - line.iloc[p][["x", "y", "z"]], axis=1
                )
                ind = np.argmin(dis)
                if dis[ind] < thre:
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

    if sort:
        sort_nodes(tree)
    return tree, lines


def sort_swc(df: pd.DataFrame):
    warnings.warn(
        "`sort_swc` has been renamed to `sort_nodes`, and will be remove in next version",
        DeprecationWarning,
    )
    return sort_nodes(df)


def sort_nodes(df: pd.DataFrame):
    """Sort the indices of neuron tree.

    The index for parent are always less than children.
    """
    ids, pids = df["id"].to_numpy(), df["pid"].to_numpy()
    indices, new_ids, new_pids = sort_swc_impl(ids, pids)
    for col in df.columns:
        df[col] = df[col][indices].to_numpy()

    df["id"], df["pid"] = new_ids, new_pids


def sort_swc_impl(
    old_ids: npt.NDArray[np.int32], old_pids: npt.NDArray[np.int32]
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Sort the indices of neuron tree."""
    assert np.count_nonzero(old_pids == -1) == 1, "should be single root"

    id_map = np.full_like(old_ids, fill_value=-3)  # new_id to old_id
    new_pids = np.full_like(old_ids, fill_value=-3)
    new_id = 0
    first_root = old_ids[(old_pids == -1).argmax()]
    s: List[Tuple[npt.NDArray[np.int32], int]] = [(first_root, -1)]
    while len(s) != 0:
        old_id, new_pid = s.pop()
        id_map[new_id] = old_id
        new_pids[new_id] = new_pid
        s.extend((j, new_id) for j in old_ids[old_pids == old_id])  # (old_id, new_pid)
        new_id = new_id + 1

    id2idx = dict(zip(old_ids, range(len(old_ids))))  # old_id to old_idx
    indices = np.array([id2idx[i] for i in id_map], dtype=np.int32)  # new_id to old_idx
    new_ids = np.arange(len(new_pids))
    return indices, new_ids, new_pids


def reset_index(df: pd.DataFrame) -> None:
    """Reset node index to start with zero."""
    roots = df["pid"] == -1
    root_loc = roots.argmax()
    root_id = df.loc[root_loc, "id"]
    df["id"] = df["id"] - root_id
    df["pid"] = df["pid"] - root_id
    df.loc[root_loc, "pid"] = -1


def check_single_root(df: pd.DataFrame) -> bool:
    """Check is it only one root."""
    return len(np.unique(_get_dsu(df))) == 1


def _get_dsu(df: pd.DataFrame) -> npt.NDArray[np.int32]:
    """Get disjoint set union."""
    dsu = np.where(df["pid"] == -1, df["id"], df["pid"])  # Disjoint Set Union

    id2idx = dict(zip(df["id"], range(len(df))))
    dsu = np.array([id2idx[i] for i in dsu], dtype=np.int32)

    while True:
        flag = True
        for i, p in enumerate(dsu):
            if dsu[i] != dsu[p]:
                dsu[i] = dsu[p]
                flag = False

        if flag:
            break

    return dsu
