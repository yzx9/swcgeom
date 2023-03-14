"""SWC format utils.

Methods ending with a underline imply an in-place transformation.
"""

from typing import Callable, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Topology, get_dsu

__all__ = [
    "mark_roots_as_somas",
    "mark_roots_as_somas_",
    "link_roots_to_nearest",
    "link_roots_to_nearest_",
    "sort_nodes",
    "sort_nodes_",
    "sort_nodes_impl",
    "reset_index",
    "reset_index_",
]


def mark_roots_as_somas(
    df: pd.DataFrame, update_type: int | Literal[False] = 1
) -> pd.DataFrame:
    return _copy_and_apply(mark_roots_as_somas_, df, update_type=update_type)


def mark_roots_as_somas_(
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


def link_roots_to_nearest(df: pd.DataFrame) -> pd.DataFrame:
    return _copy_and_apply(link_roots_to_nearest_, df)


def link_roots_to_nearest_(df: pd.DataFrame) -> None:
    """Merge multiple roots in swc.

    The first root are reserved, and the others was.
    """
    dsu = get_dsu(df)
    roots = df[df["pid"] == -1].iterrows()
    next(roots)  # type: ignore # skip the first one
    for i, row in roots:
        vs = df[["x", "y", "z"]] - row[["x", "y", "z"]]
        dis = np.linalg.norm(vs.to_numpy(), axis=1)
        subtree = dsu == dsu[i]  # type: ignore
        dis = np.where(subtree, np.Infinity, dis)  # avoid link to same tree
        dsu = np.where(subtree, dsu[dis.argmin()], dsu)  # merge set
        df.loc[i, "pid"] = df["id"].iloc[dis.argmin()]  # type: ignore


def sort_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the indices of neuron tree.

    The index for parent are always less than children.
    """
    return _copy_and_apply(sort_nodes_, df)


def sort_nodes_(df: pd.DataFrame) -> None:
    """Sort the indices of neuron tree.

    The index for parent are always less than children.
    """
    ids, pids = df["id"].to_numpy(), df["pid"].to_numpy()
    (new_ids, new_pids), indices = sort_nodes_impl((ids, pids))
    for col in df.columns:
        df[col] = df[col][indices].to_numpy()

    df["id"], df["pid"] = new_ids, new_pids


def sort_nodes_impl(topology: Topology) -> Tuple[Topology, npt.NDArray[np.int32]]:
    """Sort the indices of neuron tree.

    Returns
    -------
    new_topology : Topology
    id_map : List of int
        Map from new id to original id.
    """
    old_ids, old_pids = topology
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
    return (new_ids, new_pids), indices


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reset node index to start with zero."""
    return _copy_and_apply(reset_index_, df)


def reset_index_(df: pd.DataFrame) -> None:
    """Reset node index to start with zero."""
    roots = df["pid"] == -1
    root_loc = roots.argmax()
    root_id = df.loc[root_loc, "id"]
    df["id"] = df["id"] - root_id
    df["pid"] = df["pid"] - root_id
    df.loc[root_loc, "pid"] = -1


def _copy_and_apply(fn: Callable, df: pd.DataFrame, *args, **kwargs):
    df = df.copy()
    fn(df, *args, **kwargs)
    return df
