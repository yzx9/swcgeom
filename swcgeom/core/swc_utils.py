"""SWC format."""

from typing import Literal, Tuple

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
        df.loc[i, "pid"] = df["id"][dis.argmin()]  # type: ignore


def sort_swc(df: pd.DataFrame):
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
    s = [(old_ids[(old_pids == -1).argmax()], -1)]
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
