"""Utils for SWC format file."""

import os
from typing import Any, Dict, List, Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from .swc import SWC
from .tree import Tree

__all__ = ["from_swc", "to_swc", "to_sub_tree"]


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
    name_map : dict[str, str], optional
        Map standard name to actual name. The standard names are `id`,
        `type`, `x`, `y`, `z`, `r` and `pid`.
    """

    cols: List[Tuple[str, npt.DTypeLike]] = [
        ("id", np.int32),
        ("type", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("r", np.float32),
        ("pid", np.int32),
    ]

    def get_name(k: str) -> str:
        return name_map[k] if name_map is not None and k in name_map else k

    df = pd.read_csv(
        swc_path,
        sep=" ",
        comment="#",
        names=[get_name(k) for k, v in cols],
        dtype=cast(Any, {get_name(k): v for k, v in cols}),
    ).rename({get_name(k): k for k, v in cols})

    root = df.loc[0]["id"]
    if root != 0:
        df["id"] = df["id"] - root
        df["pid"] = df["pid"] - root

    df.loc[0, "pid"] = -1

    tree = Tree(df.shape[0], **{k: df[k].to_numpy() for k, v in cols})
    tree.source = os.path.abspath(swc_path)
    return tree


def to_swc(swc_like: SWC, swc_path: str) -> None:
    """Write swc file."""
    ids, typee, pid = swc_like.id(), swc_like.type(), swc_like.pid()
    x, y, z, r = swc_like.x(), swc_like.y(), swc_like.r(), swc_like.z()

    def get_row_str(idx: int) -> str:
        xx, yy, zz, rr = [f"{v[idx]:.4f}" for v in (x, y, z, r)]
        items = [ids[idx], typee[idx], xx, yy, zz, rr, pid[idx]]
        return " ".join(map(str, items))

    with open(swc_path, "w", encoding="utf-8") as f:
        f.write(f"# source: {swc_like.source if swc_like.source else 'Unknown'}\n")
        f.write("# id type x y z r pid\n")
        f.writelines(map(get_row_str, ids))


NData = Dict[str, npt.NDArray]


def to_sub_tree(
    swc_like: SWC, sub_id: npt.ArrayLike, sub_pid: npt.ArrayLike
) -> Tuple[SWC, Dict[int, int]]:
    """Create sub tree from origin tree."""

    sub_id = np.array(sub_id, dtype=np.int32)
    sub_pid = np.array(sub_pid, dtype=np.int32)
    n_nodes = sub_id.shape[0]

    id_map = {idx: i for i, idx in enumerate(sub_id)}
    new_pid = [id_map[i] if i != -1 else -1 for i in sub_pid]

    ndata = {k: swc_like.get_ndata(k)[sub_id] for k in swc_like.get_keys()}
    ndata.update(
        id=np.arange(0, n_nodes),
        pid=np.array(new_pid, dtype=np.int32),
    )

    tree = Tree(n_nodes, **ndata)
    tree.source = swc_like.source
    return tree, id_map
