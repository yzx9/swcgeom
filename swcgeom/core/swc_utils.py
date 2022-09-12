"""Utils for SWC format file."""

import os
from typing import Any, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from .tree import Tree

__all__ = ["SWCProtocol", "from_swc", "to_swc"]


class SWCProtocol(Protocol):
    """Abstract class that including swc infomation."""

    ndata: dict[str, npt.NDArray[Any]]
    source: str | None
    # fmt: off
    def id(self)   -> npt.NDArray[np.int32]:   ... # pylint: disable=invalid-name
    def type(self) -> npt.NDArray[np.int32]:   ...
    def x(self)    -> npt.NDArray[np.float32]: ...
    def y(self)    -> npt.NDArray[np.float32]: ...
    def z(self)    -> npt.NDArray[np.float32]: ...
    def r(self)    -> npt.NDArray[np.float32]: ...
    def pid(self)  -> npt.NDArray[np.int32]:   ...
    # fmt:on


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

    def get_name(key: str) -> str:
        return name_map[key] if name_map and key in name_map else key

    cols = {
        "id": np.int32,
        "type": np.int32,
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
        "r": np.float32,
        "pid": np.int32,
    }
    names = [get_name(k) for k in cols]
    dtype = {get_name(k): v for k, v in cols.items()}
    df = pd.read_csv(swc_path, sep=" ", comment="#", names=names, dtype=dtype)

    root = df.iloc[0][get_name("id")]
    if root != 0:
        df[get_name("id")] = df[get_name("id")] - root
        df[get_name("pid")] = df[get_name("pid")] - root

    if df.iloc[0][get_name("pid")] != -1:
        df.iloc[0][get_name("pid")] = -1

    tree = Tree(df.shape[0], **{k: df[get_name(k)].to_numpy() for k in cols})
    tree.source = os.path.abspath(swc_path)
    return tree


def to_swc(nodes: SWCProtocol, swc_path: str) -> None:
    """Write swc file."""
    ids, typee, pid = nodes.id(), nodes.type(), nodes.pid()
    x, y, z, r = nodes.x(), nodes.y(), nodes.r(), nodes.z()

    def get_row_str(idx: int) -> str:
        xx, yy, zz, rr = [f"{v[idx]:.4f}" for v in (x, y, z, r)]
        items = [ids[idx], typee[idx], xx, yy, zz, rr, pid[idx]]
        return " ".join(map(str, items))

    with open(swc_path, "w", encoding="utf-8") as f:
        f.write(f"# source: {nodes.source if nodes.source else 'Unknown'}\n")
        f.write("# id type x y z r pid\n")
        f.writelines(map(get_row_str, ids))
