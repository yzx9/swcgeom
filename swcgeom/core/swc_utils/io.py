"""Read and write swc format."""

import warnings
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from .checker import is_single_root
from .normalizer import (
    link_roots_to_nearest_,
    mark_roots_as_somas_,
    reset_index_,
    sort_nodes_,
)

__all__ = ["swc_cols", "read_swc", "to_swc"]

swc_cols: List[Tuple[str, npt.DTypeLike]] = [
    ("id", np.int32),
    ("type", np.int32),
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("r", np.float32),
    ("pid", np.int32),
]


def read_swc(
    swc_file: str,
    extra_cols: List[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort_nodes: bool = False,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Read swc file.

    Parameters
    ----------
    swc_file : str
        Path of swc file, the id should be consecutively incremented.
    extra_cols : List[str], optional
        Read more cols in swc file.
    fix_roots : `somas`|`nearest`|False, default `False`
        Fix multiple roots.
    sort_nodes : bool, default `False`
        Sort the indices of neuron tree, the index for parent are
        always less than children.
    reset_index : bool, default `True`
        Reset node index to start with zero, DO NOT set to false if
        you are not sure what will happend.
    """

    names = [k for k, v in swc_cols]
    if extra_cols:
        names.extend(extra_cols)

    df = pd.read_csv(
        swc_file,
        sep=" ",
        comment="#",
        names=names,
        dtype=cast(Any, dict(swc_cols)),
        index_col=False,
    )

    # fix swc
    if fix_roots is not False and np.count_nonzero(df["pid"] == -1) > 1:
        match fix_roots:
            case "somas":
                mark_roots_as_somas_(df)
            case "nearest":
                link_roots_to_nearest_(df)
            case _:
                raise ValueError(f"unknown fix type: {fix_roots}")

    if sort_nodes:
        sort_nodes_(df)
    elif reset_index:
        reset_index_(df)

    # check swc
    if not is_single_root(df):
        warnings.warn(f"core: not signle root, swc: {swc_file}")

    if (df["pid"] == -1).argmax() != 0:
        warnings.warn(f"core: root is not the first node, swc: {swc_file}")

    return df


def to_swc(
    get_ndata: Callable[[str], npt.NDArray],
    extra_cols: Optional[List[str]] = None,
    id_offset: int = 1,
) -> Iterable[str]:
    def get_v(name: str, idx: int) -> str:
        vs = get_ndata(name)
        v = vs[idx]
        if np.issubdtype(vs.dtype, np.floating):
            return f"{v:.4f}"

        if name == "id" or (name == "pid" and v != -1):
            v += id_offset

        return str(v)

    names = [name for name, _ in swc_cols]
    if extra_cols is not None:
        names.extend(extra_cols)

    yield f"# {' '.join(names)}\n"
    for idx in get_ndata("id"):
        yield " ".join(get_v(name, idx) for name in names) + "\n"
