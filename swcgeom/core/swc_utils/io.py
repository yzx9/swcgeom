"""Read and write swc format."""

import warnings
from collections import OrderedDict
from typing import Any, Callable, Iterable, List, Literal, Optional, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import SWCNames, get_names
from .checker import is_single_root
from .normalizer import (
    link_roots_to_nearest_,
    mark_roots_as_somas_,
    reset_index_,
    sort_nodes_,
)

__all__ = ["read_swc", "to_swc"]


def read_swc(
    swc_file: str,
    extra_cols: List[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort_nodes: bool = False,
    reset_index: bool = True,
    *,
    names: Optional[SWCNames] = None,
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
    names : SWCNames, optional
    """

    names = get_names(names)

    df = pd.read_csv(
        swc_file,
        sep=" ",
        comment="#",
        names=names.cols() + (extra_cols if extra_cols else []),
        dtype=cast(Any, dict(_get_dtypes(names))),
        index_col=False,
    )

    # fix swc
    if fix_roots is not False and np.count_nonzero(df[names.pid] == -1) > 1:
        match fix_roots:
            case "somas":
                mark_roots_as_somas_(df)
            case "nearest":
                link_roots_to_nearest_(df)
            case _:
                raise ValueError(f"unknown fix type `{fix_roots}`")

    if sort_nodes:
        sort_nodes_(df)
    elif reset_index:
        reset_index_(df)

    # check swc
    if not is_single_root(df, names=names):
        warnings.warn(f"not a simple tree in `{swc_file}`")

    if (df[names.pid] == -1).argmax() != 0:
        warnings.warn(f"root is not the first node in `{swc_file}`")

    return df


def to_swc(
    get_ndata: Callable[[str], npt.NDArray],
    *,
    extra_cols: Optional[List[str]] = None,
    id_offset: int = 1,
    names: Optional[SWCNames] = None,
) -> Iterable[str]:
    names = get_names(names)

    def get_v(k: str, idx: int) -> str:
        vs = get_ndata(k)
        v = vs[idx]
        if np.issubdtype(vs.dtype, np.floating):
            return f"{v:.4f}"

        if k == names.id or (k == names.pid and v != -1):
            v += id_offset

        return str(v)

    cols = names.cols() + (extra_cols if extra_cols is not None else [])
    yield f"# {' '.join(cols)}\n"
    for idx in get_ndata(names.id):
        yield " ".join(get_v(k, idx) for k in cols) + "\n"


def _get_dtypes(names: SWCNames) -> OrderedDict[str, np.dtype]:
    d = OrderedDict()
    d[names.id] = np.int32
    d[names.type] = np.int32
    d[names.x] = np.float32
    d[names.y] = np.float32
    d[names.z] = np.float32
    d[names.r] = np.float32
    d[names.pid] = np.int32
    return d
