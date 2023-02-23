"""SWC format."""

import warnings
from typing import Any, Dict, Iterable, List, Literal, Tuple, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

from .swc_utils import check_single_root, link_roots_to_nearest, mark_roots_as_somas
from .swc_utils import reset_index as _reset_index
from .swc_utils import sort_nodes

__all__ = ["swc_cols", "eswc_cols", "read_swc", "SWCLike", "SWCTypeVar"]

swc_cols: List[Tuple[str, npt.DTypeLike]] = [
    ("id", np.int32),
    ("type", np.int32),
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("r", np.float32),
    ("pid", np.int32),
]

eswc_cols: List[Tuple[str, npt.DTypeLike]] = [
    ("level", np.int32),
    ("mode", np.int32),
    ("timestamp", np.int32),
    ("teraflyindex", np.int32),
    ("feature_value", np.int32),
]


class SWCLike:
    """Abstract class that including swc infomation."""

    source: str = ""

    def __len__(self) -> int:
        return self.number_of_nodes()

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample,)."""
        return self.get_ndata("id")

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample,)."""
        return self.get_ndata("type")

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample,)."""
        return self.get_ndata("x")

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample,)."""
        return self.get_ndata("y")

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample,)."""
        return self.get_ndata("z")

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample,)."""
        return self.get_ndata("r")

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample,)."""
        return self.get_ndata("pid")

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the coordinates of shape(n_sample, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=1)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the coordinates and radius array of shape(n_sample, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=1)

    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def get_adjacency_matrix(self) -> sp.coo_matrix:
        n_nodes = len(self)
        row, col = self.pid()[1:], self.id()[1:]  # ignore root
        triad = (np.ones_like(row), (row, col))
        return sp.coo_matrix(triad, shape=(n_nodes, n_nodes), dtype=np.int32)

    def number_of_nodes(self) -> int:
        """Get the number of nodes."""
        return self.id().shape[0]

    def number_of_edges(self) -> int:
        """Get the number of edges."""
        return self.number_of_nodes() - 1  # for tree structure: n = e + 1

    # fmt: off
    @overload
    def to_swc(self, fname: str, *, extra_cols: List[str] | None = ..., source: bool = ..., id_offset: int = ...) -> None: ...
    @overload
    def to_swc(self, *, extra_cols: List[str] | None = ..., source: bool = ..., id_offset: int = ...) -> str: ...
    # fmt: on
    def to_swc(
        self,
        fname: str | None = None,
        *,
        extra_cols: List[str] | None = None,
        source: bool = True,
        id_offset: int = 1,
    ) -> str | None:
        """Write swc file."""
        it = self._to_swc(extra_cols=extra_cols, source=source, id_offset=id_offset)
        if fname is None:
            return "".join(it)

        with open(fname, "w", encoding="utf-8") as f:
            f.writelines(it)

    def _to_swc(
        self, extra_cols: List[str] | None, source: bool, id_offset: int
    ) -> Iterable[str]:
        def get_v(name: str, idx: int) -> str:
            vs = self.get_ndata(name)
            v = vs[idx]
            if np.issubdtype(vs.dtype, np.floating):
                return f"{v:.4f}"

            if name == "id" or (name == "pid" and v != -1):
                v += id_offset

            return str(v)

        names = [name for name, _ in swc_cols]
        if extra_cols is not None:
            names.extend(extra_cols)

        if source:
            yield f"# source: {self.source if self.source else 'Unknown'}\n"
        yield f"# {' '.join(names)}\n"
        for idx in self.id():
            yield " ".join(get_v(name, idx) for name in names) + "\n"

    # fmt: off
    @overload
    def to_eswc(self, swc_path: str, **kwargs) -> None: ...
    @overload
    def to_eswc(self, **kwargs) -> str: ...
    # fmt: on
    def to_eswc(self, swc_path: str | None = None, **kwargs) -> str | None:
        kwargs.setdefault("swc_path", swc_path)
        kwargs.setdefault("extra_cols", [])
        kwargs["extra_cols"].extend(k for k, t in eswc_cols)
        return self.to_swc(**kwargs)


class DictSWC(SWCLike):
    ndata: Dict[str, npt.NDArray]

    def __init__(self, **kwargs: npt.NDArray):
        self.ndata = kwargs

    def keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.ndata[key]


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)


def read_swc(
    swc_file: str,
    extra_cols: List[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort: bool = False,
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
    sort : bool, default `False`
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
                mark_roots_as_somas(df)
            case "nearest":
                link_roots_to_nearest(df)
            case _:
                raise ValueError(f"unknown fix type: {fix_roots}")

    if sort:
        sort_nodes(df)
    elif reset_index:
        _reset_index(df)

    # check swc
    if not check_single_root(df):
        warnings.warn(f"core: not signle root, swc: {swc_file}")

    if (df["pid"] == -1).argmax() != 0:
        warnings.warn(f"core: root is not the first node, swc: {swc_file}")

    return df
