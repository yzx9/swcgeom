"""SWC format."""

import warnings
from typing import Any, Iterable, Literal, List, Tuple, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

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


def read_swc(
    swc_file: str,
    extra_cols: List[str] | None = None,
    fix_roots: Literal["somas", "nearest", False] = False,
    sort: bool = False,
    reindex: bool = True,
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
        TODO
    reindex : bool, default `True`
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
                swc_fix_roots_as_somas(df)
            case "nearest":
                swc_fix_roots_by_link_nearest(df)
            case _:
                raise ValueError(f"unknown fix type: {fix_roots}")

    if sort:
        swc_sort(df)
    elif reindex:
        swc_reindex(df)

    # check swc
    if (k := np.count_nonzero(df["pid"] == -1)) == 0:
        warnings.warn(f"core: no root found, swc: {swc_file}")
    elif k > 1:
        warnings.warn(f"core: more than one root found, swc: {swc_file}")

    if (df["pid"] == -1).argmax() != 0:
        warnings.warn(f"core: root is not the first node, swc: {swc_file}")

    return df


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
    def to_swc(self, swc_path: str, *, extra_cols: List[str] | None = ..., id_offset: int = ...) -> None: ...
    @overload
    def to_swc(self, *, extra_cols: List[str] | None = ..., id_offset: int = ...) -> str: ...
    # fmt: on
    def to_swc(
        self,
        swc_path: str | None = None,
        *,
        extra_cols: List[str] | None = None,
        id_offset: int = 1,
    ) -> str | None:
        """Write swc file."""
        it = self._to_swc(extra_cols=extra_cols, id_offset=id_offset)
        if swc_path is None:
            return "".join(it)

        with open(swc_path, "w", encoding="utf-8") as f:
            f.writelines(it)

    def _to_swc(self, extra_cols: List[str] | None, id_offset: int) -> Iterable[str]:
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

        yield f"# source: {self.source if self.source else 'Unknown'}\n"
        yield f"# {' '.join(names)}\n"
        for idx in self.id():
            yield " ".join(get_v(name, idx) for name in names) + "\n"


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)


def swc_fix_roots_as_somas(df: pd.DataFrame) -> None:
    """Merge multiple roots in swc.

    The first root are reserved and others are linked to it.
    """
    roots = df["pid"] == -1
    root_loc = roots.argmax()
    root_id = df.loc[root_loc, "id"]
    df["pid"] = np.where(df["pid"] != -1, df["pid"], root_id)
    df.loc[root_loc, "pid"] = -1


def swc_fix_roots_by_link_nearest(df: pd.DataFrame) -> None:
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


def swc_sort(df: pd.DataFrame):
    """Sort the indices of neuron tree.

    The index for parent compartments are always less than child
    compartments.
    """
    ids, pids = df["id"].to_numpy(), df["pid"].to_numpy()
    indices, new_pids = swc_sort_tree_impl(ids, pids)
    for col in df.columns:
        df[col] = df[col][indices]  # type: ignore

    df["id"] = np.arange(len(new_pids))
    df["pid"] = new_pids


def swc_sort_tree_impl(
    old_ids: npt.NDArray[np.int32], old_pids: npt.NDArray[np.int32]
) -> Tuple[List[int], npt.NDArray[np.int32]]:
    """Sort the indices of neuron tree."""
    assert np.count_nonzero(old_pids == -1) == 1, "not only one root"

    id_map = np.full_like(old_ids, fill_value=-3)  # new_id to old_id
    new_pids = np.full_like(old_ids, fill_value=-3)
    new_id = 0
    s: List[Tuple[int, int]] = [
        (old_ids[(old_pids == -1).argmax()], -1)
    ]  # (old_id, new_pid)
    while len(s) != 0:
        old_id, new_pid = s.pop()
        id_map[new_id] = old_id
        new_pids[new_id] = new_pid
        s.extend((j, new_id) for j in old_ids[old_pids == old_id])
        new_id = new_id + 1

    id2idx = dict(zip(old_ids, range(len(old_ids))))  # old_id to old_idx
    indices = [id2idx[i] for i in id_map]  # new_id to old_idx
    return indices, new_pids


def swc_reindex(df: pd.DataFrame) -> None:
    """Reset node index to start with zero."""
    roots = df["pid"] == -1
    root_loc = roots.argmax()
    root_id = df.loc[root_loc, "id"]
    df["id"] = df["id"] - root_id
    df["pid"] = df["pid"] - root_id
    df.loc[root_loc, "pid"] = -1


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
