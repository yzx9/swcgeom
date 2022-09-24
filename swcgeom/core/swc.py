"""SWC format."""

from typing import Any, Iterable, List, Tuple, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp

__all__ = ["swc_cols", "read_swc", "SWCLike", "SWCTypeVar"]

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
    swc_file: str, extra_cols: List[str] | None = None, reindex=True
) -> pd.DataFrame:
    """Read swc file.

    Parameters
    ----------
    swc_file : str
        Path of swc file, the id should be consecutively incremented.
    extra_cols : List[str], optional
        Read more cols in swc file.
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

    if reindex:
        root = df.loc[0]["id"]
        if root != 0:
            df["id"] = df["id"] - root
            df["pid"] = df["pid"] - root

        df.loc[0, "pid"] = -1

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

    def to_swc(self, swc_path: str, extra_cols: List[str] | None = None) -> None:
        """Write swc file."""
        names = [name for name, _ in swc_cols]
        if extra_cols:
            names.extend(extra_cols)

        def get_v(value: npt.NDArray, idx: int) -> str:
            if np.issubdtype(value.dtype, np.floating):
                return f"{value[idx]:.4f}"

            return str(value[idx])

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write(f"# source: {self.source if self.source else 'Unknown'}\n")
            f.write(f"# {' '.join(names)}\n")
            for idx in self.id():
                values = [get_v(self.get_ndata(name), idx) for name in names]
                f.write(f"{' '.join(values)}\n")


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)
