"""SWC format."""

from typing import Any, Iterable, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

__all__ = ["SWCLike", "SWCTypeVar"]


class SWCLike:
    """Abstract class that including swc infomation."""

    source: str | None = None

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

    def to_swc(self, swc_path: str) -> None:
        """Write swc file."""
        ids, typee, pid = self.id(), self.type(), self.pid()
        x, y, z, r = self.x(), self.y(), self.z(), self.r()

        def get_row_str(idx: int) -> str:
            xx, yy, zz, rr = [f"{v[idx]:.4f}" for v in (x, y, z, r)]
            items = [idx, typee[idx], xx, yy, zz, rr, pid[idx]]
            return " ".join(map(str, items))

        with open(swc_path, "w", encoding="utf-8") as f:
            f.write(f"# source: {self.source if self.source else 'Unknown'}\n")
            f.write("# id type x y z r pid\n")
            for idx in ids:
                f.write(get_row_str(idx) + "\n")


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)
