"""SWC format."""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from typing_extensions import Self

from .swc_utils import read_swc, swc_cols, to_swc

__all__ = ["swc_cols", "eswc_cols", "read_swc", "SWCLike", "DictSWC", "SWCTypeVar"]


eswc_cols: List[Tuple[str, npt.DTypeLike]] = [
    ("level", np.int32),
    ("mode", np.int32),
    ("timestamp", np.int32),
    ("teraflyindex", np.int32),
    ("feature_value", np.int32),
]


class SWCLike(ABC):
    """ABC of SWC."""

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

    def xyzw(self) -> npt.NDArray[np.float32]:
        """Get the homogeneous coordinates of shape(n_sample, 4)."""
        w = np.zeros_like(self.x())
        return np.stack([self.x(), self.y(), self.z(), w], axis=1)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the coordinates and radius array of shape(n_sample, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=1)

    @abstractmethod
    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    @abstractmethod
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
    def to_swc(self, fname: str, *, extra_cols: List[str] | None = ..., source: bool | str = ..., id_offset: int = ...) -> None: ...
    @overload
    def to_swc(self, *, extra_cols: List[str] | None = ..., source: bool | str = ..., id_offset: int = ...) -> str: ...
    # fmt: on
    def to_swc(
        self,
        fname: Optional[str] = None,
        *,
        extra_cols: Optional[List[str]] = None,
        source: bool | str = True,
        id_offset: int = 1,
    ) -> str | None:
        """Write swc file."""
        it = self._to_swc(extra_cols=extra_cols, source=source, id_offset=id_offset)
        if fname is None:
            return "".join(it)

        with open(fname, "w", encoding="utf-8") as f:
            f.writelines(it)

        return None

    def _to_swc(self, source: bool | str, **kwargs) -> Iterable[str]:
        if source is not False:
            if not isinstance(source, str):
                source = self.source if self.source else "Unknown"
            yield f"# source: {source}\n"

        yield from to_swc(self.get_ndata, **kwargs)

    # fmt: off
    @overload
    def to_eswc(self, fname: str, **kwargs) -> None: ...
    @overload
    def to_eswc(self, **kwargs) -> str: ...
    # fmt: on
    def to_eswc(
        self,
        fname: Optional[str] = None,
        swc_path: Optional[str] = None,
        extra_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> str | None:
        if swc_path is None:
            warnings.warn(
                "`swc_path` has been renamed to `fname` since v0.5.1, "
                "and will be removed in next version",
                DeprecationWarning,
            )
            fname = swc_path

        extra_cols = extra_cols or []
        extra_cols.extend(k for k, t in eswc_cols)
        return self.to_swc(fname, extra_cols=extra_cols, **kwargs)  # type: ignore


class DictSWC(SWCLike):
    """SWC implementation on dict."""

    ndata: Dict[str, npt.NDArray]

    def __init__(self, **kwargs: npt.NDArray):
        super().__init__()
        self.ndata = kwargs

    def keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.ndata[key]

    def copy(self) -> Self:
        """Make a copy."""
        return deepcopy(self)


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)
