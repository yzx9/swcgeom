"""SWC format."""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from typing_extensions import Self

from swcgeom.core.swc_utils import (
    SWCNames,
    SWCTypes,
    get_names,
    get_types,
    read_swc,
    to_swc,
)

__all__ = [
    "swc_cols",
    "eswc_cols",
    "SWCLike",
    "DictSWC",
    "SWCTypeVar",
    # TODO: `read_swc` has been deprecated and will be removed in next
    # version, import from `swcgeom.core.swc_utils` instead
    "read_swc",
]


swc_names_default = get_names()
swc_cols: List[Tuple[str, npt.DTypeLike]] = [
    (swc_names_default.id, np.int32),
    (swc_names_default.type, np.int32),
    (swc_names_default.x, np.float32),
    (swc_names_default.y, np.float32),
    (swc_names_default.z, np.float32),
    (swc_names_default.r, np.float32),
    (swc_names_default.pid, np.int32),
]

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
    comments: List[str] = []
    names: SWCNames
    types: SWCTypes

    def __init__(self) -> None:
        super().__init__()
        self.types = get_types()

    def __len__(self) -> int:
        return self.number_of_nodes()

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample,)."""
        return self.get_ndata(self.names.id)

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample,)."""
        return self.get_ndata(self.names.type)

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample,)."""
        return self.get_ndata(self.names.x)

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample,)."""
        return self.get_ndata(self.names.y)

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample,)."""
        return self.get_ndata(self.names.z)

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample,)."""
        return self.get_ndata(self.names.r)

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample,)."""
        return self.get_ndata(self.names.pid)

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the coordinates of shape(n_sample, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=1)

    def xyzw(self) -> npt.NDArray[np.float32]:
        """Get the homogeneous coordinates of shape(n_sample, 4)."""
        w = np.ones_like(self.x())
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
        comments: bool = True,
        id_offset: int = 1,
    ) -> str | None:
        """Write swc file."""
        data = []
        if source is not False:
            if not isinstance(source, str):
                source = self.source if self.source else "Unknown"
            data.append(f"source: {source}")
            data.append("")

        if comments is True:
            data.extend(self.comments)

        it = to_swc(
            self.get_ndata, comments=data, extra_cols=extra_cols, id_offset=id_offset
        )

        if fname is None:
            return "".join(it)

        with open(fname, "w", encoding="utf-8") as f:
            f.writelines(it)

        return None

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


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWCLike)


class DictSWC(SWCLike):
    """SWC implementation on dict."""

    ndata: Dict[str, npt.NDArray]

    def __init__(
        self,
        *,
        source: str = "",
        comments: Optional[Iterable[str]] = None,
        names: Optional[SWCNames] = None,
        **kwargs: npt.NDArray,
    ):
        super().__init__()
        self.source = source
        self.comments = list(comments) if comments is not None else []
        self.names = get_names(names)
        self.ndata = kwargs

    def keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def values(self) -> Iterable[npt.NDArray[Any]]:
        return self.ndata.values()

    def items(self) -> Iterable[Tuple[str, npt.NDArray[Any]]]:
        return self.ndata.items()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        return self.ndata[key]

    def copy(self) -> Self:
        """Make a copy."""
        return deepcopy(self)
