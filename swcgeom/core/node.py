"""Nueron node."""

import warnings
from typing import Any, Generic, Iterable

import numpy as np
import numpy.typing as npt

from swcgeom.core.swc import DictSWC, SWCTypeVar
from swcgeom.core.swc_utils import SWCNames

__all__ = ["Node"]


class Node(Generic[SWCTypeVar]):
    """Neural node."""

    attach: SWCTypeVar
    idx: int | np.integer
    names: SWCNames

    # fmt: off
    @property
    def id(self) -> int: return self[self.names.id]
    @id.setter
    def id(self, v: int): self[self.names.id] = v

    @property
    def type(self) -> int: return self[self.names.type]
    @type.setter
    def type(self, v: int): self[self.names.type] = v

    @property
    def x(self) -> float: return self[self.names.x]
    @x.setter
    def x(self, v: float): self[self.names.x] = v

    @property
    def y(self) -> float: return self[self.names.y]
    @y.setter
    def y(self, v: float): self[self.names.y] = v

    @property
    def z(self) -> float: return self[self.names.z]
    @z.setter
    def z(self, v: float): self[self.names.z] = v

    @property
    def r(self) -> float: return self[self.names.r]
    @r.setter
    def r(self, v: float): self[self.names.r] = v

    @property
    def pid(self) -> int: return self[self.names.pid]
    @pid.setter
    def pid(self, v: int): self[self.names.pid] = v
    # fmt: on

    def __init__(self, attach: SWCTypeVar, idx: int | np.integer) -> None:
        super().__init__()
        self.attach = attach
        self.idx = idx
        self.names = attach.names

    def __getitem__(self, key: str) -> Any:
        return self.attach.get_ndata(key)[self.idx]

    def __setitem__(self, k: str, v: Any) -> None:
        self.attach.get_ndata(k)[self.idx] = v

    def __str__(self) -> str:
        return self.format_swc()

    def __repr__(self) -> str:
        return self.format_swc()

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z` of node, an array of shape (3,)"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z`, `r` of node, an array of shape (4,)"""
        return np.array([self.x, self.y, self.z, self.r], dtype=np.float32)

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def distance(self, b: "Node") -> float:
        """Get the distance of two nodes."""
        return np.linalg.norm(self.xyz() - b.xyz()).item()

    def format_swc(self) -> str:
        """Get the SWC format string."""
        x, y, z, r = [f"{f:.4f}" for f in [self.x, self.y, self.z, self.r]]
        items = [self.id, self.type, x, y, z, r, self.pid]
        return " ".join(map(str, items))

    def child_ids(self) -> npt.NDArray[np.int32]:
        warnings.warn(
            "`Node.child_ids` has been deprecated since v0.3.1 and "
            "will be removed in next version",
            DeprecationWarning,
        )
        return self.attach.id()[self.attach.pid() == self.id]

    def is_bifurcation(self) -> bool:
        return np.count_nonzero(self.attach.pid() == self.id) > 1

    def is_tip(self) -> bool:
        return self.id not in self.attach.pid()

    def detach(self) -> "Node[DictSWC]":
        """Detach from current attached object."""
        # pylint: disable=consider-using-dict-items
        attact = DictSWC(
            **{k: np.array([self[k]]) for k in self.keys()},
            source=self.attach.source,
            names=self.names,
        )
        attact.ndata[self.names.id] = np.array([0])
        attact.ndata[self.names.pid] = np.array([-1])
        return Node(attact, 0)
