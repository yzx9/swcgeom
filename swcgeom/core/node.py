"""Nueron node."""

from typing import Any, Generic, Iterable

import numpy as np
import numpy.typing as npt

from .swc import DictSWC, SWCTypeVar

__all__ = ["Node"]


class Node(Generic[SWCTypeVar]):
    """Neural node."""

    attach: SWCTypeVar
    idx: int

    # fmt: off
    @property
    def id(self) -> int: return self["id"]
    @id.setter
    def id(self, v: int): self["id"] = v

    @property
    def type(self) -> int: return self["type"]
    @type.setter
    def type(self, v: int): self["type"] = v

    @property
    def x(self) -> float: return self["x"]
    @x.setter
    def x(self, v: float): self["x"] = v

    @property
    def y(self) -> float: return self["y"]
    @y.setter
    def y(self, v: float): self["y"] = v

    @property
    def z(self) -> float: return self["z"]
    @z.setter
    def z(self, v: float): self["z"] = v

    @property
    def r(self) -> float: return self["r"]
    @r.setter
    def r(self, v: float): self["r"] = v

    @property
    def pid(self) -> int: return self["pid"]
    @pid.setter
    def pid(self, v: int): self["pid"] = v
    # fmt: on

    def __init__(self, attach: SWCTypeVar, idx: int) -> None:
        super().__init__()
        self.attach = attach
        self.idx = idx

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
        return self.attach.id()[self.attach.pid() == self.id]

    def is_bifurcation(self) -> bool:
        return len(self.child_ids()) > 1

    def is_tip(self) -> bool:
        return len(self.child_ids()) == 0

    def detach(self) -> "Node":
        """Detach from current attached object."""
        attact = DictSWC(**{k: self[k] for k in self.keys()})
        return Node(attact, self.idx)
