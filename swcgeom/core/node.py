"""Nueron node."""

from typing import Any, Generic

import numpy as np
import numpy.typing as npt

from .swc import SWCTypeVar

__all__ = ["Node", "NodeAttached"]


class _Node:
    r"""Node of neuron tree."""

    def __str__(self) -> str:
        return self.format_swc()

    def __getitem__(self, k: str) -> Any:
        raise NotImplementedError()

    def __setitem__(self, k: str, v: Any) -> Any:
        raise NotImplementedError()

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

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z` of node, an array of shape (3,)"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z`, `r` of node, an array of shape (4,)"""
        return np.array([self.x, self.y, self.z, self.r], dtype=np.float32)

    def distance(self, b: "_Node") -> float:
        """Get the distance of two nodes."""
        return np.linalg.norm(self.xyz() - b.xyz()).item()

    def format_swc(self) -> str:
        """Get the SWC format string."""
        x, y, z, r = [f"{f:.4f}" for f in [self.x, self.y, self.z, self.r]]
        items = [self.id, self.type, x, y, z, r, self.pid]
        return " ".join(map(str, items))


class Node(_Node):
    """Detached node that do not depend on the external object."""

    ndata: dict[str, Any]

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ndata = kwargs

    def __getitem__(self, k: str) -> Any:
        return self.ndata[k]

    def __setitem__(self, k: str, v: Any) -> Any:
        self.ndata[k] = v


class NodeAttached(_Node, Generic[SWCTypeVar]):
    """Node attached to external object."""

    attach: SWCTypeVar
    idx: int

    def __repr__(self) -> str:
        return self.format_swc()

    def __init__(self, attach: SWCTypeVar, idx: int) -> None:
        super().__init__()
        self.attach = attach
        self.idx = idx

    def __getitem__(self, key: str) -> Any:
        return self.attach.get_ndata(key)[self.idx]

    def __setitem__(self, k: str, v: Any) -> None:
        self.attach.get_ndata(k)[self.idx] = v

    def child_ids(self) -> npt.NDArray[np.int32]:
        pid = self.attach.pid()
        return pid[pid == self.id]

    def is_bifurcation(self) -> bool:
        return len(self.child_ids()) > 1

    def is_tip(self) -> bool:
        return len(self.child_ids()) == 0

    def detach(self) -> Node:
        return Node(**{k: self[k] for k in self})
