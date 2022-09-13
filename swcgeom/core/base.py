"""Utils for SWC format file."""

from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

__all__ = ["SWC", "Node", "NodeAttached"]


class SWC:
    """Abstract class that including swc infomation."""

    ndata: dict[str, npt.NDArray[Any]]
    source: str | None

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample,)."""
        return self.ndata["id"]

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample,)."""
        return self.ndata["type"]

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample,)."""
        return self.ndata["x"]

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample,)."""
        return self.ndata["y"]

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample,)."""
        return self.ndata["z"]

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample,)."""
        return self.ndata["r"]

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample,)."""
        return self.ndata["pid"]

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the coordinates of shape(n_sample, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=1)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the coordinates and radius array of shape(n_sample, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=1)


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
        items = [self.id, self.type, x, y, z, r, self.id]  # TODO: id, pid
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


T = TypeVar("T", bound=SWC)


class NodeAttached(_Node, Generic[T]):
    """Node attached to external object."""

    attach: T
    idx: int

    def __str__(self) -> str:
        return self.format_swc()

    def __init__(self, attach: T, idx: int) -> None:
        super().__init__()
        self.attach = attach
        self.idx = idx

    def __getitem__(self, k: str) -> Any:
        return self.attach.ndata[k][self.idx]

    def __setitem__(self, k: str, v: Any) -> None:
        self.attach.ndata[k][self.idx] = v

    def detach(self) -> Node:
        return Node(**{k: self[k] for k in self})
