"""Nueron node."""

from typing import Any, Generic, Iterable, List, overload

import numpy as np
import numpy.typing as npt

from .swc import SWCLike, SWCTypeVar

__all__ = ["Node", "NodeAttached", "Nodes"]


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

    def detach(self) -> Node:
        return Node(**{k: self[k] for k in self})


class Nodes(SWCLike):
    """Nodes of neuron tree."""

    class Node(NodeAttached["Nodes"]):
        """Node of neuron tree."""

    def __iter__(self) -> Iterable[Node]:
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        return self.id().shape[0]

    def __repr__(self) -> str:
        return f"{len(self)} Neuron nodes."

    # fmt:off
    @overload
    def __getitem__(self, key: int) -> Node: ...
    @overload
    def __getitem__(self, key: slice) -> List[Node]: ...
    @overload
    def __getitem__(self, key: str) -> npt.NDArray: ...
    # fmt:on
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.get_node(i) for i in range(*key.indices(len(self)))]

        if isinstance(key, int):
            length = len(self)
            if key < -length or key >= length:
                raise IndexError(f"The index ({key}) is out of range.")

            if key < 0:  # Handle negative indices
                key += length

            return self.get_node(key)

        if isinstance(key, str):
            return self.get_ndata(key)

        raise TypeError("Invalid argument type.")

    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray:
        raise NotImplementedError()

    def get_node(self, idx: int) -> Node:
        return self.Node(self, idx)

    def length(self) -> float:
        """Sum of length of stems."""
        xyz = self.xyz()
        return np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)).item()
