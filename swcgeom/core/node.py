from typing import Any

import numpy as np
import numpy.typing as npt

from .tree import Tree


__all__ = ["Node", "NodeDetached"]


class NodeBase:
    """Node of neuron tree"""

    def __str__(self) -> str:
        return self.format_swc()

    def __getitem__(self, k: str) -> Any:
        raise NotImplementedError()

    def __setitem__(self, k: str, v: Any) -> Any:
        raise NotImplementedError()

    # fmt: off
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
    # fmt: on

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z` of branch, an array of shape (3,)"""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the `x`, `y`, `z`, `r` of branch, an array of shape (4,)"""
        return np.array([self.x, self.y, self.z, self.r], dtype=np.float32)

    def distance(self, b: "NodeBase") -> float:
        """Get the distance of two nodes."""
        return np.linalg.norm(self.xyz() - b.xyz()).item()

    def format_swc(self) -> str:
        """Get the SWC format string."""
        x, y, z, r = ["%.4f" % f for f in [self.x, self.y, self.z, self.r]]
        items = [0, self.type, x, y, z, r, 0]  # TODO: id, pid
        return " ".join(map(str, items))


class NodeDetached(NodeBase):
    ndata: dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        self.ndata = {}

    def __getitem__(self, k: str) -> Any:
        return self.ndata[k]

    def __setitem__(self, k: str, v: Any) -> Any:
        self.ndata[k] = v


class Node(NodeBase):
    """Node of neuron tree"""

    tree: Tree
    idx: int

    def __init__(self, tree: Tree, idx: int) -> None:
        super().__init__()
        self.tree = tree
        self.idx = idx

    def __getitem__(self, k: str) -> Any:
        return self.tree.ndata[k][self.idx]

    def __setitem__(self, k: str, v: Any) -> None:
        self.tree.ndata[k][self.idx] = v
