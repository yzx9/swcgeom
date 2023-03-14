"""Depth distribution of tree."""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from ..core import BranchTree, Tree

__all__ = ["NodeFeatures", "BifurcationFeatures", "TipFeatures"]


class NodeFeatures:
    """Evaluate node feature of tree."""

    tree: Tree

    @cached_property
    def _branch_tree(self) -> BranchTree:
        return BranchTree.from_tree(self.tree)

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_radial_distance(self) -> npt.NDArray[np.float32]:
        """Get the end-to-end straight-line distance to soma.

        Returns
        -------
        radial_distance : npt.NDArray[np.float32]
            Array of shape (N,).
        """
        xyz = self.tree.xyz() - self.tree.soma().xyz()
        radial_distance = np.linalg.norm(xyz, axis=1)
        return radial_distance

    def get_branch_order(self) -> npt.NDArray[np.int32]:
        """Get branch order of tree.

        Bifurcation order is the number of bifurcations between current
        position and the root.

        Returns
        -------
        order : npt.NDArray[np.int32]
            Array of shape (k,), which k is the number of branchs.
        """
        order = np.zeros_like(self._branch_tree.id(), dtype=np.int32)

        def assign_depth(n: Tree.Node, pre_depth: int | None) -> int:
            cur_order = pre_depth + 1 if pre_depth is not None else 0
            order[n.id] = cur_order
            return cur_order

        self._branch_tree.traverse(enter=assign_depth)
        nodes = [n.id for n in self._branch_tree if not n.is_tip()]  # filter tips
        return order[nodes]


class _SubsetNodesFeatures(ABC):
    _features: NodeFeatures

    @property
    @abstractmethod
    def nodes(self) -> npt.NDArray[np.bool_]:
        raise NotImplementedError()

    def __init__(self, features: NodeFeatures) -> None:
        self._features = features

    def get_radial_distance(self) -> npt.NDArray[np.float32]:
        return self._features.get_radial_distance()[self.nodes]

    @classmethod
    def from_tree(cls, tree: Tree) -> Self:
        return cls(NodeFeatures(tree))


class BifurcationFeatures(_SubsetNodesFeatures):
    """Evaluate bifurcation node feature of tree."""

    @cached_property
    def nodes(self) -> npt.NDArray[np.bool_]:
        return np.array([n.is_bifurcation() for n in self._features.tree])


class TipFeatures(_SubsetNodesFeatures):
    """Evaluate tip node feature of tree."""

    @cached_property
    def nodes(self) -> npt.NDArray[np.bool_]:
        return np.array([n.is_tip() for n in self._features.tree])
