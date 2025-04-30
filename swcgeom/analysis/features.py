
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Feature analysis of tree."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, deprecated

from swcgeom.core import Branch, BranchTree, Tree

__all__ = [
    "NodeFeatures",
    "BifurcationFeatures",
    "TipFeatures",
    "PathFeatures",
    "BranchFeatures",
]

T = TypeVar("T", bound=Branch)

# Node Level


class NodeFeatures:
    """Evaluate node feature of tree."""

    tree: Tree

    @cached_property
    def _branch_tree(self) -> BranchTree:
        return BranchTree.from_tree(self.tree)

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> npt.NDArray[np.float32]:
        """Get number of nodes.

        Args:
            count: array of shape (1,)
        """
        return np.array([self.tree.number_of_nodes()], dtype=np.float32)

    def get_radial_distance(self) -> npt.NDArray[np.float32]:
        """Get the end-to-end straight-line distance to soma.

        Returns:
            radial_distance: Array of shape (N,).
        """
        xyz = self.tree.xyz() - self.tree.soma().xyz()
        radial_distance = np.linalg.norm(xyz, axis=1)
        return radial_distance

    def get_branch_order(self) -> npt.NDArray[np.int32]:
        """Get branch order of criticle nodes of tree.

        Branch order is the number of bifurcations between current
        position and the root.

        Criticle node means that soma, bifucation nodes, tips.

        Returns:
            order: Array of shape (N,), which k is the number of branches.
        """
        order = np.zeros_like(self._branch_tree.id(), dtype=np.int32)

        def assign_depth(n: Tree.Node, pre_depth: int | None) -> int:
            cur_order = pre_depth + 1 if pre_depth is not None else 0
            order[n.id] = cur_order
            return cur_order

        self._branch_tree.traverse(enter=assign_depth)
        return order


class _SubsetNodesFeatures(ABC):
    _features: NodeFeatures

    @property
    @abstractmethod
    def nodes(self) -> npt.NDArray[np.bool_]:
        raise NotImplementedError()

    def __init__(self, features: NodeFeatures) -> None:
        self._features = features

    def get_count(self) -> npt.NDArray[np.float32]:
        """Get number of nodes.

        Returns:
            count: Array of shape (1,).
        """
        return np.array([np.count_nonzero(self.nodes)], dtype=np.float32)

    def get_radial_distance(self) -> npt.NDArray[np.float32]:
        """Get the end-to-end straight-line distance to soma.

        Returns:
            radial_distance: Array of shape (N,).
        """
        return self._features.get_radial_distance()[self.nodes]

    @classmethod
    def from_tree(cls, tree: Tree) -> Self:
        return cls(NodeFeatures(tree))


class FurcationFeatures(_SubsetNodesFeatures):
    """Evaluate furcation node feature of tree."""

    @cached_property
    def nodes(self) -> npt.NDArray[np.bool_]:
        return np.array([n.is_furcation() for n in self._features.tree])


@deprecated("Use FurcationFeatures instead")
class BifurcationFeatures(FurcationFeatures):
    """Evaluate bifurcation node feature of tree.

    NOTE: Deprecated due to the wrong spelling of furcation. For now, it is just an
    alias of `FurcationFeatures` and raise a warning. It will be change to raise an
    error in the future.
    """


class TipFeatures(_SubsetNodesFeatures):
    """Evaluate tip node feature of tree."""

    @cached_property
    def nodes(self) -> npt.NDArray[np.bool_]:
        return np.array([n.is_tip() for n in self._features.tree])


# Path Level


class PathFeatures:
    """Path analysis of tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> int:
        return len(self._paths)

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of paths."""

        length = [path.length() for path in self._paths]
        return np.array(length, dtype=np.float32)

    def get_tortuosity(self) -> npt.NDArray[np.float32]:
        """Get tortuosity of path."""

        return np.array([path.tortuosity() for path in self._paths], dtype=np.float32)

    @cached_property
    def _paths(self) -> list[Tree.Path]:
        return self.tree.get_paths()


class BranchFeatures:
    """Analysis bransh of tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> int:
        return len(self._branches)

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of branches."""

        length = [br.length() for br in self._branches]
        return np.array(length, dtype=np.float32)

    def get_tortuosity(self) -> npt.NDArray[np.float32]:
        """Get tortuosity of path."""

        return np.array([br.tortuosity() for br in self._branches], dtype=np.float32)

    def get_angle(self, eps: float = 1e-7) -> npt.NDArray[np.float32]:
        """Get agnle between branches.

        Returns:
            angle: An array of shape (N, N), which N is length of branches.
        """
        return self.calc_angle(self._branches, eps=eps)

    @staticmethod
    def calc_angle(branches: list[T], eps: float = 1e-7) -> npt.NDArray[np.float32]:
        """Calc agnle between branches.

        Returns:
            angle: An array of shape (N, N), which N is length of branches.
        """
        vector = np.array([br[-1].xyz() - br[0].xyz() for br in branches])
        vector_dot = np.matmul(vector, vector.T)
        vector_norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
        vector_norm_dot = np.matmul(vector_norm, vector_norm.T) + eps
        arccos = np.clip(vector_dot / vector_norm_dot, -1, 1)
        angle = np.arccos(arccos)
        return angle

    @cached_property
    def _branches(self) -> list[Tree.Branch]:
        return self.tree.get_branches()
