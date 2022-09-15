"""Transformation in tree."""

import numpy as np

from ..core import BranchTree, Tree
from .base import Transform

__all__ = ["TreeToBranchTree", "TreeNormalizer"]


class TreeToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)


class TreeNormalizer(Transform[Tree, Tree]):
    """Noramlize coordinates and radius to 0-1."""

    def __call__(self, x: Tree) -> "Tree":
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1."""
        new_tree = x.copy()
        for key in ["x", "y", "z", "r"]:  # TODO: does r is the same?
            v_max = np.max(new_tree.ndata[key])
            v_min = np.min(new_tree.ndata[key])
            new_tree.ndata[key] = (new_tree.ndata[key] - v_min) / v_max

        return new_tree
