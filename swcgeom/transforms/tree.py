"""Transformation in tree."""

from typing import Callable

import numpy as np

from ..core import BranchTree, Tree, cut_tree
from .base import Transform

__all__ = ["ToBranchTree", "TreeNormalizer", "CutByBifurcationOrder"]


class ToBranchTree(Transform[Tree, BranchTree]):
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


class CutByBifurcationOrder(Transform[Tree, Tree]):
    """Cut tree by bifurcation order"""

    max_bifurcation_order: int
    enter: Callable[[Tree.Node, int | None], tuple[int, bool]]

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_bifurcation_order = max_bifurcation_order

        def enter(n: Tree.Node, parent_level: int | None) -> tuple[int, bool]:
            if parent_level is None:
                level = 0
            elif n.is_bifurcation():
                level = parent_level + 1
            else:
                level = parent_level
            return (level, level >= self.max_bifurcation_order)

        self.enter = enter

    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self.enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_bifurcation_order}"
