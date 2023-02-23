"""Transformation in tree."""

from typing import Callable, List, Tuple

import numpy as np

from ..core import REMOVE, BranchTree, Tree, cut_tree, propagate_remove, to_sub_tree
from .base import Transform

__all__ = [
    "ToBranchTree",
    "TreeNormalizer",
    "CutByBifurcationOrder",
    "CutShortTipBranch",
]


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

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_bifurcation_order = max_bifurcation_order

    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self._enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_bifurcation_order}"

    def _enter(self, n: Tree.Node, parent_level: int | None) -> tuple[int, bool]:
        if parent_level is None:
            level = 0
        elif n.is_bifurcation():
            level = parent_level + 1
        else:
            level = parent_level
        return (level, level >= self.max_bifurcation_order)


class CutShortTipBranch(Transform[Tree, Tree]):
    thre: float
    callback: Callable[[Tree.Branch], None] | None

    def __init__(
        self, thre: float = 5, callback: Callable[[Tree.Branch], None] | None = None
    ) -> None:
        self.thre = thre
        self.callback = callback

    def __repr__(self) -> str:
        return f"CutShortTipBranch-{self.thre}"

    def __call__(self, x: Tree) -> Tree:
        new_id = x.id().copy()

        def collect_short_branch(
            n: Tree.Node, children: List[Tuple[float, Tree.Node] | None]
        ) -> Tuple[float, Tree.Node] | None:
            if len(children) == 0:  # tip
                return 0, n

            if len(children) == 1:
                if children[0] is None:
                    return None

                dis, child = children[0]
                dis += n.distance(child)
                return dis, n

            for c in children:
                if c is None:
                    continue

                dis, child = c
                if dis + n.distance(child) <= self.thre:
                    new_id[child.id] = REMOVE  # TODO: change this to a callback

                    if self.callback is not None:
                        path = [n.id]  # n does not delete, but will include in callback
                        while child is not None:
                            path.append(child.id)
                            child = cc[0] if len((cc := child.children())) > 0 else None
                        self.callback(Tree.Branch(n.attach, path))

            return None

        x.traverse(leave=collect_short_branch)
        propagate_remove(x, new_id)
        new_tree, _ = to_sub_tree(x, new_id, x.pid().copy())
        return new_tree
