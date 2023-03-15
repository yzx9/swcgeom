"""Transformation in tree."""

import warnings
from typing import Callable, List, Optional, Tuple

from ..core import BranchTree, Tree, cut_tree, to_subtree
from .base import Transform
from .geometry import Normalizer

__all__ = [
    "ToBranchTree",
    "TreeNormalizer",
    "CutByBifurcationOrder",
    "CutShortTipBranch",
]


# pylint: disable=too-few-public-methods
class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)


class TreeNormalizer(Normalizer[Tree]):
    """Noramlize coordinates and radius to 0-1."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "`TreeNormalizer` has been deprecate, it is replaced by "
            "`Normalizer` beacuse it applies more widely.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class CutByBifurcationOrder(Transform[Tree, Tree]):
    """Cut tree by bifurcation order."""

    max_bifurcation_order: int

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_bifurcation_order = max_bifurcation_order

    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self._enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_bifurcation_order}"

    def _enter(self, n: Tree.Node, parent_level: int | None) -> Tuple[int, bool]:
        if parent_level is None:
            level = 0
        elif n.is_bifurcation():
            level = parent_level + 1
        else:
            level = parent_level
        return (level, level >= self.max_bifurcation_order)


class CutShortTipBranch(Transform[Tree, Tree]):
    """Cut off too short terminal branches.

    This method is usually applied in the post-processing of manual
    reconstruction. When the user draw lines, a line head is often left
    at the junction of two lines.
    """

    thre: float
    callbacks: List[Callable[[Tree.Branch], None]]

    def __init__(
        self, thre: float = 5, callback: Optional[Callable[[Tree.Branch], None]] = None
    ) -> None:
        self.thre = thre
        self.callbacks = []

        if callback is not None:
            self.callbacks.append(callback)

    def __repr__(self) -> str:
        return f"CutShortTipBranch-{self.thre}"

    def __call__(self, x: Tree) -> Tree:
        removals: List[int] = []
        self.callbacks.append(lambda br: removals.append(br[1].id))
        x.traverse(leave=self._leave)
        self.callbacks.pop()
        return to_subtree(x, removals)

    def _leave(
        self, n: Tree.Node, children: List[Tuple[float, Tree.Node] | None]
    ) -> Tuple[float, Tree.Node] | None:
        if len(children) == 0:  # tip
            return 0, n

        if len(children) == 1 and children[0] is not None:  # elongation
            dis, child = children[0]
            dis += n.distance(child)
            return dis, n

        for c in children:
            if c is None:
                continue

            dis, child = c
            if dis + n.distance(child) > self.thre:
                continue

            path = [n.id]  # n does not delete, but will include in callback
            while child is not None:  # TODO: perf
                path.append(child.id)
                child = cc[0] if len((cc := child.children())) > 0 else None

            br = Tree.Branch(n.attach, path)
            for cb in self.callbacks:
                cb(br)

        return None
