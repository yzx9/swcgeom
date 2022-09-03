"""Transformation in tree."""

from ..core import BranchTree, Tree
from .base import Transform

__all__ = ["ToBranchTree"]


class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)
