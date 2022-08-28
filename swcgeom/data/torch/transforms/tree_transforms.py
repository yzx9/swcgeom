from ....core import BranchTree, Tree
from ...transforms import Transform


class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __init__(self) -> None:
        super().__init__("branchtree")

    def apply(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)
