from ....core import BranchTree, Tree
from ...transforms import Transform


class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)

    def get_name(self) -> str:
        return f"branchtree"
