from ... import BranchTree
from .tree_folder_dataset import TreeFolderDataset


class BranchTreeFolderDataset(TreeFolderDataset):
    """Branch tree folder in swc format."""

    def __getitem__(self, idx: int) -> tuple[BranchTree, int]:
        """Get a branch tree.

        Returns
        -------
        x : BranchTree
            A branch tree from swc format.
        y : int
            Label of x.
        """
        x, y = super().__getitem__(idx)
        return BranchTree.from_tree(x), y
