from swcgeom.core.tree import Tree
from swcgeom.data.torch.transforms.tree_transforms import ToBranchTree
from swcgeom.data.transforms import Transforms

from ... import BranchTree
from .tree_folder_dataset import TreeFolderDataset


# TODO: remove in 0.2.0
class BranchTreeFolderDataset(TreeFolderDataset):
    """Branch tree folder in swc format.

    .. deprecated:: 0.1.7
        `BranchTreeFolderDataset` will be removed in swcgeom 0.2.0, it is
        replaced by `TreeFolderDataset(self.swc_dir, transforms=transforms)`
        because the latter provides a more flexible approach.
    """

    def __init__(self, swc_dir: str) -> None:
        """Create tree dataset.

        Parameters
        ----------
        swc_dir : str
            Path of SWC file directory.
        """
        transforms = Transforms[Tree, BranchTree](ToBranchTree())
        super().__init__(swc_dir, transforms=transforms)

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
