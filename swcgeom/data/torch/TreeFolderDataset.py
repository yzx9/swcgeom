import os

import torch.utils.data

from ... import Tree


class TreeFolderDataset(torch.utils.data.Dataset):
    """Tree folder in swc format."""

    swc_dir: str
    swcs: list[str]

    def __init__(self, swc_dir: str) -> None:
        """Create tree dataset.

        Parameters
        ==========
        swc_dir : str.
            Path of SWC file directory.
        """
        self.swc_dir = swc_dir
        self.swcs = self.find_swcs(swc_dir)

    def __len__(self) -> int:
        """Get length of set of trees."""
        return len(self.swcs)

    def __getitem__(self, idx: int) -> tuple[Tree, int]:
        """Get a tree.

        Returns
        =======
        x : Tree.
            A tree from swc format.
        y : int.
            Label of x.
        """
        return Tree.from_swc(self.swcs[idx]), 0

    @staticmethod
    def find_swcs(swc_dir: str) -> list[str]:
        """Find all swc files."""
        swcs = list[str]()
        for root, dirs, files in os.walk(swc_dir):
            files = [f for f in files if os.path.splitext(f)[-1] == ".swc"]
            swcs.extend([os.path.join(root, f) for f in files])

        return swcs
